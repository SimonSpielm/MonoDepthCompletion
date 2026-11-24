#Debug purpose
import os, sys
current_dir = os.path.dirname(__file__)
main_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.insert(0, main_dir)


import numpy as np
import copy, time, argparse

import rclpy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header as StdHeader
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge


import links_and_nodes as ln
from robotutils.geom.se3 import euler2tf, simulink2tf
from robotutils.geom.so3 import rotmat2quat

from navigation.mapping.arm_filter import SelfFilter
from navigation.mapping.synchronization_buffer import SynchronizationBuffer_LnInterface

HZ = 15
MAX_VALID = 15000
MAX_DIST = 8000        #in mm
MIN_DIST = 0#820      # in mm
POINTCLOUD_DOWNSAMPLING = 1.0
HUMAN_PADDING = 100 #px
NOINFO_BEHAVIOUR = 0 #65535 #0



class AzureKinectNode:
    def __init__(self, node, args):

        self.node = node

        self.use_ln = args.use_ln
        self.use_rosbag = args.use_rosbag if args.use_ln is False else False
        self.mask_humans = args.mask_humans
        self.filter_robot = args.filter_robot
        self.publish_pointcloud = args.publish_pointcloud

        self.setup_ros_publishers()
        self.bridge = CvBridge()

        self.frame_global = "map"
        self.frame_base = "Wt"
        self.frame_cam = "camera"
        self.frame_ground = "ground"

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self.node)
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self.node)

        self.ln_clnt = ln.client("MapPublisher", [])
        self.ln_inport_W_T_Cam = self.ln_clnt.subscribe("user_config_handler.user_info", "python_modules/periphery/user_info")
        self.odom_buffer = SynchronizationBuffer_LnInterface(self.ln_clnt)

        self.node.get_logger().info("Read W_T_CAM...")
        self.ln_inport_W_T_Cam.read()
        self.Wt_T_Cam = self.ln_inport_W_T_Cam.packet.W_T_CAM.reshape((4, 4))

        self._last_update = time.time()
        self._last_ts = time.time()

        if not self.use_rosbag:
            if self.use_ln:
                self.setup_ln()
                self.color_info_msg = self.get_camera_info_from_camk(self.frame_cam) #TODO check if color and depth intrinsic are same
            else:
                self.setup_device()
                self.color_info_msg = self.get_camera_info(self.frame_cam, "color")
            self.depth_info_msg = self.color_info_msg
            
            self.k = np.array(self.color_info_msg.k).reshape([3,3])

            if self.mask_humans:
                import mediapipe as mp
                self.pose = mp.solutions.holistic.Holistic(static_image_mode=False)
            if self.filter_robot:
                self.arm_filter = SelfFilter(self.c, POINTCLOUD_DOWNSAMPLING)

        self.node.get_logger().info("Init done...")

    def setup_ln(self):
        self.ln_inport_rgb = self.ln_clnt.subscribe("camera.perception/rgb", "perception/rgb")
        self.ln_inport_depth = self.ln_clnt.subscribe("camera.perception/depth", "perception/depth")
        self.ln_inport_camk = self.ln_clnt.subscribe("camera.perception/cam_k", "perception/cam_k")
        

    def setup_device(self):
        import k4a
        self.node.get_logger().info("Opening Azure Kinect device...")
        self.device = k4a.Device.open(0)

        device_config = k4a.DeviceConfiguration(
            color_format=k4a.EImageFormat.COLOR_BGRA32,
            color_resolution=k4a.EColorResolution.RES_720P,
            depth_mode=k4a.EDepthMode.WFOV_2X2BINNED,
            camera_fps=k4a.EFramesPerSecond.FPS_15,
            synchronized_images_only=True
        )

        if self.device.start_cameras(device_config) != k4a.EStatus.SUCCEEDED:
            print("Error, check ROS log ")
            raise IOError("Failed to start Azure Kinect cameras.")

        self.calibration = self.device.get_calibration(
            depth_mode=device_config.depth_mode,
            color_resolution=device_config.color_resolution
        )

        self.transform = k4a.Transformation(self.calibration)
        self.node.get_logger().info("Kinect device setup complete.")

    def setup_ros_publishers(self):
        self.pub_color = self.node.create_publisher(Image, "camera/color/image_raw", 1)
        self.pub_depth = self.node.create_publisher(Image, "camera/depth/image_raw", 1)
        self.pub_color_info = self.node.create_publisher(CameraInfo, "camera/color/camera_info", 1)
        self.pub_depth_info = self.node.create_publisher(CameraInfo, "camera/depth/camera_info", 1)
        self.pub_pc = self.node.create_publisher(PointCloud2, "camera/depth/points", 1)

    def publish_odometry(self, ts):
        stamp = self.node.get_clock().now().to_msg()
        if self.use_rosbag:
            Ws_T_Wt = euler2tf([0,0,0,0,0,0])
            self.publish_tf_matrix(Ws_T_Wt, self.frame_global, self.frame_base, self.tf_broadcaster, stamp)
        else:
            Ws_T_Wt = simulink2tf(self.odom_buffer.get_closest_odom_for_t(ts))
            self.publish_tf_matrix(Ws_T_Wt, self.frame_global, self.frame_base, self.tf_broadcaster, stamp)
        self.publish_tf_matrix(euler2tf([0,0,-0.58,0,0,0]), self.frame_base, self.frame_ground, self.tf_broadcaster, stamp)
        self.publish_tf_matrix(self.Wt_T_Cam, self.frame_base, self.frame_cam, self.tf_broadcaster, stamp) #TODO check why static broadcaster doesnt work
        return stamp

    def publish_tf_matrix(self, T, parent_frame, child_frame, broadcaster, stamp=None):
        if stamp is None:
            stamp = self.node.get_clock().now().to_msg()
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = parent_frame
        tf_msg.child_frame_id = child_frame
        tf_msg.transform.translation.x = float(T[0, 3])
        tf_msg.transform.translation.y = float(T[1, 3])
        tf_msg.transform.translation.z = float(T[2, 3])

        R_mat = T[:3, :3]
        q = rotmat2quat(R_mat)
        tf_msg.transform.rotation.x = q[0]
        tf_msg.transform.rotation.y = q[1]
        tf_msg.transform.rotation.z = q[2]
        tf_msg.transform.rotation.w = q[3]

        broadcaster.sendTransform(tf_msg)

    def get_camera_info_from_camk(self, frame_id: str) -> CameraInfo:
        # Read intrinsics (row-major 3x3)
        self.ln_inport_camk.read()
        K = np.asarray(self.ln_inport_camk.packet.data, dtype=float).reshape(3, 3)

        # Store camera parameters (derived, no duplication)
        self.c = {
            "fx": K[0, 0],
            "fy": K[1, 1],
            "cx": K[0, 2],
            "cy": K[1, 2],
            "width": 1280,
            "height": 720,
            "depth_scale": 0.001,
        }

        info = CameraInfo()
        info.header.frame_id = frame_id
        info.width = self.c["width"]
        info.height = self.c["height"]

        # Distortion (assumed rectified / unknown)
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5

        # Intrinsics
        info.k = K.ravel().tolist()

        # Rectification
        info.r = np.eye(3, dtype=float).ravel().tolist()

        # Projection matrix [K | 0]
        P = np.zeros((3, 4), dtype=float)
        P[:3, :3] = K
        info.p = P.ravel().tolist()

        return info



    def get_camera_info(self, frame_id, camera_type="color"):
        cam_cal = (
            self.calibration.color_cam_cal
            if camera_type == "color"
            else self.calibration.depth_cam_cal
        )
        intr = cam_cal.intrinsics.parameters.param

        fx, fy, cx, cy = map(float, (intr.fx, intr.fy, intr.cx, intr.cy))
        k1, k2, p1, p2, k3 = map(float, (intr.k1, intr.k2, intr.p1, intr.p2, intr.k3))
        width, height = int(cam_cal.resolution_width), int(cam_cal.resolution_height)

        info = CameraInfo()
        info.header.frame_id = frame_id
        info.width = width
        info.height = height
        info.distortion_model = "plumb_bob"
        info.d = [k1, k2, p1, p2, k3]

        self.c = {
            "cx": cx,
            "cy": cy,
            "depth_scale": 0.001,
            "fx": fx,
            "fy": fy,
            "height": height,
            "width": width,
        }

        K = np.array([[fx, 0,  cx],
                    [0,  fy, cy],
                    [0,  0,  1]], dtype=float)
        R = np.eye(3, dtype=float)
        P = np.array([[fx, 0,  cx, 0],
                    [0,  fy, cy, 0],
                    [0,  0,  1,  0]], dtype=float)

        info.k = K.ravel().tolist()
        info.r = R.ravel().tolist()
        info.p = P.ravel().tolist()

        return info


    def get_rgbd_ln(self):
        self.ln_inport_rgb.read(-1)
        self.ln_inport_depth.read(-1)
        packet_rgb = self.ln_inport_rgb.packet
        packet_depth = self.ln_inport_depth.packet
        if packet_rgb is None or packet_depth is None:
            return None, None, None
        rgb = np.array(packet_rgb.data).reshape(packet_rgb.height, packet_rgb.width, 3).astype("uint8")
        depth = np.array(packet_depth.data).reshape(packet_depth.height, packet_depth.width).astype("uint16")
        time_ts = packet_depth.timestamp
        return rgb, depth, time_ts


    def get_rgbd_k4a(self):
        capture = self.device.get_capture(0)
        if not capture:
            return None, None, None
        bgra_image = copy.deepcopy(capture.color.data)
        bgr = bgra_image[..., :3]
        rgb = bgr[..., ::-1]
        depth = copy.deepcopy(self.transform.depth_image_to_color_camera(capture.depth).data)
        return rgb, depth, time.time()


    def mask_human(self, rgb, depth):   # returns (depth, mask)
        h, w = rgb.shape[:2]
        results = self.pose.process(rgb)

        human_mask = np.zeros(depth.shape, dtype=bool)

        if not results.pose_landmarks:
            return depth, human_mask

        keypoints = results.pose_landmarks.landmark
        xs = [int(p.x * w) for p in keypoints if p.visibility > 0.7]
        ys = [int(p.y * h) for p in keypoints if p.visibility > 0.7]

        if len(xs) == 0 or len(ys) == 0:
            return depth, human_mask

        min_x = max(min(xs) - HUMAN_PADDING, 0)
        max_x = min(max(xs) + HUMAN_PADDING, w)
        min_y = max(min(ys) - HUMAN_PADDING, 0)
        max_y = min(max(ys) + HUMAN_PADDING, h)

        human_mask[min_y:max_y, min_x:max_x] = True

        depth[human_mask] = 0
        return depth, human_mask

    def capture(self):
        if self.use_ln:
            rgb, depth, time_ts = self.get_rgbd_ln()
        else:
            rgb, depth, time_ts = self.get_rgbd_k4a()

        if depth is None:
            self.node.get_logger().warning("Failed to capture RGB or depth.")
            return
        return rgb, depth, time_ts


    def publish_rgbd(self, rgb, depth, time=None):

        if time is None:
            time = self.node.get_clock().now().to_msg()
        stamp = time
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="16UC1")
        rgb_msg.header.stamp = stamp
        depth_msg.header.stamp = stamp
        rgb_msg.header.frame_id = self.frame_cam
        depth_msg.header.frame_id = self.frame_cam

        self.pub_color.publish(rgb_msg)
        self.pub_depth.publish(depth_msg)
        self.pub_color_info.publish(self.color_info_msg)
        self.pub_depth_info.publish(self.depth_info_msg)
        self.node.get_logger().info("Published RGB and depth.")


    def pub_pointcloud(self, pointcloud, time=None):
        if time is None:
            time = self.node.get_clock().now().to_msg()
        header = StdHeader()
        header.stamp = time
        header.frame_id = self.frame_cam
        cloud_msg = point_cloud2.create_cloud_xyz32(header, pointcloud)
        self.pub_pc.publish(cloud_msg)


    def main(self):
        self.node.get_logger().info("Starting Azure Kinect publishing...")
        
        def on_timer():
            rgb, depth, ts = self.capture()
            if ts == self._last_ts:
                print("skipped")
                return

            if rgb is not None and depth is not None:
                self.process_rgb_depth(rgb, depth, ts)
                self.node.get_logger().info(f"Updated with: {round(1/(time.time()-self._last_update),2)}Hz")
                self._last_update = time.time()
                self._last_ts = ts

        self.node.create_timer(1.0 / float(HZ), on_timer)
        rclpy.spin(self.node)


    def process_rgb_depth(self,rgb, depth, ts):
        if not self.use_rosbag:
            odom_ros_time = self.publish_odometry(ts)

            noinfo_mask = (depth == 0)
            ignore_mask = np.zeros(depth.shape, dtype=bool)


            mask_near = (depth > 0) & (depth <= MIN_DIST)
            mask_far  = (depth > 0) & (depth >= MAX_DIST)
            ignore_mask |= (mask_far | mask_near)
            depth[mask_far | mask_near] = 0

            if self.mask_humans:
                depth, human_mask = self.mask_human(rgb, depth)
                ignore_mask |= human_mask

            if self.filter_robot:
                depth, arm_mask = self.arm_filter.get_masked_depth(depth, ts)
                ignore_mask |= arm_mask

            depth[(noinfo_mask) & (~ignore_mask) & (depth == 0)] = NOINFO_BEHAVIOUR
            depth[(ignore_mask)] = 0

            self.publish_rgbd(rgb, depth, time=odom_ros_time)

            if self.publish_pointcloud:
                pointcloud, mask = self.arm_filter.depth_to_pointcloud(depth=depth, ds=POINTCLOUD_DOWNSAMPLING)
                self.pub_pointcloud(pointcloud, time=odom_ros_time)

    def shutdown(self):
        if not self.use_ln:
            self.node.get_logger().info("Shutting down Kinect device...")
            self.device.stop_cameras()
            self.device.close()
            self.node.get_logger().info("Device closed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_ln', action='store_true', help='Camera stream from ln? Default:k4a')
    parser.add_argument("--use_rosbag", action="store_true", help="Camera data from rosbag")
    parser.add_argument("--mask_humans", action="store_true", help="Mask humans")
    parser.add_argument("--publish_pointcloud", action="store_true", help="Publish filtered pointcloud")
    parser.add_argument("--filter_robot", action="store_true", help="Publish filtered pointcloud")
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("MapPublisher")
    azure = AzureKinectNode(node=node, args=args)
    try:
        azure.main()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        azure.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()