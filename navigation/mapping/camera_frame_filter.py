"""
Step 4: Camera frame filter (former Azure Kinect interface, processing part).

Subscribes the completed camera frame of depth_completion/depth_completion_node.py (or, with
--depth_source raw / --raw, the raw frame of ln_ros_camera_bridge.py), filters it
and publishes:

    /tf                                map -> Wt (odometry at the frame timestamp),
                                       Wt -> ground, Wt -> camera
    /camera/filtered/depth/image_raw   filtered depth (16UC1, mm)  -> nvblox
    /camera/filtered/points            filtered pointcloud (camera frame) -> octomap / collision map

Filters: min/max range, flying pixels at depth edges, humans (mediapipe, --mask_humans),
robot arm (--filter_robot).

Depth source:
    --depth_source completed   MoGe-filled depth from /camera/completed/depth/image_raw (default)
    --depth_source raw         original depth from ln_ros_camera_bridge.py (/camera/depth/image_raw)
    --raw                      shortcut for --depth_source raw (bypasses MoGe completely)

Pointcloud output is only published with --publish_pointcloud (off by default). The
filtered depth is still published because nvblox integrates depth images, not the cloud.

The frame's header.stamp is the original LN capture timestamp. It is used to look up
odometry and joint angles in the SynchronizationBuffer, and all outputs (incl. TF)
are stamped with it, so nvblox / octomap find the camera pose of exactly this frame.

Usage:
    python3 -m navigation.mapping.camera_frame_filter --mask_humans --filter_robot
"""
#Debug purpose
import os, sys
current_dir = os.path.dirname(__file__)
main_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.insert(0, main_dir)

import argparse
import time

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from message_filters import Subscriber, TimeSynchronizer
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2

import links_and_nodes as ln
from robotutils.geom.se3 import euler2tf, simulink2tf

from navigation.mapping.arm_filter import SelfFilter, depth_to_pointcloud
from navigation.mapping.ros_utils import tf_from_matrix, to_sec
from navigation.mapping.synchronization_buffer import SynchronizationBuffer_LnInterface

MAX_DIST = 80000        #in mm
MIN_DIST = 0#820      # in mm
POINTCLOUD_DOWNSAMPLING = 1.0
HUMAN_PADDING = 100 #px
NOINFO_BEHAVIOUR = 0 #65535 #0
EDGE_REL_THRESH = 0.1  # flying pixels: depth range in 3x3 > 10 % of the depth -> drop, 0 = off

INPUT_TOPICS = {
    "completed": ("camera/completed/color/image_raw", "camera/completed/depth/image_raw",
                  "camera/completed/camera_info"),
    "raw": ("camera/color/image_raw", "camera/depth/image_raw", "camera/color/camera_info"),
}


def flying_pixel_mask(depth):
    """Pixels at depth discontinuities (mixed foreground/background, "flying pixels").

    A pixel is dropped if the depth range (max - min) in its 3x3 neighbourhood exceeds
    EDGE_REL_THRESH times its own depth. Holes (0) are ignored as neighbours.
    Grazing surfaces (floor at 8 m: ~4 % per 2 px) stay well below the threshold.
    """
    valid = depth > 0
    if EDGE_REL_THRESH <= 0:
        return np.zeros(depth.shape, dtype=bool)
    kernel = np.ones((3, 3), np.uint8)
    dmax = cv2.dilate(depth, kernel)
    dmin = cv2.erode(np.where(valid, depth, np.uint16(65535)), kernel)
    return valid & ((dmax.astype(np.float32) - dmin) > EDGE_REL_THRESH * depth)


def camera_params_from_info(info: CameraInfo) -> dict:
    return {
        "fx": info.k[0],
        "fy": info.k[4],
        "cx": info.k[2],
        "cy": info.k[5],
        "width": info.width,
        "height": info.height,
    }


class CameraFrameFilter:
    def __init__(self, node, args):
        self.node = node
        self.mask_humans = args.mask_humans
        self.filter_robot = args.filter_robot
        self.publish_pointcloud = args.publish_pointcloud
        self.bridge = CvBridge()

        self.frame_global = "map"
        self.frame_base = "Wt"
        self.frame_cam = "camera"
        self.frame_ground = "ground"

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self.node)

        self.ln_clnt = ln.client("CameraFrameFilter", [])
        self.ln_inport_W_T_Cam = self.ln_clnt.subscribe("user_config_handler.user_info", "python_modules/periphery/user_info")
        self.sync_buffer = SynchronizationBuffer_LnInterface(self.ln_clnt)

        self.node.get_logger().info("Read W_T_CAM...")
        self.ln_inport_W_T_Cam.read()
        self.Wt_T_Cam = self.ln_inport_W_T_Cam.packet.W_T_CAM.reshape((4, 4))

        if self.mask_humans:
            import mediapipe as mp
            self.pose = mp.solutions.holistic.Holistic(static_image_mode=False)

        self.c = None            # camera parameters, set from the first camera_info
        self.arm_filter = None   # needs self.c

        self.pub_depth = self.node.create_publisher(Image, "camera/filtered/depth/image_raw", 1)
        self.pub_pc = None
        if self.publish_pointcloud:
            self.pub_pc = self.node.create_publisher(PointCloud2, "camera/filtered/points", 1)

        # The raw and completed topics belong to different frames. Exact time sync only
        # works within one source, so --depth_source raw switches all three topics.
        source = "raw" if (args.raw or args.depth_source == "raw") else "completed"
        color_topic, depth_topic, info_topic = INPUT_TOPICS[source]
        subs = [Subscriber(self.node, Image, color_topic, qos_profile=qos_profile_sensor_data),
                Subscriber(self.node, Image, depth_topic, qos_profile=qos_profile_sensor_data),
                Subscriber(self.node, CameraInfo, info_topic, qos_profile=qos_profile_sensor_data)]
        self.sync = TimeSynchronizer(subs, queue_size=5)   # identical stamps per frame
        self.sync.registerCallback(self.on_frame)

        self._last_update = time.time()
        self.node.get_logger().info(
            f"Init done, depth source '{source}', pointcloud "
            f"{'ENABLED -> /camera/filtered/points' if self.publish_pointcloud else 'DISABLED (no --publish_pointcloud)'}, "
            f"waiting for '{depth_topic}'...")

    def publish_odometry(self, ts, stamp):
        Ws_T_Wt = simulink2tf(self.sync_buffer.get_closest_odom_for_t(ts))
        self.tf_broadcaster.sendTransform([
            tf_from_matrix(Ws_T_Wt, self.frame_global, self.frame_base, stamp),
            tf_from_matrix(euler2tf([0,0,-0.58,0,0,0]), self.frame_base, self.frame_ground, stamp),
            tf_from_matrix(self.Wt_T_Cam, self.frame_base, self.frame_cam, stamp), #TODO check why static broadcaster doesnt work
        ])

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

    def filter_depth(self, rgb, depth, ts):
        noinfo_mask = (depth == 0)
        ignore_mask = np.zeros(depth.shape, dtype=bool)

        mask_near = (depth > 0) & (depth <= MIN_DIST)
        mask_far  = (depth > 0) & (depth >= MAX_DIST)
        ignore_mask |= (mask_far | mask_near)
        depth[mask_far | mask_near] = 0

        edge_mask = flying_pixel_mask(depth)
        ignore_mask |= edge_mask
        depth[edge_mask] = 0

        if self.mask_humans:
            depth, human_mask = self.mask_human(rgb, depth)
            ignore_mask |= human_mask

        if self.filter_robot:
            depth, arm_mask = self.arm_filter.get_masked_depth(depth, ts)
            ignore_mask |= arm_mask

        depth[(noinfo_mask) & (~ignore_mask) & (depth == 0)] = NOINFO_BEHAVIOUR
        depth[(ignore_mask)] = 0
        return depth

    def on_frame(self, color_msg, depth_msg, info_msg):
        if self.c is None:
            self.c = camera_params_from_info(info_msg)
            if self.filter_robot:
                self.arm_filter = SelfFilter(self.c, self.Wt_T_Cam, self.sync_buffer)

        header = depth_msg.header             # original LN timestamp, frame "camera"
        ts = to_sec(header.stamp)
        self.publish_odometry(ts, header.stamp)

        rgb = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1").copy()
        depth = self.filter_depth(rgb, depth, ts)

        out_depth = self.bridge.cv2_to_imgmsg(depth, encoding="16UC1")
        out_depth.header = header
        self.pub_depth.publish(out_depth)

        if self.publish_pointcloud:
            pointcloud, _ = depth_to_pointcloud(depth, self.c, ds=POINTCLOUD_DOWNSAMPLING)
            self.pub_pc.publish(point_cloud2.create_cloud_xyz32(header, pointcloud))

        now = time.time()
        self.node.get_logger().info(
            f"Updated with: {round(1/max(now-self._last_update, 1e-6),2)}Hz, "
            f"latency {now - ts:.3f}s", throttle_duration_sec=5.0)
        self._last_update = now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_humans", action="store_true", help="Mask humans")
    parser.add_argument("--filter_robot", action="store_true", help="Mask the robot arm")
    parser.add_argument("--publish_pointcloud", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Publish /camera/filtered/points (default: off; enable with "
                             "--publish_pointcloud)")
    parser.add_argument("--depth_source", choices=("completed", "raw"), default="completed",
                        help="'completed' = MoGe-filled depth, 'raw' = original depth from "
                             "ln_ros_camera_bridge.py")
    parser.add_argument("--raw", action="store_true",
                        help="Shortcut for --depth_source raw (use the raw camera topics)")
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("camera_frame_filter")
    CameraFrameFilter(node=node, args=args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
