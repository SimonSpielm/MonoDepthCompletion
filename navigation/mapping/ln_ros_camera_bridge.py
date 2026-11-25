"""
Step 2: LN -> ROS camera bridge.

Reads the LN camera topics (from the real camera driver or ln_camera_publisher.py)
and publishes them unchanged as ROS messages. All messages of one frame carry the
LN capture timestamp as header.stamp; it is passed through the whole pipeline.

    LN camera.perception/rgb    -> /camera/color/image_raw    (rgb8)
    LN camera.perception/depth  -> /camera/depth/image_raw    (16UC1, mm)
    LN camera.perception/cam_k  -> /camera/color/camera_info, /camera/depth/camera_info

No timer: inport.read() blocks until the next packet, so frames are forwarded as
soon as they arrive.

Usage:
    python3 -m navigation.mapping.ln_ros_camera_bridge
"""
#Debug purpose
import os, sys
current_dir = os.path.dirname(__file__)
main_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.insert(0, main_dir)

import time

import links_and_nodes as ln
import numpy as np
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

from navigation.mapping.ros_utils import to_ros_time

FRAME_CAM = "camera"


class LnRosCameraBridge:
    def __init__(self, node):
        self.node = node
        self.bridge = CvBridge()

        self.pub_color = self.node.create_publisher(Image, "camera/color/image_raw", 1)
        self.pub_depth = self.node.create_publisher(Image, "camera/depth/image_raw", 1)
        self.pub_color_info = self.node.create_publisher(CameraInfo, "camera/color/camera_info", 1)
        self.pub_depth_info = self.node.create_publisher(CameraInfo, "camera/depth/camera_info", 1)

        self.ln_clnt = ln.client("LnRosCameraBridge", [])
        self.ln_inport_rgb = self.ln_clnt.subscribe("camera.perception/rgb", "perception/rgb")
        self.ln_inport_depth = self.ln_clnt.subscribe("camera.perception/depth", "perception/depth")
        self.ln_inport_camk = self.ln_clnt.subscribe("camera.perception/cam_k", "perception/cam_k")

        self.node.get_logger().info("Read cam_k...")
        self.info_msg = self.get_camera_info_from_camk(FRAME_CAM)  # color and depth share K (registered depth)

        self._last_ts = None
        self._frames = 0
        self._skipped = 0
        self._last_update = time.time()
        self.node.get_logger().info("Init done...")

    def get_camera_info_from_camk(self, frame_id: str) -> CameraInfo:
        self.ln_inport_camk.read()
        K = np.asarray(self.ln_inport_camk.packet.data, dtype=float).reshape(3, 3)

        info = CameraInfo()
        info.header.frame_id = frame_id
        info.distortion_model = "plumb_bob"   # assumed rectified / unknown
        info.d = [0.0] * 5
        info.k = K.ravel().tolist()
        info.r = np.eye(3, dtype=float).ravel().tolist()
        P = np.zeros((3, 4), dtype=float)     # [K | 0]
        P[:3, :3] = K
        info.p = P.ravel().tolist()
        return info

    def get_rgbd_ln(self):
        self.ln_inport_rgb.read()      # blocking
        self.ln_inport_depth.read()    # blocking
        packet_rgb = self.ln_inport_rgb.packet
        packet_depth = self.ln_inport_depth.packet
        if packet_rgb is None or packet_depth is None:
            return None, None, None
        rgb = np.array(packet_rgb.data).reshape(packet_rgb.height, packet_rgb.width, 3).astype("uint8")
        depth = np.array(packet_depth.data).reshape(packet_depth.height, packet_depth.width).astype("uint16")
        return rgb, depth, float(packet_depth.timestamp)

    def publish_rgbd(self, rgb, depth, ts):
        stamp = to_ros_time(ts)

        rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="16UC1")
        for msg in (rgb_msg, depth_msg):
            msg.header.stamp = stamp
            msg.header.frame_id = FRAME_CAM

        self.info_msg.header.stamp = stamp
        self.info_msg.height, self.info_msg.width = rgb.shape[:2]

        self.pub_color.publish(rgb_msg)
        self.pub_depth.publish(depth_msg)
        self.pub_color_info.publish(self.info_msg)
        self.pub_depth_info.publish(self.info_msg)

    def run(self):
        self.node.get_logger().info("Bridging LN camera to ROS...")
        while rclpy.ok():
            rgb, depth, ts = self.get_rgbd_ln()
            if depth is None:
                self.node.get_logger().warning("Failed to read RGB or depth.")
                continue
            if ts == self._last_ts:
                self._skipped += 1
                continue
            self.publish_rgbd(rgb, depth, ts)
            self._last_ts = ts
            self._frames += 1

            now = time.time()
            elapsed = now - self._last_update
            if elapsed >= 5.0:
                self.node.get_logger().info(
                    f"Published with {self._frames / elapsed:.2f} Hz "
                    f"(frames {self._frames}, skipped duplicate stamps {self._skipped})")
                self._frames = 0
                self._skipped = 0
                self._last_update = now


def main():
    rclpy.init()
    node = rclpy.create_node("ln_ros_camera_bridge")
    bridge = LnRosCameraBridge(node)
    try:
        bridge.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
