"""
Step 2: LN -> ROS camera bridge.

Reads the LN camera topics (from the real camera driver or ln_camera_publisher.py)
and publishes them unchanged as ROS messages. Both images of one frame carry the
LN capture timestamp as header.stamp; it is passed through the whole pipeline.

    LN camera.perception/rgb    -> /camera/color/image_raw    (rgb8, qos_buffer)
    LN camera.perception/depth  -> /camera/depth/image_raw    (16UC1, mm, qos_trigger)
    LN camera.perception/cam_k  -> /camera/color/camera_info  (latched, sent once)

The depth is registered to the colour camera, so both images share one K and one
camera_info. Colour is published before depth, so a stage triggered by the depth
usually has the matching colour image already. QoS rules: pipeline_io.py.

No timer: inport.read() blocks until the next packet, so frames are forwarded as
soon as they arrive. Single thread.

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

from navigation.mapping.pipeline_io import (COLOR_TOPIC, DEPTH_TOPIC, INFO_TOPIC, StageMonitor, qos_buffer,
                                            qos_latched, qos_trigger)
from navigation.mapping.ros_utils import to_ros_time

FRAME_CAM = "camera"


class LnRosCameraBridge:
    def __init__(self, node):
        self.node = node
        self.bridge = CvBridge()

        self.pub_color = self.node.create_publisher(Image, COLOR_TOPIC, qos_buffer())
        self.pub_depth = self.node.create_publisher(Image, DEPTH_TOPIC, qos_trigger())
        self.pub_info = self.node.create_publisher(CameraInfo, INFO_TOPIC, qos_latched())

        self.ln_clnt = ln.client("LnRosCameraBridge", [])
        self.ln_inport_rgb = self.ln_clnt.subscribe("camera.perception/rgb", "perception/rgb")
        self.ln_inport_depth = self.ln_clnt.subscribe("camera.perception/depth", "perception/depth")
        self.ln_inport_camk = self.ln_clnt.subscribe("camera.perception/cam_k", "perception/cam_k")

        self.node.get_logger().info("Read cam_k...")
        self.info_msg = self.get_camera_info_from_camk(FRAME_CAM)  # color and depth share K (registered depth)

        self._last_ts = None
        self._info_sent = False
        self.monitor = StageMonitor(self.node.get_logger().info, "bridge")
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

        if not self._info_sent:              # static: once, latched
            self.info_msg.header.stamp = stamp
            self.info_msg.height, self.info_msg.width = rgb.shape[:2]
            self.pub_info.publish(self.info_msg)
            self._info_sent = True

        self.pub_color.publish(rgb_msg)      # colour first, see module docstring
        self.pub_depth.publish(depth_msg)

    def run(self):
        self.node.get_logger().info("Bridging LN camera to ROS...")
        while rclpy.ok():
            rgb, depth, ts = self.get_rgbd_ln()
            if depth is None:
                self.node.get_logger().warning("Failed to read RGB or depth.")
                continue
            self.monitor.received()
            if ts == self._last_ts:
                self.monitor.count("duplicates")
                continue
            t0 = time.perf_counter()
            self.publish_rgbd(rgb, depth, ts)
            self._last_ts = ts
            self.monitor.published(to_ros_time(ts), publish=1000 * (time.perf_counter() - t0))


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
