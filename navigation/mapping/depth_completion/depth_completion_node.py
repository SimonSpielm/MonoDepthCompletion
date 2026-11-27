#!/usr/bin/env python3
r"""
Step 3: depth completion with a monocular model (ROS 2, GPU, no visualisation).

Fills the holes of the sensor depth and publishes only the completed depth, with the ORIGINAL
header (stamp = LN capture time, frame_id = camera):

    /camera/depth/image_raw   (trigger)      --\
    /camera/color/image_raw   (model input)   >-- model + fusion -->  /camera/completed/depth/image_raw
    /camera/color/camera_info (FoV, latched) --/                       (16UC1, mm)

Colour and camera_info are not re-published: downstream stages take them from the bridge and
match the colour image by stamp (pipeline_io.ColorByStamp).

One thread, no timer, no synchroniser. The node keeps the last colour image and the last depth
image; whenever one of them arrives and both have the same stamp (the bridge stamps both with
the LN capture time), the frame is processed inside that callback. A depth image that arrives
before its colour image therefore simply waits for it. While a frame is processed, DDS keeps
only the newest depth image (QoS keep last 1), so older frames are dropped instead of queued.

Pipeline (details in README.md and depth_fusion.py):
1. The model (MODEL, see depth_models.py) predicts metric depth, using the real focal length
   from camera_info.
2. Global scale: median(measurement / model), smoothed over time.
3. Gradient fusion: correction field c (log D = log M + c) that matches the measurements, is
   smooth along surfaces (the model provides the shape), does not act across edges and decays
   to 0 behind the measured range. Solved with preconditioned CG on the GPU. Measured pixels
   are never changed.
4. Edge filter against floating points: filled pixels next to jumps in inverse depth are
   discarded.

Usage (conda env with torch and the model package, ROS sourced):
    python navigation/mapping/depth_completion/depth_completion_node.py
"""
import os
import sys

sys.path = [p for p in sys.path if "conan_cache" not in p]          # drop the old Conan numpy
_ROS = "/opt/ros/jazzy/lib/python3.12/site-packages"
if os.path.isdir(_ROS) and _ROS not in sys.path:
    sys.path.append(_ROS)
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import math
import time

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

# Model (depth_models.MODELS) / resolution. Measured on the robot's RTX A6000, 1280x720, incl.
# completion (bench/run_suite.py, report rmc-lx0481_20261127_1009), error < 3 m in large holes:
#   moge     600:  34 ms, 1.39 %, gross 1.1 %  <- default
#   moge2-b  600:  28 ms, 1.98 %, gross 3.7 %  | moge2-s 600: 24 ms, 2.55 %, gross 5.1 %
#   depthpro2-b stream 4: 52 ms, 1.56 %, steadier (flicker 0.9 vs 1.2 %), research licence
MODEL = "moge"
NUM_TOKENS = 600
STREAM = True         # Depth Pro 2 only: streaming mode (attends to previous frames, steadier)
STREAM_WINDOW = 4     # frames attended incl. the current one (M4 Pro, S: 4 -> 184 ms, 16 -> 312 ms; 16 adds gross errors)


# algorithm (ROS-free, also used by the offline benchmark) and the shared pipeline rules
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from depth_fusion import GradientFusion, fill  # noqa: E402
from depth_models import load_model  # noqa: E402
from pipeline_io import (COLOR_TOPIC, COMPLETED_DEPTH_TOPIC, DEPTH_TOPIC, INFO_TOPIC,  # noqa: E402
                         StageMonitor, qos_buffer, qos_latched, qos_trigger, same_stamp)

# ---------------------------------------------------------------- ROS

def to_rgb(msg):
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}[msg.encoding]
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * ch]
    img = img.reshape(msg.height, msg.width, ch)[..., :3]
    return np.ascontiguousarray(img[..., ::-1] if msg.encoding.startswith("bgr") else img)


def to_depth(msg):
    if msg.encoding == "32FC1":
        d = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.step // 4)[:, : msg.width]
        return np.nan_to_num(d.copy(), nan=0.0, posinf=0.0)
    d = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[:, : msg.width]
    return d.astype(np.float32) * 0.001


def depth_to_msg(depth_mm, header):
    msg = Image()
    msg.header = header
    msg.height, msg.width = depth_mm.shape
    msg.encoding = "16UC1"
    msg.is_bigendian = 0
    msg.step = msg.width * 2
    msg.data = depth_mm.tobytes()
    return msg


class DepthCompletion(Node):
    def __init__(self, model):
        super().__init__("depth_completion")
        self.model = model
        self.fusion = GradientFusion()
        self.log_s = None                    # smoothed log global scale
        self.rgb_msg = None                  # last colour image
        self.depth_msg = None                # last depth image not processed yet
        self.info_msg = None                 # static, only for the focal length
        self.monitor = StageMonitor(self.get_logger().info, "completion")

        self.create_subscription(Image, DEPTH_TOPIC, self.on_depth, qos_trigger())
        self.create_subscription(Image, COLOR_TOPIC, self.on_rgb, qos_buffer())
        self.create_subscription(CameraInfo, INFO_TOPIC, self.on_info, qos_latched())
        self.pub_depth = self.create_publisher(Image, COMPLETED_DEPTH_TOPIC, qos_trigger())

    def on_info(self, msg):
        self.info_msg = msg

    def on_rgb(self, msg):
        self.rgb_msg = msg
        self.try_process()

    def on_depth(self, msg):
        self.monitor.received()
        self.depth_msg = msg                 # a newer depth replaces an unprocessed older one
        self.try_process()

    def try_process(self):
        d, c = self.depth_msg, self.rgb_msg
        if d is None or c is None or self.info_msg is None or not same_stamp(d.header.stamp, c.header.stamp):
            return
        self.depth_msg = None
        self.process(c, d)

    def process(self, color_msg, depth_msg):
        sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(self.model.dev.type, lambda: None)
        t0 = time.perf_counter()
        rgb, depth = to_rgb(color_msg), to_depth(depth_msg)
        H, W = rgb.shape[:2]
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        fov_x = math.degrees(2 * math.atan(self.info_msg.width / (2 * self.info_msg.k[0])))
        t1 = time.perf_counter()

        pred = self.model(rgb, fov_x)
        sync()
        t2 = time.perf_counter()

        filled, self.log_s = fill(torch.from_numpy(depth).to(pred.device), pred, self.fusion, self.log_s)
        depth_mm = (filled * 1000.0).round().clamp(0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16)
        t3 = time.perf_counter()

        self.pub_depth.publish(depth_to_msg(depth_mm, depth_msg.header))
        t4 = time.perf_counter()
        self.monitor.published(depth_msg.header.stamp, convert=1000 * (t1 - t0), model=1000 * (t2 - t1),
                               fusion=1000 * (t3 - t2), publish=1000 * (t4 - t3))


def main():
    print(f"Loading {MODEL} ({NUM_TOKENS} tokens) ...")
    model = load_model(MODEL, NUM_TOKENS, stream=STREAM, window=STREAM_WINDOW)
    if model.dev.type == "cpu":
        print("WARNING: no GPU, model and fusion run on the CPU (very slow).")
    rclpy.init()
    node = DepthCompletion(model)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
