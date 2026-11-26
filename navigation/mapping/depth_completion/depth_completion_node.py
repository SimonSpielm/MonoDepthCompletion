#!/usr/bin/env python3
r"""
Step 3: depth completion with a monocular model (ROS 2, GPU, no visualisation).

Fills the holes of the sensor depth and publishes the completed depth together with the
matching RGB image and camera_info, all with the ORIGINAL header (stamp = LN capture time,
frame_id = camera):

    /camera/depth/image_raw   (trigger)     --\                       /camera/completed/depth/image_raw (16UC1, mm)
    /camera/color/image_raw   (model input)  >-- model + fusion -->  /camera/completed/color/image_raw
    /camera/color/camera_info (FoV, static) --/                       /camera/completed/camera_info

The depth image is the only trigger, there is no message_filters synchronizer. The model
estimates depth FROM the RGB image, so RGB is still needed as input: it is kept in a small
buffer and looked up by the stamp of the depth image (the bridge stamps both with the same LN
capture time). A lost RGB message therefore only costs that one frame and never stalls the
stream. The RGB image used for a frame is re-published next to its completed depth, so
downstream nodes (filter, nvblox colouring) get color, depth and camera_info with one identical
stamp and need no pairing of their own.

Pipeline (details in README.md and depth_fusion.py):
1. The model (MODEL, see depth_models.py) predicts metric depth, using the real focal length
   from camera_info.
2. Global scale: median(measurement / model), smoothed over time.
3. Gradient fusion: correction field c (log D = log M + c) that matches the measurements, is
   smooth along surfaces (the model provides the shape), does not act across edges and decays
   to 0 behind the measured range. Solved with preconditioned CG on the GPU. Measured pixels
   are never changed.
4. Edge filters against floating points: filled pixels on the model's smeared edge (too steep
   for a real surface) and pixels in the zone around real depth edges that match neither the
   near nor the far side are discarded.

The model is slower than the camera: only the newest frame is processed, older ones are dropped.

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
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

# Model (depth_models.MODELS) / resolution. Measured on M4 Pro (MPS), 1280x720, incl. fusion
# (bench/speed.py), error < 3 m in large holes (bench/depth_bench.py --hold-px 128):
#   moge2-s  600:  85 ms, 3.8 %  | moge2-b  600: 107 ms, 3.1 %  <- 8 Hz target
#   moge2-b 1200: 182 ms, 2.8 %  | moge    1200: 306 ms, 1.9 %  | depthpro2-l: 1029 ms, 1.6 %
MODEL = "moge2-b"
NUM_TOKENS = 600
COLOR_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_raw"
INFO_TOPIC = "/camera/color/camera_info"
OUT_DEPTH_TOPIC = "/camera/completed/depth/image_raw"
OUT_COLOR_TOPIC = "/camera/completed/color/image_raw"
OUT_INFO_TOPIC = "/camera/completed/camera_info"
RGB_BUFFER = 10        # last received RGB images, looked up by stamp (~0.7 s at 15 Hz)


# algorithm (ROS-free, also used by the offline benchmark)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_fusion import GradientFusion, fill  # noqa: E402
from depth_models import load_model  # noqa: E402

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


def stamp_key(stamp):
    return (stamp.sec, stamp.nanosec)


class DepthCompletion(Node):
    def __init__(self):
        super().__init__("depth_completion")
        self.lock = threading.Lock()
        self.new_frame = threading.Event()
        self.latest_depth = None             # newest depth not yet processed (trigger)
        self.rgb_buffer = OrderedDict()      # stamp -> RGB message
        self.info_msg = None                 # static, only for the focal length
        self._n_depth = 0

        self.create_subscription(Image, DEPTH_TOPIC, self.depth_cb, qos_profile_sensor_data)
        self.create_subscription(Image, COLOR_TOPIC, self.rgb_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, INFO_TOPIC, self.info_cb, qos_profile_sensor_data)

        self.pub_depth = self.create_publisher(Image, OUT_DEPTH_TOPIC, 1)
        self.pub_color = self.create_publisher(Image, OUT_COLOR_TOPIC, 1)
        self.pub_info = self.create_publisher(CameraInfo, OUT_INFO_TOPIC, 1)

    def info_cb(self, msg):
        self.info_msg = msg

    def rgb_cb(self, msg):
        with self.lock:
            self.rgb_buffer[stamp_key(msg.header.stamp)] = msg
            while len(self.rgb_buffer) > RGB_BUFFER:
                self.rgb_buffer.popitem(last=False)
        self.new_frame.set()     # the matching depth may already be waiting

    def depth_cb(self, msg):
        with self.lock:
            self.latest_depth = msg          # drop the older, unprocessed depth
            self._n_depth += 1
        self.new_frame.set()

    def take_depth_count(self):
        with self.lock:
            n, self._n_depth = self._n_depth, 0
        return n

    def take(self, timeout=0.5):
        """Newest depth + the RGB image with the identical stamp, otherwise None.

        If the RGB image has not arrived yet (order of arrival), the depth is kept until it
        arrives or a newer depth replaces it.
        """
        if not self.new_frame.wait(timeout):
            return None
        with self.lock:
            self.new_frame.clear()
            depth = self.latest_depth
            if depth is None:
                return None
            rgb = self.rgb_buffer.get(stamp_key(depth.header.stamp))
            if rgb is None:
                return None
            self.latest_depth = None
        return rgb, depth

    def publish(self, color_msg, depth_mm, info_msg, header):
        # color = exactly the RGB image the depth was estimated from (same stamp);
        # camera_info gets the same stamp
        info_msg.header.stamp = header.stamp
        self.pub_color.publish(color_msg)
        self.pub_depth.publish(depth_to_msg(depth_mm, header))
        self.pub_info.publish(info_msg)


# ---------------------------------------------------------------- main loop

def main():
    print(f"Lade {MODEL} ({NUM_TOKENS} Tokens) ...")
    moge = load_model(MODEL, NUM_TOKENS)
    if moge.dev.type == "cpu":
        print("WARNUNG: keine GPU, Modell und Fusion laufen auf der CPU (sehr langsam).")
    rclpy.init()
    node = DepthCompletion()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    fusion = GradientFusion()
    log_s = None
    t_sum, n_sum = {k: 0.0 for k in ("Konvert.", "MoGe", "Fusion", "Publish")}, 0
    last = time.perf_counter()

    def tick(key, t0):
        t1 = time.perf_counter()
        t_sum[key] += 1000 * (t1 - t0)
        return t1

    try:
        while rclpy.ok():
            frame = node.take()
            info_msg = node.info_msg
            if frame is None or info_msg is None:   # no pair / no camera_info yet
                continue
            color_msg, depth_msg = frame
            t = time.perf_counter()
            rgb, depth = to_rgb(color_msg), to_depth(depth_msg)
            H, W = rgb.shape[:2]
            if depth.shape != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            fov_x = math.degrees(2 * math.atan(info_msg.width / (2 * info_msg.k[0])))
            t = tick("Konvert.", t)

            pred = moge(rgb, fov_x)
            if moge.dev.type == "cuda":
                torch.cuda.synchronize()
            elif moge.dev.type == "mps":
                torch.mps.synchronize()
            t = tick("MoGe", t)

            # edge filters (ramp, support check) with the defaults of depth_fusion.fill
            f_px = float(info_msg.k[0]) * W / info_msg.width
            filled, log_s = fill(torch.from_numpy(depth).to(pred.device), pred, fusion, log_s,
                                 f_px=f_px)
            depth_mm = (filled * 1000.0).round().clamp(0, 65535).to(torch.int32).cpu().numpy() \
                .astype(np.uint16)
            t = tick("Fusion", t)

            node.publish(color_msg, depth_mm, info_msg, depth_msg.header)
            tick("Publish", t)

            n_sum += 1
            now = time.perf_counter()
            if now - last >= 2.0:   # averages of the last ~2 s to the console
                dt = now - last
                a = {k: v / max(n_sum, 1) for k, v in t_sum.items()}
                print(f"Tiefe in {node.take_depth_count() / dt:5.1f} Hz | verarb. {n_sum / dt:5.1f} Hz "
                      f"| s={math.exp(log_s):.2f} | "
                      + " | ".join(f"{k} {v:.1f}" for k, v in a.items()) + " ms", flush=True)
                t_sum, n_sum, last = dict.fromkeys(t_sum, 0.0), 0, now
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
