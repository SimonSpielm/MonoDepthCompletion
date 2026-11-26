#!/usr/bin/env python3
r"""
Step 3: MoGe-2 mono depth completion (ROS 2, GPU, no visualisation).

Fills the holes of the sensor depth and publishes the completed depth together with the
matching RGB image and camera_info, all with the ORIGINAL header (stamp = LN capture
timestamp, frame_id = camera):

    /camera/depth/image_raw   (trigger)  --\                    /camera/completed/depth/image_raw (16UC1, mm)
    /camera/color/image_raw   (MoGe input) >-- MoGe + Fusion -->  /camera/completed/color/image_raw
    /camera/color/camera_info (FoV, static)/                     /camera/completed/camera_info

The depth image is the only trigger, there is no message_filters synchronizer.
MoGe predicts depth FROM the RGB image, so RGB is still needed as input: it is kept
in a small buffer and looked up by the stamp of the depth image (the bridge stamps
both with the same LN capture time). A lost RGB message therefore only costs that
one frame and never stalls the stream. The RGB image that was used for a frame is
re-published next to its completed depth, so downstream nodes (filter, nvblox colouring)
get color, depth and camera_info with one identical stamp and need no pairing of their own.

1. MoGe-2 schätzt metrische Tiefe (mit echter Brennweite aus camera_info).
2. Globaler Faktor: Median(Messung / MoGe), über die Zeit geglättet.
3. Gradienten-Fusion: Korrekturfeld c (log D = log M + c), das die Messungen trifft, entlang
   von Oberflächen glatt ist (MoGe gibt die Form vor), nicht über Kanten wirkt und hinter dem
   Messbereich auf 0 abklingt. Gelöst per vorkonditioniertem CG auf der GPU.
   Gemessene Pixel bleiben unverändert.

MoGe is slower than the camera: only the newest frame is processed, older ones are dropped.

Usage (conda env with torch + MoGe, ROS sourced):
    python moge_depth_completion.py
"""
import os
import sys

sys.path = [p for p in sys.path if "conan_cache" not in p]          # altes Conan-numpy raus
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

MODEL = "Ruicheng/moge-2-vitl"
NUM_TOKENS = 1200
COLOR_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_raw"
INFO_TOPIC = "/camera/color/camera_info"
OUT_DEPTH_TOPIC = "/camera/completed/depth/image_raw"
OUT_COLOR_TOPIC = "/camera/completed/color/image_raw"
OUT_INFO_TOPIC = "/camera/completed/camera_info"
FILL_EDGE_RTOL = 0.04  # gefüllte Pixel an Tiefenkanten verwerfen: Tiefenspanne im 3x3 > 4 %, 0 = aus
RGB_BUFFER = 10        # zuletzt empfangene RGB-Bilder, nach Zeitstempel abrufbar (~0.7 s bei 15 Hz)


# Algorithmus (ROS-frei, auch vom Offline-Benchmark genutzt)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_fusion import GradientFusion, MoGe, fill  # noqa: E402

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


class MoGeDepthCompletion(Node):
    def __init__(self):
        super().__init__("moge_depth_completion")
        self.lock = threading.Lock()
        self.new_frame = threading.Event()
        self.latest_depth = None             # neueste, noch nicht verarbeitete Tiefe (Trigger)
        self.rgb_buffer = OrderedDict()      # stamp -> RGB-Nachricht
        self.info_msg = None                 # statisch, nur für die Brennweite
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
        self.new_frame.set()     # die passende Tiefe kann schon warten

    def depth_cb(self, msg):
        with self.lock:
            self.latest_depth = msg          # ältere, noch nicht verarbeitete Tiefe verwerfen
            self._n_depth += 1
        self.new_frame.set()

    def take_depth_count(self):
        with self.lock:
            n, self._n_depth = self._n_depth, 0
        return n

    def take(self, timeout=0.5):
        """Neueste Tiefe + RGB mit identischem Zeitstempel, sonst None.

        Fehlt das RGB-Bild noch (Reihenfolge der Ankunft), bleibt die Tiefe liegen, bis es
        kommt oder eine neuere Tiefe sie ersetzt.
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
        # Color is the exact RGB image MoGe used (same stamp); camera_info gets that stamp too
        info_msg.header.stamp = header.stamp
        self.pub_color.publish(color_msg)
        self.pub_depth.publish(depth_to_msg(depth_mm, header))
        self.pub_info.publish(info_msg)


# ---------------------------------------------------------------- Hauptschleife

def main():
    print("Lade MoGe-2 ...")
    moge = MoGe(MODEL, NUM_TOKENS)
    if moge.dev.type != "cuda":
        print("WARNUNG: keine CUDA-GPU, MoGe und Fusion laufen auf der CPU (sehr langsam).")
    rclpy.init()
    node = MoGeDepthCompletion()
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
            if frame is None or info_msg is None:   # noch kein Paar / noch keine camera_info
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
            t = tick("MoGe", t)

            filled, log_s = fill(torch.from_numpy(depth).to(pred.device), pred, fusion, log_s,
                                 edge_rtol=FILL_EDGE_RTOL)
            depth_mm = (filled * 1000.0).round().clamp(0, 65535).to(torch.int32).cpu().numpy() \
                .astype(np.uint16)
            t = tick("Fusion", t)

            node.publish(color_msg, depth_mm, info_msg, depth_msg.header)
            tick("Publish", t)

            n_sum += 1
            now = time.perf_counter()
            if now - last >= 2.0:   # Mittelwerte der letzten ~2 s in die Konsole
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
