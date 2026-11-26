#!/usr/bin/env python3
"""
Live viewer for the depth completion (ROS 2): RGB | sensor depth | completed depth.

Runs exactly the code of the ROS node (depth_models.py + depth_fusion.fill), only with a
display instead of publishing. Meant for trying models and parameters on the live camera.

    python navigation/mapping/depth_completion/viewer.py                          # moge2-b, 600 tokens
    python navigation/mapping/depth_completion/viewer.py --model moge --num-tokens 1200
    python navigation/mapping/depth_completion/viewer.py --model depthpro2-s --stream
    python navigation/mapping/depth_completion/viewer.py --set ramp_dmin=0 support_px=0  # filters off

Models: see depth_models.MODELS. --stream only for Depth Pro 2 (temporally more stable).
--set: parameters of depth_fusion.fill / GradientFusion, e.g. down=4 support_px=8.

Completed panel: white contour = filled regions, magenta = discarded by the edge filters.
Console every 2 s: rate, timings and (key e) the error below 3 m. For that, random 32 px blocks
of measured pixels are hidden, filled and compared with the measurement; this costs a second
fill per frame.

Keys: q = quit, e = error evaluation on/off.
"""
import argparse
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
import torch
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_fusion import GradientFusion, fill  # noqa: E402
from depth_models import MODELS, load_model  # noqa: E402

FUSION_KEYS = ("down", "lam", "mu_near", "mu_far", "far_band", "sigma_edge", "z_ref",
               "outlier", "iters", "tol")
NEAR_M = 3.0


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


class Listener(Node):
    """Newest RGB/depth pair and the latest camera_info."""

    def __init__(self, a):
        super().__init__("depth_completion_viewer")
        self.lock, self.latest, self.info = threading.Lock(), None, None
        subs = [Subscriber(self, Image, a.color, qos_profile=qos_profile_sensor_data),
                Subscriber(self, Image, a.depth, qos_profile=qos_profile_sensor_data)]
        self.sync = ApproximateTimeSynchronizer(subs, queue_size=5, slop=0.02)
        self.sync.registerCallback(self.cb)
        self.create_subscription(CameraInfo, a.info, lambda m: setattr(self, "info", m),
                                 qos_profile_sensor_data)

    def cb(self, c, d):
        with self.lock:
            self.latest = (c, d)

    def take(self):
        with self.lock:
            m, self.latest = self.latest, None
        return m


def colorize(d, lo, hi):
    n = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((255 * n).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[d <= 0] = 0
    return img


def label(img, s):
    for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(img, s, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, th, cv2.LINE_AA)


def holdout_error(depth, d_t, pred, fusion_kw, fill_kw, log_s, f_px, rng):
    """AbsRel [%] and MAE [cm] below NEAR_M on hidden 32 px blocks, otherwise None."""
    H, W = depth.shape
    pick = rng.random((-(-H // 32), -(-W // 32))) < 0.15
    blocks = np.kron(pick, np.ones((32, 32), bool))[:H, :W] & (depth > 0)
    hidden = torch.where(torch.from_numpy(blocks).to(d_t.device), torch.zeros_like(d_t), d_t)
    f, _ = fill(hidden, pred, GradientFusion(**fusion_kw), log_s, f_px=f_px, **fill_kw)
    f = f.cpu().numpy()
    m = blocks & (f > 0) & (depth < NEAR_M)
    if m.sum() < 50:
        return None
    err = np.abs(f[m] - depth[m])
    return 100 * float((err / depth[m]).mean()), 100 * float(err.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="moge2-b", choices=MODELS)
    ap.add_argument("--num-tokens", type=int, default=600, help="MoGe only")
    ap.add_argument("--stream", action="store_true", help="Depth Pro 2 only: live stream mode")
    ap.add_argument("--set", nargs="*", default=[], help="parameters, e.g. ramp_dmin=0 down=4")
    ap.add_argument("--color", default="/camera/color/image_raw")
    ap.add_argument("--depth", default="/camera/depth/image_raw")
    ap.add_argument("--info", default="/camera/color/camera_info")
    ap.add_argument("--width", type=int, default=640, help="width per panel")
    a = ap.parse_args()

    params = {k: float(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    fusion_kw = {k: (int(v) if k in ("down", "iters") else v) for k, v in params.items() if k in FUSION_KEYS}
    fill_kw = {k: (int(v) if k == "support_px" else v) for k, v in params.items() if k not in FUSION_KEYS}

    print(f"Loading {a.model} ...")
    model = load_model(a.model, a.num_tokens, stream=a.stream)
    dev = model.dev
    rclpy.init()
    node = Listener(a)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    fusion, log_s, rng = GradientFusion(**fusion_kw), None, np.random.default_rng(0)
    evaluate, err = True, None
    t_sum, n, last = {"model": 0.0, "fill": 0.0, "eval": 0.0}, 0, time.perf_counter()

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()

    cv2.namedWindow("depth viewer", cv2.WINDOW_NORMAL)
    try:
        while rclpy.ok():
            pair, info = node.take(), node.info
            if pair is None or info is None:
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue
            rgb, depth = to_rgb(pair[0]), to_depth(pair[1])
            H, W = rgb.shape[:2]
            if depth.shape != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            f_px = float(info.k[0]) * W / info.width
            fov_x = math.degrees(2 * math.atan(W / (2 * f_px)))

            t0 = time.perf_counter()
            pred = model(rgb, fov_x)
            sync()
            t1 = time.perf_counter()
            d_t = torch.from_numpy(depth).to(dev)
            filled_t, log_s = fill(d_t, pred, fusion, log_s, f_px=f_px, **fill_kw)
            filled = filled_t.cpu().numpy()
            t2 = time.perf_counter()
            if evaluate:
                err = holdout_error(depth, d_t, pred, fusion_kw, fill_kw, log_s, f_px, rng) or err
            t3 = time.perf_counter()
            for k, dt in zip(t_sum, (t1 - t0, t2 - t1, t3 - t2)):
                t_sum[k] += 1000 * dt
            n += 1

            # display
            valid = depth > 0
            dropped = ~valid & (filled <= 0) & torch.isfinite(pred).cpu().numpy()
            lo = float(np.percentile(depth[valid], 2)) if valid.any() else 0.0
            hi = float(np.percentile(filled[filled > 0], 98)) if (filled > 0).any() else 5.0
            p_fill = colorize(filled, lo, hi)
            p_fill[dropped] = (255, 0, 255)
            cnts, _ = cv2.findContours((~valid & (filled > 0)).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(p_fill, cnts, -1, (255, 255, 255), 1)
            h = int(H * a.width / W)
            panels = [cv2.resize(p, (a.width, h), interpolation=cv2.INTER_NEAREST)
                      for p in (rgb[..., ::-1], colorize(depth, lo, hi), p_fill)]
            label(panels[0], f"RGB  {a.model}")
            label(panels[1], f"sensor  {lo:.1f}-{hi:.1f} m, holes {100 * (~valid).mean():.0f} %")
            label(panels[2], f"completed  s={math.exp(log_s):.2f}"
                  + (f"  <3m {err[0]:.1f} % / {err[1]:.1f} cm" if evaluate and err else ""))
            cv2.imshow("depth viewer", np.hstack(panels))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("e"):
                evaluate, err = not evaluate, None

            now = time.perf_counter()
            if now - last >= 2.0:
                avg = {k: v / n for k, v in t_sum.items()}
                print(f"{n / (now - last):5.1f} Hz | "
                      + " | ".join(f"{k} {v:.0f} ms" for k, v in avg.items())
                      + (f" | error <3 m: {err[0]:.2f} % / {err[1]:.1f} cm" if evaluate and err else ""),
                      flush=True)
                t_sum, n, last = dict.fromkeys(t_sum, 0.0), 0, now
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
