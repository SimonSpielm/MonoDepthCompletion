#!/usr/bin/env python3
"""
ROS 2 Viewer: RGB | Rohtiefe | gefüllte Tiefe (Prompt Depth Anything) nebeneinander.

Alles in einer Datei.

Setup (im conda-Env, Python-Version muss zu ROS passen, z. B. 3.12 für Jazzy):
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install transformers opencv-python pillow "numpy<2"

Start:
  source /opt/ros/<distro>/setup.bash
  python navigation/mapping/fill_depth.py
  # andere Topics:
  python navigation/mapping/fill_depth.py --color /camera/color/image_raw \
                                          --depth /camera/depth/image_raw

  # ohne torch.compile (z. B. zum Vergleichen):
  python navigation/mapping/fill_depth.py --compile off

Tasten im Fenster:  q = beenden,  m = Markierung geschätzter Pixel an/aus
"""
import argparse
import threading
import time

import cv2
import numpy as np
import rclpy
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image



# ---------------------------------------------------------------- Modell

MODEL = "depth-anything/prompt-depth-anything-vits-hf"
# Alternative für Glas/Spiegel: "depth-anything/prompt-depth-anything-vits-transparent-hf"


def prefill_lowres(depth_m, size=(256, 192)):
    """Löcher per Nearest-Neighbour schließen und auf LiDAR-artige Auflösung bringen.
    PromptDA ist auf lückenlose, niedrig aufgelöste Tiefe (ARKit 256x192) trainiert."""
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not valid.any():
        raise ValueError("Keine gültige Tiefe im Frame.")
    # Label = Index des nächsten gültigen Pixels (in Rasterreihenfolge, ab 1)
    _, lab = cv2.distanceTransformWithLabels((~valid).astype(np.uint8), cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    idx = np.flatnonzero(valid.ravel())
    dense = depth_m.ravel()[idx[lab.ravel() - 1]].reshape(depth_m.shape).astype(np.float32)
    return cv2.resize(dense, size, interpolation=cv2.INTER_NEAREST)


class DepthFiller:
    def __init__(self, model_id=MODEL, device=None, compile_mode="reduce-overhead"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = (AutoModelForDepthEstimation.from_pretrained(model_id, torch_dtype=self.dtype)
                      .to(self.device).eval())
        if compile_mode != "off" and self.device == "cuda":
            # Erster Frame dauert dann lange (Kompilieren), danach schneller.
            mode = None if compile_mode == "default" else compile_mode
            self.model = torch.compile(self.model, mode=mode)

    @torch.inference_mode()
    def __call__(self, rgb, depth_m):
        H, W = rgb.shape[:2]
        if depth_m.shape != (H, W):
            depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)
        valid = np.isfinite(depth_m) & (depth_m > 0)

        inputs = self.proc(images=rgb, return_tensors="pt")
        prompt = torch.from_numpy(prefill_lowres(depth_m))[None, None]  # (1,1,h,w), Meter
        out = self.model(pixel_values=inputs["pixel_values"].to(self.device, self.dtype),
                         prompt_depth=prompt.to(self.device, self.dtype))
        pred = self.proc.post_process_depth_estimation(out, target_sizes=[(H, W)])[0]
        pred = pred["predicted_depth"].float().cpu().numpy()

        filled = np.where(valid, depth_m, pred)   # Messwerte behalten, nur Löcher füllen
        return filled, ~valid


# ---------------------------------------------------------------- ROS + Anzeige

def color_to_rgb(msg):
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(msg.encoding)
    if ch is None:
        raise ValueError(f"Farb-Encoding {msg.encoding} nicht unterstützt")
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * ch]
    img = img.reshape(msg.height, msg.width, ch)
    if msg.encoding.startswith("bgr"):
        img = img[..., [2, 1, 0]]
    return np.ascontiguousarray(img[..., :3])


def depth_to_m(msg):
    if msg.encoding in ("16UC1", "mono16"):
        d = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[:, : msg.width]
        return d.astype(np.float32) * 0.001  # mm -> m
    if msg.encoding == "32FC1":
        d = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.step // 4)[:, : msg.width]
        return np.nan_to_num(d.copy(), nan=0.0)
    raise ValueError(f"Tiefen-Encoding {msg.encoding} nicht unterstützt")


def colorize(depth_m, dmin, dmax):
    valid = depth_m > 0
    n = np.clip((depth_m - dmin) / max(dmax - dmin, 1e-6), 0, 1)
    vis = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~valid] = 0  # Löcher schwarz
    return vis


def label(img, text):
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


class Listener(Node):
    def __init__(self, color_topic, depth_topic):
        super().__init__("promptda_viewer")
        self.lock = threading.Lock()
        self.latest = None
        subs = [Subscriber(self, Image, color_topic, qos_profile=qos_profile_sensor_data),
                Subscriber(self, Image, depth_topic, qos_profile=qos_profile_sensor_data)]
        self.sync = ApproximateTimeSynchronizer(subs, queue_size=5, slop=0.05)
        self.sync.registerCallback(self.cb)
        self.get_logger().info(f"Warte auf {color_topic} + {depth_topic} ...")

    def cb(self, cmsg, dmsg):
        with self.lock:
            self.latest = (cmsg, dmsg)  # nur das neueste Paar behalten

    def take(self):
        with self.lock:
            m, self.latest = self.latest, None
        return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", default="/camera/color/image_raw")
    ap.add_argument("--depth", default="/camera/depth/image_raw")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--min", type=float, default=None, help="Farbskala min [m] (sonst auto)")
    ap.add_argument("--max", type=float, default=None, help="Farbskala max [m] (sonst auto)")
    ap.add_argument("--width", type=int, default=640, help="Breite je Panel in der Anzeige")
    ap.add_argument("--compile", default="reduce-overhead",
                    choices=["off", "default", "reduce-overhead", "max-autotune"],
                    help="torch.compile-Modus (off = aus)")
    a = ap.parse_args()

    print("Lade Modell ...")
    filler = DepthFiller(a.model, compile_mode=a.compile)
    if a.compile != "off":
        print(f"torch.compile ({a.compile}) aktiv - erster Frame kann ~1 min dauern.")

    rclpy.init()
    node = Listener(a.color, a.depth)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    show_mask, fps, last = True, 0.0, time.perf_counter()
    win = "RGB | Rohtiefe | gefuellt"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while rclpy.ok():
            pair = node.take()
            if pair is None:
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue

            rgb = color_to_rgb(pair[0])
            depth = depth_to_m(pair[1])
            H, W = rgb.shape[:2]
            if depth.shape != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

            t = time.perf_counter()
            try:
                filled, est = filler(rgb, depth)
            except ValueError as e:  # z. B. Frame ganz ohne Tiefe
                node.get_logger().warn(str(e))
                continue
            infer_ms = 1000 * (time.perf_counter() - t)

            # gemeinsame Farbskala für beide Tiefenbilder
            v = depth[depth > 0]
            dmin = a.min if a.min is not None else (np.percentile(v, 2) if v.size else 0.0)
            dmax = a.max if a.max is not None else (np.percentile(v, 98) if v.size else 5.0)

            p_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            p_raw = colorize(depth, dmin, dmax)
            p_fill = colorize(filled, dmin, dmax)
            if show_mask:  # Rand der geschätzten Bereiche weiß umranden
                cnts, _ = cv2.findContours(est.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(p_fill, cnts, -1, (255, 255, 255), 1)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now

            h = int(H * a.width / W)
            panels = [cv2.resize(p, (a.width, h)) for p in (p_rgb, p_raw, p_fill)]
            label(panels[0], "RGB")
            label(panels[1], f"Rohtiefe ({dmin:.2f}-{dmax:.2f} m), Loecher: {est.mean():.0%}")
            label(panels[2], f"gefuellt  {infer_ms:.0f} ms  {fps:.1f} Hz")
            cv2.imshow(win, np.hstack(panels))

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("m"):
                show_mask = not show_mask
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()