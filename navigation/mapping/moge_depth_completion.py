#!/usr/bin/env python3
r"""
Step 3: MoGe-2 mono depth completion (ROS 2, GPU, no visualisation).

Subscribes the raw camera topics of ln_ros_camera_bridge.py, fills the holes of the
sensor depth and republishes color + completed depth + camera_info with the
ORIGINAL header (stamp = LN capture timestamp, frame_id = camera):

    /camera/color/image_raw    \                     /camera/completed/color/image_raw
    /camera/depth/image_raw     >-- MoGe + Fusion -->  /camera/completed/depth/image_raw (16UC1, mm)
    /camera/color/camera_info  /                     /camera/completed/camera_info

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

import cv2
import numpy as np
import rclpy
import torch
import torch.nn.functional as F
from message_filters import Subscriber, TimeSynchronizer
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

MODEL = "Ruicheng/moge-2-vitl"
NUM_TOKENS = 1200
COLOR_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_raw"
INFO_TOPIC = "/camera/color/camera_info"
OUT_COLOR_TOPIC = "/camera/completed/color/image_raw"
OUT_DEPTH_TOPIC = "/camera/completed/depth/image_raw"
OUT_INFO_TOPIC = "/camera/completed/camera_info"
PRINT_EVERY = 30    # alle N Frames Geschwindigkeit + Zeiten in die Konsole


# ---------------------------------------------------------------- Gradienten-Fusion (GPU)

def _block_stats(z, m, f):
    """Maskierter Mittelwert, Varianz und Abdeckung je f×f-Block (avg_pool = cv2 INTER_AREA)."""
    H, W = z.shape
    h, w = H // f, W // f
    m = m[: h * f, : w * f].float()
    zm = z[: h * f, : w * f] * m
    pool = lambda a: F.avg_pool2d(a[None, None], f, f)[0, 0]
    frac = pool(m)
    fi = 1.0 / torch.clamp(frac, min=1e-6)
    mean = pool(zm) * fi
    var = torch.clamp(pool(zm * zm) * fi - mean ** 2, min=0)
    mean = torch.where(frac == 0, torch.zeros_like(mean), mean)
    return mean, var, frac


class GradientFusion:
    """Eingaben und Ausgabe sind torch-Tensoren (float32, gleiches Gerät)."""

    def __init__(self, down=4, lam=1.0, mu_near=1e-4, mu_far=0.005, far_band=0.15,
                 sigma_edge=0.02, z_ref=4.0, outlier=1.0, iters=80, tol=1e-4):
        self.down, self.lam = down, lam
        self.mu_near, self.mu_far, self.far_band = mu_near, mu_far, far_band
        self.sigma_edge, self.z_ref, self.outlier = sigma_edge, z_ref, outlier
        self.iters, self.tol = iters, tol
        self._c_prev = None      # Warmstart mit dem Korrekturfeld des letzten Frames
        self._up_cache = None

    @staticmethod
    def _apply(c, diag, wx, wy):
        out = diag * c
        out[:, :-1] -= wx * c[:, 1:]
        out[:, 1:] -= wx * c[:, :-1]
        out[:-1, :] -= wy * c[1:, :]
        out[1:, :] -= wy * c[:-1, :]
        return out

    def _cg(self, b, diag, wx, wy, x):
        r = b - self._apply(x, diag, wx, wy)
        z = r / diag
        p = z.clone()
        rz = (r * z).sum()
        bn = torch.sqrt((b * b).sum()) + 1e-12
        for _ in range(self.iters):
            Ap = self._apply(p, diag, wx, wy)
            alpha = rz / torch.clamp((p * Ap).sum(), min=1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            if bool(torch.sqrt((r * r).sum()) < self.tol * bn):
                break
            z = r / diag
            rz_new = (r * z).sum()
            p = z + (rz_new / torch.clamp(rz, min=1e-30)) * p
            rz = rz_new
        return x

    def __call__(self, depth_m, est):
        f = self.down
        H, W = depth_m.shape
        pv = est > 0
        both = (depth_m > 0) & pv
        zero = torch.zeros_like(est)
        logM = torch.where(pv, torch.log(torch.where(pv, est, torch.ones_like(est))), zero)
        logZ = torch.where(both, torch.log(torch.where(both, depth_m, torch.ones_like(est))), zero)
        r_full = torch.where(both, logZ - logM, zero)

        gM, gMvar, gMfrac = _block_stats(logM, pv, f)
        r, rvar, frac = _block_stats(r_full, both, f)
        zlog, _, _ = _block_stats(logZ, both, f)

        # Messgewichte: Abdeckung, Sensorgenauigkeit ~1/z², Kanten/Streuung, Ausreißer
        wm = frac * torch.clamp(torch.exp(2 * (math.log(self.z_ref) - zlog)), max=1.0)
        wm = wm * torch.exp(-(torch.sqrt(gMvar) + torch.sqrt(rvar)) / self.sigma_edge)
        wm = torch.where(frac < 0.2, torch.zeros_like(wm), wm)
        has = wm > 0
        if bool(has.any()):
            rh, zh = r[has], zlog[has]
            med = torch.quantile(rh, 0.5) if rh.numel() <= 16_000_000 else rh.median()
            wm = wm * (1.0 / (1.0 + ((r - med) / self.outlier) ** 2))
            t = torch.clamp((gM - torch.quantile(zh, 0.98)) / self.far_band, 0, 1)
        else:
            t = torch.ones_like(gM)
        mu = self.mu_near + (self.mu_far - self.mu_near) * t   # Abklingen hinter Messbereich

        # kantenbewusste Glattheit
        ok = gMfrac > 0.5
        wx = self.lam * torch.exp(-torch.abs(torch.diff(gM, dim=1)) / self.sigma_edge) \
            * (ok[:, 1:] & ok[:, :-1])
        wy = self.lam * torch.exp(-torch.abs(torch.diff(gM, dim=0)) / self.sigma_edge) \
            * (ok[1:, :] & ok[:-1, :])
        deg = torch.zeros_like(gM)
        deg[:, :-1] += wx
        deg[:, 1:] += wx
        deg[:-1, :] += wy
        deg[1:, :] += wy

        x = self._c_prev if (self._c_prev is not None and self._c_prev.shape == gM.shape) \
            else torch.zeros_like(gM)
        x = self._cg(wm * r, wm + mu + deg, wx, wy, x)
        wm2 = wm * torch.clamp(self.outlier / torch.clamp(torch.abs(r - x), min=1e-9), max=1.0)  # Huber
        x = self._cg(wm2 * r, wm2 + mu + deg, wx, wy, x)
        self._c_prev = x

        H2, W2 = H // 2, W // 2
        logM2 = logM[: 2 * H2: 2, : 2 * W2: 2]          # = cv2 INTER_NEAREST bei Faktor 2
        c_half = self._guided_upsample(x, gM, logM2, H2, W2)
        c_full = F.interpolate(c_half[None, None], size=(H, W), mode="bilinear",
                               align_corners=False)[0, 0]
        return torch.where(pv, est * torch.exp(c_full), zero)

    def _guided_upsample(self, c, gM, logM, H, W):
        """Bilinear, Gitterpunkte mit ähnlicher MoGe-Tiefe zählen mehr (kantentreu)."""
        h, w = c.shape
        key = (h, w, H, W, str(c.device))
        if self._up_cache is None or self._up_cache[0] != key:
            ys = (torch.arange(H, dtype=torch.float32, device=c.device) + 0.5) * h / H - 0.5
            xs = (torch.arange(W, dtype=torch.float32, device=c.device) + 0.5) * w / W - 0.5
            y0 = torch.clamp(torch.floor(ys).long(), 0, h - 1)
            x0 = torch.clamp(torch.floor(xs).long(), 0, w - 1)
            y1, x1 = torch.clamp(y0 + 1, 0, h - 1), torch.clamp(x0 + 1, 0, w - 1)
            fy = torch.clamp(ys - torch.floor(ys), 0, 1)[:, None]
            fx = torch.clamp(xs - torch.floor(xs), 0, 1)[None, :]
            tabs = [(yy[:, None] * w + xx[None, :], wy * wx)
                    for yy, wy in ((y0, 1 - fy), (y1, fy)) for xx, wx in ((x0, 1 - fx), (x1, fx))]
            self._up_cache = (key, tabs)
        cf, gf = c.reshape(-1), gM.reshape(-1)
        acc = torch.zeros((H, W), dtype=torch.float32, device=c.device)
        wsum = torch.full((H, W), 1e-12, dtype=torch.float32, device=c.device)
        for idx, bw in self._up_cache[1]:
            d = torch.abs(logM - gf[idx]) / self.sigma_edge
            wgt = bw / (1.0 + d * d)
            acc += wgt * cf[idx]
            wsum += wgt
        return acc / wsum


@torch.inference_mode()
def fill(depth_m, pred, fusion, prev_log_s=None, ema=0.3):
    """Gibt (gefüllt [m], log_s) zurück. depth_m: 0 = Loch, pred: inf = ungültig (Tensoren)."""
    valid = depth_m > 0
    pvalid = torch.isfinite(pred) & (pred > 0)
    sub = (valid & pvalid)[::4, ::4]
    log_s = prev_log_s if prev_log_s is not None else 0.0
    if int(sub.sum()) > 200:
        ratio = depth_m[::4, ::4][sub] / pred[::4, ::4][sub]
        cur = float(torch.log(torch.quantile(ratio, 0.5) if ratio.numel() <= 16_000_000
                              else ratio.median()))
        log_s = cur if prev_log_s is None else (1 - ema) * prev_log_s + ema * cur
    est = torch.where(pvalid, pred * math.exp(log_s), torch.zeros_like(pred))
    est = fusion(depth_m, est)
    return torch.where(valid, depth_m, est), log_s


# ---------------------------------------------------------------- MoGe

class MoGe:
    def __init__(self):
        from moge.model.v2 import MoGeModel
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MoGeModel.from_pretrained(MODEL).to(self.dev).eval()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    @torch.inference_mode()
    def __call__(self, rgb, fov_x):
        img = torch.from_numpy(rgb).to(self.dev).permute(2, 0, 1).float() / 255.0
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.dev.type == "cuda"):
            out = self.model.infer(img, fov_x=fov_x, num_tokens=NUM_TOKENS)
        d = out["depth"].float()
        if "mask" in out:
            d = torch.where(out["mask"].bool(), d, torch.full_like(d, float("inf")))
        return d          # torch-Tensor auf dem MoGe-Gerät


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


class MoGeDepthCompletion(Node):
    def __init__(self):
        super().__init__("moge_depth_completion")
        self.lock, self.latest = threading.Lock(), None
        self.new_frame = threading.Event()
        # Color, depth and info of one frame carry the identical LN stamp -> exact sync
        subs = [Subscriber(self, Image, COLOR_TOPIC, qos_profile=qos_profile_sensor_data),
                Subscriber(self, Image, DEPTH_TOPIC, qos_profile=qos_profile_sensor_data),
                Subscriber(self, CameraInfo, INFO_TOPIC, qos_profile=qos_profile_sensor_data)]
        self.sync = TimeSynchronizer(subs, queue_size=5)
        self.sync.registerCallback(self.cb)

        self.pub_color = self.create_publisher(Image, OUT_COLOR_TOPIC, 1)
        self.pub_depth = self.create_publisher(Image, OUT_DEPTH_TOPIC, 1)
        self.pub_info = self.create_publisher(CameraInfo, OUT_INFO_TOPIC, 1)

    def cb(self, color, depth, info):
        with self.lock:
            self.latest = (color, depth, info)   # ältere, noch nicht verarbeitete Frames verwerfen
        self.new_frame.set()

    def take(self, timeout=0.5):
        if not self.new_frame.wait(timeout):
            return None
        with self.lock:
            m, self.latest = self.latest, None
            self.new_frame.clear()
        return m

    def publish(self, color_msg, depth_mm, info_msg, header):
        # Color and info are forwarded unchanged; the completed depth gets the original header
        self.pub_color.publish(color_msg)
        self.pub_depth.publish(depth_to_msg(depth_mm, header))
        self.pub_info.publish(info_msg)


# ---------------------------------------------------------------- Hauptschleife

def main():
    print("Lade MoGe-2 ...")
    moge = MoGe()
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
            if frame is None:
                continue
            color_msg, depth_msg, info_msg = frame
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

            filled, log_s = fill(torch.from_numpy(depth).to(pred.device), pred, fusion, log_s)
            depth_mm = (filled * 1000.0).round().clamp(0, 65535).to(torch.int32).cpu().numpy() \
                .astype(np.uint16)
            t = tick("Fusion", t)

            node.publish(color_msg, depth_mm, info_msg, depth_msg.header)
            tick("Publish", t)

            n_sum += 1
            if n_sum == PRINT_EVERY:   # Mittelwerte der letzten Frames in die Konsole
                now = time.perf_counter()
                a = {k: v / n_sum for k, v in t_sum.items()}
                print(f"{n_sum / (now - last):5.1f} Hz | s={math.exp(log_s):.2f} | "
                      + " | ".join(f"{k} {v:.1f}" for k, v in a.items()) + " ms", flush=True)
                t_sum, n_sum, last = dict.fromkeys(t_sum, 0.0), 0, now
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
