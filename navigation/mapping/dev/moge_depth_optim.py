#!/usr/bin/env python3
"""
Löcher in Sensortiefe mit MoGe-2 füllen (ROS 2). Anzeige: RGB | Rohtiefe | gefüllt + Fehler.

1. MoGe-2 schätzt metrische Tiefe (mit echter Brennweite aus camera_info).
2. Globaler Faktor: Median(Messung / MoGe), über die Zeit geglättet.
3. Gradienten-Fusion: Korrekturfeld c (log D = log M + c), das die Messungen trifft, entlang
   von Oberflächen glatt ist (MoGe gibt die Form vor), nicht über Kanten wirkt und hinter dem
   Messbereich auf 0 abklingt. Gemessene Pixel bleiben unverändert.
Fehler: pro Frame werden zufällige Messblöcke ausgeblendet, gefüllt und verglichen.
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
from collections import deque

import cv2
import numpy as np
import rclpy
import torch
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

MODEL = "Ruicheng/moge-2-vitl"
NUM_TOKENS = 1200
COLOR_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_raw"
INFO_TOPIC = "/camera/color/camera_info"
PANEL_WIDTH = 640
BINS = [0, 1, 2, 3, 4, 6, 100]  # Distanzklassen [m] für die Fehlerbalken
PRINT_EVERY = 10    # alle N Frames Geschwindigkeit + Zeiten in die Konsole
USE_GPU_FUSION = True   # Fusion auf der GPU (gleiche Mathematik wie die CPU-Version)


# ---------------------------------------------------------------- Gradienten-Fusion

def _block_stats(z, mask, f):
    """Maskierter Mittelwert, Varianz und Abdeckung je f×f-Block."""
    H, W = z.shape
    h, w = H // f, W // f
    m = mask[: h * f, : w * f].astype(np.float32)
    zm = z[: h * f, : w * f].astype(np.float32) * m
    area = lambda a: cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    frac = area(m)
    fi = 1.0 / np.maximum(frac, 1e-6)
    mean = area(zm) * fi
    var = np.maximum(area(zm * zm) * fi - mean ** 2, 0)
    mean[frac == 0] = 0
    return mean, var, frac


class GradientFusion:
    def __init__(self, down=4, lam=1.0, mu_near=1e-4, mu_far=0.005, far_band=0.15,
                 sigma_edge=0.02, z_ref=4.0, outlier=1.0, iters=80, tol=1e-4):
        self.down, self.lam = down, lam
        self.mu_near, self.mu_far, self.far_band = mu_near, mu_far, far_band
        self.sigma_edge, self.z_ref, self.outlier = sigma_edge, z_ref, outlier
        self.iters, self.tol = iters, tol
        self._c_prev = None
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
        p = z.copy()
        rz = float((r * z).sum())
        bn = float(np.sqrt((b * b).sum())) + 1e-12
        for _ in range(self.iters):
            Ap = self._apply(p, diag, wx, wy)
            alpha = rz / max(float((p * Ap).sum()), 1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            if np.sqrt((r * r).sum()) < self.tol * bn:
                break
            z = r / diag
            rz_new = float((r * z).sum())
            p = z + (rz_new / max(rz, 1e-30)) * p
            rz = rz_new
        return x

    def __call__(self, depth_m, est):
        f = self.down
        H, W = depth_m.shape
        pv = est > 0
        both = (depth_m > 0) & pv
        logM = np.zeros(est.shape, np.float32)
        logM[pv] = np.log(est[pv])
        logZ = np.zeros_like(logM)
        logZ[both] = np.log(depth_m[both])
        r_full = np.where(both, logZ - logM, 0).astype(np.float32)

        gM, gMvar, gMfrac = _block_stats(logM, pv, f)
        r, rvar, frac = _block_stats(r_full, both, f)
        zlog, _, _ = _block_stats(logZ, both, f)

        # Messgewichte: Abdeckung, Sensorgenauigkeit ~1/z², Kanten/Streuung, Ausreißer
        wm = frac * np.minimum(1.0, np.exp(2 * (np.log(self.z_ref) - zlog)))
        wm *= np.exp(-(np.sqrt(gMvar) + np.sqrt(rvar)) / self.sigma_edge)
        wm[frac < 0.2] = 0
        has = wm > 0
        if has.any():
            wm *= 1.0 / (1.0 + ((r - np.median(r[has])) / self.outlier) ** 2)
            t = np.clip((gM - np.percentile(zlog[has], 98)) / self.far_band, 0, 1)
        else:
            t = np.ones_like(gM)
        mu = self.mu_near + (self.mu_far - self.mu_near) * t   # Abklingen hinter Messbereich

        # kantenbewusste Glattheit
        ok = gMfrac > 0.5
        wx = self.lam * np.exp(-np.abs(np.diff(gM, axis=1)) / self.sigma_edge) * (ok[:, 1:] & ok[:, :-1])
        wy = self.lam * np.exp(-np.abs(np.diff(gM, axis=0)) / self.sigma_edge) * (ok[1:, :] & ok[:-1, :])
        deg = np.zeros_like(gM)
        deg[:, :-1] += wx
        deg[:, 1:] += wx
        deg[:-1, :] += wy
        deg[1:, :] += wy

        x = self._c_prev if (self._c_prev is not None and self._c_prev.shape == gM.shape) \
            else np.zeros_like(gM)
        x = self._cg(wm * r, wm + mu + deg, wx, wy, x)
        wm2 = wm * np.minimum(1.0, self.outlier / np.maximum(np.abs(r - x), 1e-9))  # Huber
        x = self._cg(wm2 * r, wm2 + mu + deg, wx, wy, x)
        self._c_prev = x

        H2, W2 = H // 2, W // 2
        logM2 = cv2.resize(logM, (W2, H2), interpolation=cv2.INTER_NEAREST)
        c_half = self._guided_upsample(x.astype(np.float32), gM, logM2, H2, W2)
        out = est * np.exp(cv2.resize(c_half, (W, H), interpolation=cv2.INTER_LINEAR))
        out[~pv] = 0
        return out.astype(np.float32)

    def _guided_upsample(self, c, gM, logM, H, W):
        """Bilinear, Gitterpunkte mit ähnlicher MoGe-Tiefe zählen mehr (kantentreu)."""
        h, w = c.shape
        if self._up_cache is None or self._up_cache[0] != (h, w, H, W):
            ys = (np.arange(H, dtype=np.float32) + 0.5) * h / H - 0.5
            xs = (np.arange(W, dtype=np.float32) + 0.5) * w / W - 0.5
            y0 = np.clip(np.floor(ys).astype(np.int32), 0, h - 1)
            x0 = np.clip(np.floor(xs).astype(np.int32), 0, w - 1)
            y1, x1 = np.clip(y0 + 1, 0, h - 1), np.clip(x0 + 1, 0, w - 1)
            fy = np.clip(ys - np.floor(ys), 0, 1)[:, None]
            fx = np.clip(xs - np.floor(xs), 0, 1)[None, :]
            tabs = [((yy[:, None] * w + xx[None, :]).astype(np.int32), (wy * wx).astype(np.float32))
                    for yy, wy in ((y0, 1 - fy), (y1, fy)) for xx, wx in ((x0, 1 - fx), (x1, fx))]
            self._up_cache = ((h, w, H, W), tabs)
        cf, gf = c.ravel(), gM.astype(np.float32).ravel()
        acc = np.zeros((H, W), np.float32)
        wsum = np.full((H, W), 1e-12, np.float32)
        for idx, bw in self._up_cache[1]:
            d = np.abs(logM - gf[idx]) / self.sigma_edge
            wgt = bw / (1.0 + d * d)
            acc += wgt * cf[idx]
            wsum += wgt
        return acc / wsum


def fill(depth_m, pred, fusion, prev_log_s=None, ema=0.3):
    """Gibt (gefüllt, Maske geschätzt, log_s) zurück. depth_m: 0 = Loch, pred: inf = ungültig."""
    valid = depth_m > 0
    pvalid = np.isfinite(pred) & (pred > 0)
    sub = (valid & pvalid)[::4, ::4]
    log_s = prev_log_s if prev_log_s is not None else 0.0
    if sub.sum() > 200:
        cur = float(np.log(np.median(depth_m[::4, ::4][sub] / pred[::4, ::4][sub])))
        log_s = cur if prev_log_s is None else (1 - ema) * prev_log_s + ema * cur
    est = np.where(pvalid, pred * math.exp(log_s), 0).astype(np.float32)
    est = fusion(depth_m, est)
    filled = np.where(valid, depth_m, est)
    return filled, ~valid & (filled > 0), log_s


# ---------------------------------------------------------------- Gradienten-Fusion auf der GPU
# Gleiche Mathematik wie GradientFusion/fill oben, nur mit torch-Tensoren auf der GPU.

import torch.nn.functional as F


def _block_stats_t(z, m, f):
    """Wie _block_stats: avg_pool mit ganzzahligem Faktor = cv2 INTER_AREA."""
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


class GradientFusionTorch(GradientFusion):
    """Gleiche Parameter wie GradientFusion; Eingaben und Ausgabe sind torch-Tensoren."""

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

        gM, gMvar, gMfrac = _block_stats_t(logM, pv, f)
        r, rvar, frac = _block_stats_t(r_full, both, f)
        zlog, _, _ = _block_stats_t(logZ, both, f)

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
        mu = self.mu_near + (self.mu_far - self.mu_near) * t

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
        wm2 = wm * torch.clamp(self.outlier / torch.clamp(torch.abs(r - x), min=1e-9), max=1.0)
        x = self._cg(wm2 * r, wm2 + mu + deg, wx, wy, x)
        self._c_prev = x

        H2, W2 = H // 2, W // 2
        logM2 = logM[: 2 * H2: 2, : 2 * W2: 2]          # = cv2 INTER_NEAREST bei Faktor 2
        c_half = self._guided_upsample(x, gM, logM2, H2, W2)
        c_full = F.interpolate(c_half[None, None], size=(H, W), mode="bilinear",
                               align_corners=False)[0, 0]
        return torch.where(pv, est * torch.exp(c_full), zero)

    def _guided_upsample(self, c, gM, logM, H, W):
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
def fill_torch(depth_m, pred, fusion, prev_log_s=None, ema=0.3):
    """Wie fill(), aber auf der GPU. depth_m/pred: torch-Tensoren (float32, gleiches Gerät).
    Gibt (gefüllt, Maske geschätzt, log_s) als numpy zurück."""
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
    filled = torch.where(valid, depth_m, est)
    filled_np = filled.cpu().numpy()
    return filled_np, (~valid).cpu().numpy() & (filled_np > 0), log_s


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


class Listener(Node):
    def __init__(self):
        super().__init__("moge_depth_fill")
        self.lock, self.latest, self.fx = threading.Lock(), None, None
        subs = [Subscriber(self, Image, COLOR_TOPIC, qos_profile=qos_profile_sensor_data),
                Subscriber(self, Image, DEPTH_TOPIC, qos_profile=qos_profile_sensor_data)]
        self.sync = ApproximateTimeSynchronizer(subs, queue_size=5, slop=0.05)
        self.sync.registerCallback(self.cb)
        self.create_subscription(CameraInfo, INFO_TOPIC,
                                 lambda m: setattr(self, "fx", (m.k[0], m.width)),
                                 qos_profile_sensor_data)

    def cb(self, c, d):
        with self.lock:
            self.latest = (c, d)

    def take(self):
        with self.lock:
            m, self.latest = self.latest, None
        return m


# ---------------------------------------------------------------- Anzeige

def colorize(d, dmin, dmax):
    v = d > 0
    n = np.clip((np.where(v, d, 0) - dmin) / max(dmax - dmin, 1e-6), 0, 1)
    img = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~v] = 0
    return img


def text(img, s, y=30, scale=0.7):
    for col, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def holdout_mask(valid, rng, block=32, frac=0.15):
    H, W = valid.shape
    pick = rng.random((-(-H // block), -(-W // block))) < frac
    return np.kron(pick, np.ones((block, block), bool))[:H, :W] & valid


def metrics(pred, gt, mask):
    m = mask & (pred > 0)
    if m.sum() < 50:
        return None
    e = np.abs(pred[m] - gt[m])
    rel = e / gt[m]
    bins = [float(rel[s].mean()) if (s := (gt[m] >= lo) & (gt[m] < hi)).sum() >= 20 else np.nan
            for lo, hi in zip(BINS[:-1], BINS[1:])]
    return {"absrel": float(rel.mean()), "mae": float(e.mean()), "bins": np.array(bins)}


def draw_metrics(width, hist, cur, height=220):
    img = np.full((height, width, 3), 30, np.uint8)
    col, grey = (110, 220, 110), (180, 180, 180)
    x0, x1, y0, y1 = 45, width // 2 - 15, 35, height - 30

    # Fehler über die Zeit
    h = [v for v in hist if np.isfinite(v)]
    ymax = max(0.02, 1.15 * max(h)) if h else 0.1
    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 90), 1)
    cv2.putText(img, f"{100 * ymax:.0f}%", (4, y0 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, grey, 1)
    xs = np.linspace(x0, x1, hist.maxlen)[-len(hist):]
    pts = [(int(x), int(y1 - min(v, ymax) / ymax * (y1 - y0))) for x, v in zip(xs, hist)
           if np.isfinite(v)]
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
    title = "Fehler ueber Zeit (AbsRel)"
    if cur:
        title += f"   aktuell: AbsRel {100 * cur['absrel']:.1f}%   MAE {100 * cur['mae']:.1f} cm"
    cv2.putText(img, title, (x0, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    # Fehler je Distanz
    bx0, bx1 = width // 2 + 45, width - 15
    nb = len(BINS) - 1
    b = cur["bins"] if cur else np.full(nb, np.nan)
    bmax = max(0.02, 1.15 * np.nanmax(b)) if np.isfinite(b).any() else 0.1
    cv2.rectangle(img, (bx0, y0), (bx1, y1), (90, 90, 90), 1)
    cw = (bx1 - bx0) / nb
    for i in range(nb):
        lab = f"{BINS[i]}-{BINS[i + 1]}" if BINS[i + 1] < 100 else f">{BINS[i]}"
        cv2.putText(img, lab, (int(bx0 + (i + 0.3) * cw), height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, grey, 1)
        if np.isfinite(b[i]):
            yb = int(y1 - min(b[i], bmax) / bmax * (y1 - y0))
            cv2.rectangle(img, (int(bx0 + (i + 0.2) * cw), yb), (int(bx0 + (i + 0.8) * cw), y1), col, -1)
            cv2.putText(img, f"{100 * b[i]:.1f}%", (int(bx0 + (i + 0.2) * cw), yb - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, grey, 1)
    cv2.putText(img, "Fehler je Distanz [m] (AbsRel)", (bx0, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (230, 230, 230), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- Hauptschleife

def main():
    print("Lade MoGe-2 ...")
    moge = MoGe()
    rclpy.init()
    node = Listener()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    use_gpu = USE_GPU_FUSION and moge.dev.type == "cuda"
    FusionCls = GradientFusionTorch if use_gpu else GradientFusion
    fusion, fusion_eval = FusionCls(), FusionCls()
    rng = np.random.default_rng(0)
    hist, cur = deque(maxlen=300), None
    log_s, fps, last = None, 0.0, time.perf_counter()
    t_sum, n_sum = {k: 0.0 for k in ("Konvert.", "MoGe", "Fusion", "Auswert.", "Vis-Vorb.",
                                     "Vis-Anzeige")}, 0

    def tick(key, t0):
        t1 = time.perf_counter()
        t_sum[key] += 1000 * (t1 - t0)
        return t1

    cv2.namedWindow("MoGe depth fill", cv2.WINDOW_NORMAL)
    try:
        while rclpy.ok():
            pair = node.take()
            if pair is None:
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue
            t = time.perf_counter()
            rgb, depth = to_rgb(pair[0]), to_depth(pair[1])
            H, W = rgb.shape[:2]
            if depth.shape != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            fov_x = math.degrees(2 * math.atan(node.fx[1] / (2 * node.fx[0]))) if node.fx else None

            t = tick("Konvert.", t)

            pred_t = moge(rgb, fov_x)
            if use_gpu:
                torch.cuda.synchronize()
            t = tick("MoGe", t)

            if use_gpu:
                depth_t = torch.from_numpy(depth).to(pred_t.device)
                filled, est, log_s = fill_torch(depth_t, pred_t, fusion, log_s)
            else:
                pred = pred_t.cpu().numpy()
                filled, est, log_s = fill(depth, pred, fusion, log_s)
            t = tick("Fusion", t)

            # Fehler: Messblöcke ausblenden, füllen, mit Messung vergleichen
            ho = holdout_mask(depth > 0, rng)
            d_ho = np.where(ho, 0, depth).astype(np.float32)
            if use_gpu:
                f_ho, _, _ = fill_torch(torch.from_numpy(d_ho).to(pred_t.device), pred_t,
                                        fusion_eval, log_s)
            else:
                f_ho, _, _ = fill(d_ho, pred, fusion_eval, log_s)
            m = metrics(f_ho, depth, ho)
            if m:
                cur = m if cur is None else {k: 0.8 * cur[k] + 0.2 * m[k] if k != "bins" else
                                             np.where(np.isfinite(cur[k]),
                                                      np.where(np.isfinite(m[k]), 0.8 * cur[k] + 0.2 * m[k], cur[k]),
                                                      m[k]) for k in m}
            hist.append(m["absrel"] if m else np.nan)
            t = tick("Auswert.", t)

            now = time.perf_counter()
            fps, last = 0.9 * fps + 0.1 / max(now - last, 1e-6), now

            v = depth[depth > 0]
            dmin = float(np.percentile(v, 2)) if v.size else 0.0
            dmax = float(np.percentile(filled[filled > 0], 98)) if (filled > 0).any() else 5.0
            p_fill = colorize(filled, dmin, dmax)
            cnts, _ = cv2.findContours(est.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(p_fill, cnts, -1, (255, 255, 255), 1)

            h = int(H * PANEL_WIDTH / W)
            panels = [cv2.resize(p, (PANEL_WIDTH, h)) for p in
                      (rgb[..., ::-1], colorize(depth, dmin, dmax), p_fill)]
            text(panels[0], "RGB" + ("" if fov_x else " (keine camera_info!)"))
            text(panels[1], f"Rohtiefe {dmin:.1f}-{dmax:.1f} m, Loecher {(depth <= 0).mean():.0%}")
            text(panels[2], f"gefuellt  {fps:.1f} Hz  s={math.exp(log_s):.2f}")
            view = np.vstack([np.hstack(panels), draw_metrics(3 * PANEL_WIDTH, hist, cur)])
            t = tick("Vis-Vorb.", t)
            cv2.imshow("MoGe depth fill", view)
            key = cv2.waitKey(1) & 0xFF
            tick("Vis-Anzeige", t)

            n_sum += 1
            if n_sum == PRINT_EVERY:   # Mittelwerte der letzten Frames in die Konsole
                a = {k: v / n_sum for k, v in t_sum.items()}
                pipe = a["Konvert."] + a["MoGe"] + a["Fusion"]
                ohne_vis = pipe + a["Auswert."]
                mit_vis = ohne_vis + a["Vis-Vorb."] + a["Vis-Anzeige"]
                print(f"Hz: mit Visualisierung {1000 / mit_vis:5.1f} | "
                      f"ohne Visualisierung {1000 / ohne_vis:5.1f} | "
                      f"nur MoGe + Fusion {1000 / pipe:5.1f}   ||   "
                      + " | ".join(f"{k} {v:.1f}" for k, v in a.items()) + " ms", flush=True)
                t_sum, n_sum = dict.fromkeys(t_sum, 0.0), 0
            if key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()