#!/usr/bin/env python3
"""
ROS 2 Viewer: RGB | Rohtiefe | gefüllte Tiefe (MoGe-2, metrisch) nebeneinander.

Idee:
  - MoGe-2 schätzt metrische Tiefe fürs ganze Bild (auch dort, wo der Sensor nichts misst).
  - Stufe 1: globaler Skalierungsfaktor (robust per Median, über die Zeit geglättet).
  - Stufe 2: lokal pro Bildbereich eigene Skalierung a und Verschiebung b
    (echt ≈ a·MoGe + b, Least Squares über die gemessenen Pixel im Fenster).
    Ohne Messungen in der Nähe bleibt es bei Stufe 1.
  - Gemessene Pixel bleiben unverändert.

Setup (im conda-Env, Python-Version muss zu ROS passen, z. B. 3.12 für Jazzy):
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install git+https://github.com/microsoft/MoGe.git
  pip install opencv-python "numpy<2"

Start:
  source /opt/ros/<distro>/setup.bash
  python navigation/mapping/dev/moge_depth.py
  python navigation/mapping/dev/moge_depth.py --depth /camera/completed/depth/image_raw
  python navigation/mapping/dev/moge_depth.py --model Ruicheng/moge-2-vits-normal   # schneller, ungenauer
  python navigation/mapping/dev/moge_depth.py --max-range 15                        # Schätzung > 15 m verwerfen
  python navigation/mapping/dev/moge_depth.py --radius 48                           # größere Fenster für Stufe 2
  python navigation/mapping/dev/moge_depth.py --radius 0                            # nur Stufe 1
  python navigation/mapping/dev/moge_depth.py --local-max 3                         # Stufe 2 nur bis 3 m

Geschwindigkeit (Standard: fp16 + num-tokens 1200):
  python navigation/mapping/dev/moge_depth.py --num-tokens 0                        # max. Detail, langsamer
  python navigation/mapping/dev/moge_depth.py --model Ruicheng/moge-2-vitb-normal   # kleineres Modell
  python navigation/mapping/dev/moge_depth.py --show-every 3 --width 400            # weniger Anzeige-Last (X11)
  Taste e (Auswertung aus) spart zusätzlich Zeit.

Tasten im Fenster:  q = beenden,  m = Markierung geschätzter Pixel an/aus,
                    e = Fehlerauswertung an/aus

Fehlerauswertung (unteres Panel): Pro Frame werden zufällige Blöcke GEMESSENER Pixel
ausgeblendet, gefüllt und mit dem echten Messwert verglichen – einmal nur mit globalem
Faktor, einmal mit Stufe 2 (aktuelle --radius/--eps/--local-max). Aussagekräftig nur
im Messbereich des Sensors; für die Ferne ohne Messung gibt es keine Referenz.
  AbsRel = mittlerer relativer Fehler |geschätzt - echt| / echt
  MAE    = mittlerer absoluter Fehler,  RMSE = Wurzel des mittleren quadr. Fehlers
"""
import os
import sys

# Conan-Pakete (u. a. altes numpy) ausblenden, ROS behalten; Modelle auf die Festplatte
sys.path = [p for p in sys.path if "conan_cache" not in p]
_ROS = "/opt/ros/jazzy/lib/python3.12/site-packages"
if os.path.isdir(_ROS) and _ROS not in sys.path:
    sys.path.append(_ROS)
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import argparse
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
# Alternativen: "Ruicheng/moge-2-vitb-normal" (104M), "Ruicheng/moge-2-vits-normal" (35M)


# ---------------------------------------------------------------- Angleichung

def local_affine(depth_m, est, radius_px=16, eps=0.001, min_frac=0.05, down=2):
    """Lokale Angleichung: in jedem Fenster echt ≈ a·est + b (Least Squares nur über
    gemessene Pixel), danach a/b über die Fenster gemittelt (Guided-Filter-Prinzip).
    a = lokale Skalierung, b = lokale Verschiebung [m]. Ohne Messungen in der Nähe: a=1, b=0.
    eps zieht a Richtung 1 (größer = eher reine Verschiebung, robuster bei wenig Messungen)."""
    H, W = depth_m.shape
    h, w = max(H // down, 1), max(W // down, 1)
    y = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
    x = cv2.resize(est, (w, h), interpolation=cv2.INTER_NEAREST)
    M = ((y > 0) & (x > 0)).astype(np.float32)
    x = x * M
    y = y * M
    k = (2 * max(radius_px // down, 1) + 1,) * 2

    def bf(z):
        return cv2.boxFilter(z, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)

    n = bf(M)                                   # Anteil gemessener Pixel im Fenster
    ni = 1.0 / np.maximum(n, 1e-6)
    mx, my = bf(x) * ni, bf(y) * ni
    var = bf(x * x) * ni - mx * mx
    cov = bf(x * y) * ni - mx * my
    a = np.clip((cov + eps) / (var + eps), 0.25, 4.0)
    b = my - a * mx
    conf = np.clip(n / min_frac, 0, 1)          # zu wenige Messungen -> neutral
    a = 1 + conf * (a - 1)
    b = conf * b
    cw = bf(conf)                               # a/b über überlappende Fenster mitteln
    cwi = 1.0 / np.maximum(cw, 1e-6)
    fade = np.clip(cw / min_frac, 0, 1)
    a_m = 1 + fade * (bf(a * conf) * cwi - 1)
    b_m = fade * (bf(b * conf) * cwi)
    a_f = cv2.resize(a_m, (W, H), interpolation=cv2.INTER_LINEAR)
    b_f = cv2.resize(b_m, (W, H), interpolation=cv2.INTER_LINEAR)
    return a_f * est + b_f


def align_and_fill(depth_m, pred, prev_log_s=None, ema=0.3, radius_px=32, eps=0.01,
                   max_range=0.0, local_max=0.0, local_band=1.0):
    """depth_m: Messung [m], 0 = Loch.  pred: MoGe-Tiefe [m] (inf/0 = ungültig, z. B. Himmel).
    Gibt (gefüllt, Maske geschätzt, log_s) zurück."""
    valid = depth_m > 0
    pvalid = np.isfinite(pred) & (pred > 0)
    both = valid & pvalid

    # 1) globaler Skalierungsfaktor (robust: Median im log-Raum, auf jedem 4. Pixel),
    #    zeitlich geglättet
    sub = both[::4, ::4]
    if sub.sum() > 200:
        ratio = depth_m[::4, ::4][sub] / pred[::4, ::4][sub]
        log_s = float(np.log(np.median(ratio)))
        if prev_log_s is not None:
            log_s = (1 - ema) * prev_log_s + ema * log_s
    else:
        log_s = prev_log_s if prev_log_s is not None else 0.0
    est = pred.astype(np.float32) * np.float32(math.exp(log_s))
    est[~pvalid] = 0

    # 2) lokale Skalierung + Verschiebung nahe an Messungen
    if radius_px > 0:
        loc = local_affine(depth_m, est, radius_px, eps)
        if local_max > 0:
            # nur bis local_max voll lokal, dann über `local_band` Meter Übergang zu global
            wgt = np.clip((local_max + local_band - est) / local_band, 0, 1)
            loc = wgt * loc + (1 - wgt) * est
        est = np.where(pvalid, loc, 0).astype(np.float32)

    filled = np.where(valid, depth_m, est)
    estimated = ~valid & (filled > 0)
    if max_range > 0:
        drop = estimated & (filled > max_range)
        filled[drop] = 0
        estimated &= ~drop
    return filled, estimated, log_s


# ---------------------------------------------------------------- Modell

class MoGeDepth:
    def __init__(self, model_id=MODEL, num_tokens=None, fp16=True):
        from moge.model.v2 import MoGeModel
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MoGeModel.from_pretrained(model_id).to(self.device).eval()
        self.num_tokens = num_tokens
        self.fp16 = fp16 and self.device.type == "cuda"
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True   # schnellere Matrixmult. (Ampere+)
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        self._use_fp16_arg = True  # ob infer() den Parameter use_fp16 kennt

    @torch.inference_mode()
    def __call__(self, rgb, fov_x=None):
        img = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float() / 255.0
        kw = dict(fov_x=fov_x, num_tokens=self.num_tokens)
        out = None
        if self._use_fp16_arg:
            try:
                out = self.model.infer(img, use_fp16=self.fp16, **kw)
            except TypeError:          # ältere/andere Version ohne use_fp16
                self._use_fp16_arg = False
        if out is None:
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                out = self.model.infer(img, **kw)
        d = out["depth"].float().cpu().numpy()
        if "mask" in out:
            d[~out["mask"].cpu().numpy().astype(bool)] = np.inf
        return d


# ---------------------------------------------------------------- ROS-Hilfen

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
        return d.astype(np.float32) * 0.001
    if msg.encoding == "32FC1":
        d = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.step // 4)[:, : msg.width]
        return np.nan_to_num(d.copy(), nan=0.0, posinf=0.0)
    raise ValueError(f"Tiefen-Encoding {msg.encoding} nicht unterstützt")


def colorize(depth_m, dmin, dmax):
    valid = np.isfinite(depth_m) & (depth_m > 0)
    n = np.clip((np.where(valid, depth_m, 0) - dmin) / max(dmax - dmin, 1e-6), 0, 1)
    vis = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~valid] = 0
    return vis


def label(img, text):
    for col, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, th, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- Fehlermetriken

BINS = [0, 1, 2, 3, 4, 6, 100]  # Distanzklassen [m] für das Balkendiagramm
C_GLOBAL = (80, 160, 255)        # orange (BGR)
C_LOCAL = (255, 180, 60)         # blau (BGR)


def holdout_mask(valid, rng, block=32, frac=0.15):
    """Zufällige Blöcke gemessener Pixel, die für die Auswertung ausgeblendet werden.
    Blöcke statt Einzelpixel, damit es echten Löchern ähnelt."""
    H, W = valid.shape
    gh, gw = -(-H // block), -(-W // block)
    pick = rng.random((gh, gw)) < frac
    m = np.kron(pick, np.ones((block, block), bool))[:H, :W]
    return m & valid


def metrics(pred, gt, mask):
    """AbsRel (mittl. relativer Fehler), RMSE [m], mittl. Fehler [cm], AbsRel je Distanzklasse."""
    m = mask & (pred > 0)
    if m.sum() < 50:
        return None
    p, g = pred[m], gt[m]
    e = np.abs(p - g)
    rel = e / g
    per_bin = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = (g >= lo) & (g < hi)
        per_bin.append(float(rel[sel].mean()) if sel.sum() >= 20 else np.nan)
    return {"absrel": float(rel.mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "mae_cm": 100 * float(e.mean()), "bins": per_bin}


def draw_metrics(width, height, hist, cur):
    """Links: AbsRel über die Zeit, rechts: AbsRel je Distanzklasse (global vs. Stufe 2)."""
    img = np.full((height, width, 3), 30, np.uint8)
    pad, half = 45, width // 2
    names = (("global", C_GLOBAL), ("Stufe 2", C_LOCAL))

    # --- Zeitverlauf
    x0, x1, y0, y1 = pad, half - 15, 35, height - 30
    vals = [v for k in ("global", "Stufe 2") for v in hist[k] if np.isfinite(v)]
    ymax = max(0.02, 1.15 * max(vals)) if vals else 0.1
    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 90), 1)
    for frac in (0.5, 1.0):
        yy = int(y1 - frac * (y1 - y0))
        cv2.putText(img, f"{100 * frac * ymax:.0f}%", (4, yy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (180, 180, 180), 1, cv2.LINE_AA)
    for k, col in names:
        h = list(hist[k])
        if len(h) < 2:
            continue
        xs = np.linspace(x0, x1, hist[k].maxlen)[-len(h):]
        pts = [(int(x), int(y1 - min(v, ymax) / ymax * (y1 - y0)))
               for x, v in zip(xs, h) if np.isfinite(v)]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
    cv2.putText(img, "AbsRel ueber Zeit (ausgeblendete Messpixel)", (x0, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    # --- Balken je Distanzklasse
    bx0, bx1 = half + pad, width - 15
    nb = len(BINS) - 1
    bvals = [v for k in ("global", "Stufe 2") if cur.get(k) for v in cur[k]["bins"]
             if np.isfinite(v)]
    bmax = max(0.02, 1.15 * max(bvals)) if bvals else 0.1
    cv2.rectangle(img, (bx0, y0), (bx1, y1), (90, 90, 90), 1)
    cw = (bx1 - bx0) / nb
    for i in range(nb):
        lab = f"{BINS[i]}-{BINS[i + 1]}" if BINS[i + 1] < 100 else f">{BINS[i]}"
        cv2.putText(img, lab, (int(bx0 + i * cw + cw * 0.2), height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        for j, (k, col) in enumerate(names):
            if not cur.get(k):
                continue
            v = cur[k]["bins"][i]
            if not np.isfinite(v):
                continue
            xa = int(bx0 + i * cw + cw * (0.15 + 0.35 * j))
            yb = int(y1 - min(v, bmax) / bmax * (y1 - y0))
            cv2.rectangle(img, (xa, yb), (xa + int(cw * 0.3), y1), col, -1)
    cv2.putText(img, f"AbsRel je Distanz [m]   (max {100 * bmax:.0f}%)", (bx0, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    # --- Legende mit aktuellen Zahlen
    ly = 50
    for k, col in names:
        c = cur.get(k)
        txt = (f"{k}: AbsRel {100 * c['absrel']:.1f}%  MAE {c['mae_cm']:.1f} cm  "
               f"RMSE {c['rmse']:.2f} m") if c else f"{k}: -"
        cv2.putText(img, txt, (x0 + 8, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        ly += 18
    return img


class Listener(Node):
    def __init__(self, color_topic, depth_topic, info_topic):
        super().__init__("moge_depth_viewer")
        self.lock = threading.Lock()
        self.latest = None
        self.fx = None
        subs = [Subscriber(self, Image, color_topic, qos_profile=qos_profile_sensor_data),
                Subscriber(self, Image, depth_topic, qos_profile=qos_profile_sensor_data)]
        self.sync = ApproximateTimeSynchronizer(subs, queue_size=5, slop=0.05)
        self.sync.registerCallback(self.cb)
        self.create_subscription(CameraInfo, info_topic, self.info_cb, qos_profile_sensor_data)
        self.get_logger().info(f"Warte auf {color_topic} + {depth_topic} ...")

    def info_cb(self, msg):
        self.fx = (msg.k[0], msg.width)

    def cb(self, cmsg, dmsg):
        with self.lock:
            self.latest = (cmsg, dmsg)

    def take(self):
        with self.lock:
            m, self.latest = self.latest, None
        return m


# ---------------------------------------------------------------- Hauptschleife

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", default="/camera/color/image_raw")
    ap.add_argument("--depth", default="/camera/depth/image_raw")
    ap.add_argument("--info", default="/camera/color/camera_info")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--num-tokens", type=int, default=1200,
                    help="Detailstufe MoGe: weniger = schneller (sinnvoll 1200-2500, 0 = Maximum)")
    ap.add_argument("--no-fp16", action="store_true", help="MoGe in voller Genauigkeit (langsamer)")
    ap.add_argument("--show-every", type=int, default=1,
                    help="Fenster nur jeden N-ten Frame zeichnen (hilft bei X11 übers Netz)")
    ap.add_argument("--max-range", type=float, default=0.0, help="Schätzung über X m verwerfen (0 = aus)")
    ap.add_argument("--radius", type=int, default=16,
                    help="Fenstergröße Stufe 2 in Pixeln (0 = nur globaler Faktor)")
    ap.add_argument("--eps", type=float, default=0.001,
                    help="Regularisierung Stufe 2: größer = eher nur Verschiebung")
    ap.add_argument("--local-max", type=float, default=0.0,
                    help="Stufe 2 nur für Pixel näher als X m (+1 m Übergang), 0 = überall")
    ap.add_argument("--min", type=float, default=None, help="Farbskala min [m]")
    ap.add_argument("--max", type=float, default=None, help="Farbskala max [m]")
    ap.add_argument("--width", type=int, default=640, help="Breite je Panel")
    ap.add_argument("--no-metrics", action="store_true", help="Fehlerauswertung beim Start aus")
    ap.add_argument("--holdout", type=float, default=0.15,
                    help="Anteil der Messblöcke, die für die Auswertung ausgeblendet werden")
    a = ap.parse_args()

    print("Lade MoGe-2 ...")
    moge = MoGeDepth(a.model, a.num_tokens or None, fp16=not a.no_fp16)

    rclpy.init()
    node = Listener(a.color, a.depth, a.info)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    show_mask, fps, last, log_s = True, 0.0, time.perf_counter(), None
    show_metrics = not a.no_metrics
    frame_i = 0
    t_eval = t_draw = 0.0
    rng = np.random.default_rng(0)
    hist = {"global": deque(maxlen=300), "Stufe 2": deque(maxlen=300)}
    ema_m = {"global": None, "Stufe 2": None}
    win = "RGB | Rohtiefe | gefuellt (MoGe-2)"
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

            fov_x = None
            if node.fx is not None:  # echte Brennweite -> korrekter metrischer Maßstab
                fx, w0 = node.fx
                fov_x = math.degrees(2 * math.atan(w0 / (2 * fx)))

            t = time.perf_counter()
            pred = moge(rgb, fov_x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_model = 1000 * (time.perf_counter() - t)
            filled, est, log_s = align_and_fill(depth, pred, log_s, radius_px=a.radius,
                                                eps=a.eps, max_range=a.max_range,
                                                local_max=a.local_max)
            t_all = 1000 * (time.perf_counter() - t)

            # Auswertung: Messblöcke ausblenden, füllen lassen, mit echtem Wert vergleichen
            t_e = time.perf_counter()
            if show_metrics:
                ho = holdout_mask(depth > 0, rng, frac=a.holdout)
                d_train = np.where(ho, 0, depth).astype(np.float32)
                for k, r in (("global", 0), ("Stufe 2", a.radius)):
                    f_k, _, _ = align_and_fill(d_train, pred, log_s, radius_px=r, eps=a.eps,
                                               local_max=a.local_max)
                    m_k = metrics(f_k, depth, ho)
                    if m_k:
                        ema_m[k] = m_k if ema_m[k] is None else {
                            "absrel": 0.8 * ema_m[k]["absrel"] + 0.2 * m_k["absrel"],
                            "rmse": 0.8 * ema_m[k]["rmse"] + 0.2 * m_k["rmse"],
                            "mae_cm": 0.8 * ema_m[k]["mae_cm"] + 0.2 * m_k["mae_cm"],
                            "bins": [b if not np.isfinite(o) else
                                     (o if not np.isfinite(b) else 0.8 * o + 0.2 * b)
                                     for o, b in zip(ema_m[k]["bins"], m_k["bins"])]}
                    hist[k].append(m_k["absrel"] if m_k else np.nan)
            t_eval = 1000 * (time.perf_counter() - t_e)

            frame_i += 1
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            if frame_i % max(a.show_every, 1):
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                continue
            t_d = time.perf_counter()

            v = depth[depth > 0]
            dmin = a.min if a.min is not None else (np.percentile(v, 2) if v.size else 0.0)
            dmax = a.max if a.max is not None else float(np.percentile(filled[filled > 0], 98)
                                                         if (filled > 0).any() else 5.0)

            p_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            p_raw = colorize(depth, dmin, dmax)
            p_fill = colorize(filled, dmin, dmax)
            if show_mask:
                cnts, _ = cv2.findContours(est.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(p_fill, cnts, -1, (255, 255, 255), 1)

            h = int(H * a.width / W)
            panels = [cv2.resize(p, (a.width, h)) for p in (p_rgb, p_raw, p_fill)]
            label(panels[0], "RGB" + ("" if fov_x else "  (keine camera_info!)"))
            label(panels[1], f"Rohtiefe {dmin:.1f}-{dmax:.1f} m, Loecher {(depth <= 0).mean():.0%}")
            label(panels[2], f"{fps:.1f} Hz  s={math.exp(log_s):.2f}")
            timing = (f"MoGe {t_model:.0f} | Angl. {t_all - t_model:.0f} | Auswert. {t_eval:.0f} | "
                      f"Anzeige {t_draw:.0f} ms")
            for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                cv2.putText(panels[2], timing, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, th,
                            cv2.LINE_AA)
            view = np.hstack(panels)
            if show_metrics:
                view = np.vstack([view, draw_metrics(view.shape[1], 240, hist, ema_m)])
            cv2.imshow(win, view)
            t_draw = 1000 * (time.perf_counter() - t_d)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("m"):
                show_mask = not show_mask
            if k == ord("e"):
                show_metrics = not show_metrics
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()