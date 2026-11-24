#!/usr/bin/env python3
"""
Reproduzierbarer Parameter-Sweep fuer die MoGe-2 Tiefenfuellung.

Warum ein separates Skript:
  moge_depth.py ist ein Live-Viewer. Pro Frame ist das Holdout-Muster anders,
  `log_s` wird per EMA ueber die Zeit geglaettetet und die Metriken laufen in
  einem fortlaufenden deque. Zwei Lauefe sind nie vergleichbar -> kein Tuning.

Hier wird der Prozess in drei getrennte Schritte zerlegt, damit der teure Teil
(MoGe-Inferenz) genau EINMAL pro Frame laeuft und der Sweep deterministisch ist:

  1) capture : N synchronisierte (RGB, Tiefe, camera_info) Frames in eine .npz
  2) infer   : MoGe genau einmal pro Frame rechnen, Resultat cachen
  3) sweep   : Alignment-Parameter auf dem Cache deterministisch durchprobieren

Variante-Konzepte (A/B-Tests gegen den Original-Code):
  conf_mode = base  : Original (Konfidenz doppelt gewichtet, a/b werden erst
                      gedaempft, dann nochmal conf-gewichtet gemittelt)
  conf_mode = one   : Konfidenz nur einmal als Gewicht + Praezenz-Fade
  eps_mode  = abs   : Original (eps in Metern, wirkt nur naehfeldstark)
  eps_mode  = rel   : eps * E[x^2] pro Fenster (skaleninvariant, eps zieht
                      `a` prozentual Richtung 1)

  `conf_mode=base --eps-mode abs` reproduziert moge_depth.py exakt.

Start
-----
  # 1) 60 Frames aufnehmen
  python navigation/mapping/moge_sweep.py capture \
      --color /camera/color/image_raw \
      --depth /camera/depth/image_raw \
      --info  /camera/color/camera_info \
      --frames 60 --out moge_sweep_out/frames.npz

  # 2) Inferenz + Sweep
  python navigation/mapping/moge_sweep.py all \
      --frames moge_sweep_out/frames.npz \
      --out moge_sweep_out/preds.npz \
      --out-dir moge_sweep_out --preset quick

  # nur den Sweep (Cache existiert)
  python navigation/mapping/moge_sweep.py sweep \
      --preds moge_sweep_out/preds.npz --out-dir moge_sweep_out --preset full

Ergebnisse
----------
  moge_sweep_out/sweep_rows.csv             : jede Einzel-Messung
  moge_sweep_out/sweep_summary_per_block.csv: Aggregat je (Konfig, Lochgroesse)
  moge_sweep_out/sweep_summary_overall.csv  : Aggregat je Konfig (ueber alle Loecher)

Metriken:
  AbsRel  : mittlerer relativer Fehler |est-gt|/gt
  MAE     : mittlerer absoluter Fehler [cm]
  RMSE    : [m]
  delta1  : Anteil mit max(est/gt, gt/est) < 1.25
  cov     : Coverage = Anteil Holdout-Pixel, die ueberhaupt gefuellt wurden
            (MoGe sagt hier "kein Wert" -> Loch bleibt -> Fehler, nicht "frei")
"""
import os
import sys

# Conan-Pakete (altes numpy) ausblenden, wie in moge_depth.py
sys.path = [p for p in sys.path if "conan_cache" not in p]
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import argparse
import csv
import itertools
import math
import time
from collections import defaultdict

import cv2
import numpy as np

MODEL = "Ruicheng/moge-2-vitl"
BINS = [0, 1, 2, 3, 4, 6, 100]

BASE_COLS = [
    "radius", "eps", "local_max", "band", "min_frac", "down",
    "clip_lo", "clip_hi", "conf_mode", "eps_mode", "ema", "max_range",
]

PRESETS = {
    "quick": dict(
        radius="0,24,32,48", eps="0.01", local_max="0,3",
        conf_mode="base,one", eps_mode="abs,rel",
    ),
    "full": dict(
        radius="16,24,32,48,64",
        eps="0.003,0.01,0.03",
        local_max="0,3",
        conf_mode="base,one",
        eps_mode="abs,rel",
        min_frac="0.05", down="4",
        blocks="16,32,64",
    ),
}


# ---------------------------------------------------------------- Listen-Helfer

def float_list(s):
    return [float(v) for v in str(s).split(",") if v.strip()]


def int_list(s):
    return [int(float(v)) for v in str(s).split(",") if v.strip()]


def str_list(s):
    return [v.strip() for v in str(s).split(",") if v.strip()]


# ---------------------------------------------------------------- ROS-Hilfen

def color_to_rgb(msg):
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(msg.encoding)
    if ch is None:
        raise ValueError(f"Farb-Encoding {msg.encoding} nicht unterstuetzt")
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
    raise ValueError(f"Tiefen-Encoding {msg.encoding} nicht unterstuetzt")


# ---------------------------------------------------------------- Alignment
# identisch zu moge_depth.py, aber mit conf_mode / eps_mode / Clip-Parameter


def local_affine(depth_m, est, radius_px=32, eps=0.01, min_frac=0.05, down=4,
                 clip_lo=0.5, clip_hi=2.0, conf_mode="base", eps_mode="abs"):
    """Lokale affine Angleichung: echt ~= a*est + b in jedem Fenster.

    conf_mode
      base : Original (Konfidenz doppelt angewendet)
      one  : Konfidenz einmal als Gewicht + Praezenz-Fade (empfohlen)
    eps_mode
      abs  : lam = eps [m]         (Original, nicht skaleninvariant)
      rel  : lam = eps * E[x^2]    (skaleninvariant, % Richtung 1)
    """
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
    ex2 = bf(x * x) * ni
    var = ex2 - mx * mx
    cov = bf(x * y) * ni - mx * my

    lam = eps * np.maximum(ex2, 1e-6) if eps_mode == "rel" \
        else np.full_like(var, float(eps))
    a_raw = np.clip((cov + lam) / (var + lam), clip_lo, clip_hi)
    b_raw = my - a_raw * mx
    conf = np.clip(n / min_frac, 0, 1)

    cw = bf(conf)
    cwi = 1.0 / np.maximum(cw, 1e-6)
    fade = np.clip(cw / min_frac, 0, 1)

    if conf_mode == "base":
        # Original: a/b werden ZUERST conf-gefaedat, dann nochmal conf-gewichtet
        a_pre = 1 + conf * (a_raw - 1)
        b_pre = conf * b_raw
        a_m = 1 + fade * (bf(a_pre * conf) * cwi - 1)
        b_m = fade * (bf(b_pre * conf) * cwi)
    else:  # "one"
        a_m = 1 + fade * (bf((a_raw - 1) * conf) * cwi)
        b_m = fade * (bf(b_raw * conf) * cwi)

    a_f = cv2.resize(a_m, (W, H), interpolation=cv2.INTER_LINEAR)
    b_f = cv2.resize(b_m, (W, H), interpolation=cv2.INTER_LINEAR)
    return a_f * est + b_f


def stage1_scale(depth_train, pred, pvalid, ema=0.0, prev_log_s=None):
    """Globaler log-Skalierungsfaktor (Median, 1/4-Pixel). Deterministisch."""
    both = (depth_train > 0) & pvalid
    sub = both[::4, ::4]
    log_s = prev_log_s if prev_log_s is not None else 0.0
    if int(sub.sum()) > 200:
        ratio = depth_train[::4, ::4][sub] / pred[::4, ::4][sub]
        new = float(np.log(np.median(ratio)))
        log_s = new if prev_log_s is None else (1.0 - ema) * prev_log_s + ema * new
    est = pred.astype(np.float32) * np.float32(math.exp(log_s))
    est[~pvalid] = 0.0
    return est, log_s


def fill_frame(depth_train, est, pvalid, cfg):
    """Stage 2 + Combine + max_range. Rechnet nur noch den teuren Teil."""
    if cfg["radius"] > 0:
        loc = local_affine(depth_train, est,
                           radius_px=cfg["radius"], eps=cfg["eps"],
                           min_frac=cfg["min_frac"], down=cfg["down"],
                           clip_lo=cfg["clip_lo"], clip_hi=cfg["clip_hi"],
                           conf_mode=cfg["conf_mode"], eps_mode=cfg["eps_mode"])
        if cfg["local_max"] > 0:
            band = max(cfg["band"], 1e-6)
            wgt = np.clip((cfg["local_max"] + band - est) / band, 0.0, 1.0)
            loc = wgt * loc + (1.0 - wgt) * est
        est = np.where(pvalid, loc, 0.0)
    filled = np.where(depth_train > 0, depth_train, est)
    if cfg["max_range"] > 0:
        drop = (depth_train <= 0) & (filled > cfg["max_range"])
        if drop.any():
            filled = filled.copy()
            filled[drop] = 0.0
    return filled


# ---------------------------------------------------------------- Metrik

def holdout_mask(valid, rng, block=32, frac=0.15):
    H, W = valid.shape
    gh, gw = -(-H // block), -(-W // block)
    pick = rng.random((gh, gw)) < frac
    m = np.kron(pick, np.ones((block, block), bool))[:H, :W]
    return m & valid


def eval_sample(filled, gt, mask):
    tot = int(mask.sum())
    if tot == 0:
        return None
    sel = mask & (filled > 0)
    cov = float(sel.sum()) / float(tot)
    out = {"absrel": float("nan"), "mae_cm": float("nan"), "rmse": float("nan"),
           "delta1": float("nan"), "coverage": cov,
           "bins": [float("nan")] * (len(BINS) - 1)}
    if int(sel.sum()) < 50:
        return out
    p = filled[sel]
    g = gt[sel]
    e = np.abs(p - g)
    rel = e / g
    out["absrel"] = float(rel.mean())
    out["mae_cm"] = 100.0 * float(e.mean())
    out["rmse"] = float(np.sqrt(np.mean(e * e)))
    out["delta1"] = float(np.mean(np.maximum(p / g, g / p) < 1.25))
    for j, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
        s = (g >= lo) & (g < hi)
        if int(s.sum()) >= 20:
            out["bins"][j] = float(rel[s].mean())
    return out


# ---------------------------------------------------------------- Konfiguration

def _canonical(cfg):
    c = dict(cfg)
    if c["radius"] <= 0:
        # lokale Parameter irrelevant -> normalisieren fuer Dedupe
        c.update(eps=0.0, min_frac=0.0, down=0, clip_lo=0.0, clip_hi=0.0,
                 conf_mode="base", eps_mode="abs")
    if c["local_max"] <= 0:
        c["band"] = 0.0
    return c


def build_configs(a):
    grids = {
        "radius": int_list(a.radius),
        "eps": float_list(a.eps),
        "local_max": float_list(a.local_max),
        "band": float_list(a.band),
        "min_frac": float_list(a.min_frac),
        "down": int_list(a.down),
        "clip_lo": float_list(a.clip_lo),
        "clip_hi": float_list(a.clip_hi),
        "conf_mode": str_list(a.conf_mode),
        "eps_mode": str_list(a.eps_mode),
        "ema": float_list(a.ema),
        "max_range": float_list(a.max_range),
    }
    seen = {}
    for vals in itertools.product(*(grids[k] for k in BASE_COLS)):
        cfg = _canonical(dict(zip(BASE_COLS, vals)))
        seen[tuple(cfg[k] for k in BASE_COLS)] = cfg
    cfgs = list(seen.values())
    cfgs.sort(key=lambda c: (c["radius"], c["conf_mode"], c["eps_mode"],
                             c["eps"], c["local_max"], c["min_frac"],
                             c["down"], c["clip_lo"], c["clip_hi"],
                             c["ema"], c["max_range"]))
    return cfgs


def cfg_str(r):
    return (f"r={r['radius']:g} eps={r['eps']:g} lmax={r['local_max']:g} "
            f"band={r['band']:g} minfrac={r['min_frac']:g} down={r['down']:g} "
            f"clip={r['clip_lo']:g}-{r['clip_hi']:g} conf={r['conf_mode']} "
            f"epsm={r['eps_mode']} ema={r['ema']:g} xmax={r['max_range']:g}")


# ---------------------------------------------------------------- Aggregation

def aggregate(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, rs in groups.items():
        d = dict(zip(keys, key))
        for m in ("absrel", "mae_cm", "rmse", "delta1", "coverage", "ms"):
            vals = [r[m] for r in rs if r.get(m) is not None and np.isfinite(r[m])]
            d[m] = float(np.mean(vals)) if vals else float("nan")
        bins = []
        for j in range(len(BINS) - 1):
            vals = [r["bins"][j] for r in rs if np.isfinite(r["bins"][j])]
            bins.append(float(np.mean(vals)) if vals else float("nan"))
        d["bins"] = bins
        d["n"] = len(rs)
        out.append(d)
    return out


def _fmt_bins(b):
    return ";".join("" if (b is None or not np.isfinite(v)) else f"{v:.5g}" for v in b)


def _write_agg(path, agg_rows):
    if not agg_rows:
        return
    has_block = "block" in agg_rows[0]
    fields = (["block"] if has_block else []) + BASE_COLS + \
             ["absrel", "mae_cm", "rmse", "delta1", "coverage", "ms", "n", "bins"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in agg_rows:
            rr = dict(r)
            rr["bins"] = _fmt_bins(r["bins"])
            w.writerow(rr)


def print_summary(per_block, overall, blocks, min_coverage, top):
    print("\n" + "=" * 82)
    print("Ergebnis je Holdout-Lochgroesse  (AbsRel = mittlerer relativer Fehler)")
    print("=" * 82)
    for block in blocks:
        rows_b = [r for r in per_block if r["block"] == block]
        base = sorted([r for r in rows_b if r["radius"] == 0], key=lambda r: r["absrel"])
        if base and np.isfinite(base[0]["absrel"]):
            b = base[0]
            print(f"\nLoch {block:>3}px   global (r=0): "
                  f"AbsRel {100 * b['absrel']:5.2f}%  MAE {b['mae_cm']:6.1f}cm  "
                  f"RMSE {b['rmse']:.2f}m  d1 {100 * b['delta1']:5.1f}%  "
                  f"cov {100 * b['coverage']:5.1f}%")
        cand = [r for r in rows_b if r["radius"] > 0 and r["coverage"] >= min_coverage
                and np.isfinite(r["absrel"])]
        cand.sort(key=lambda r: r["absrel"])
        for j, r in enumerate(cand[:top]):
            print(f"  #{j + 1} {cfg_str(r)}")
            print(f"        -> AbsRel {100 * r['absrel']:5.2f}%  MAE {r['mae_cm']:6.1f}cm  "
                  f"RMSE {r['rmse']:.2f}m  d1 {100 * r['delta1']:5.1f}%  "
                  f"cov {100 * r['coverage']:5.1f}%  {r['ms']:5.1f}ms")

    print("\n" + "=" * 82)
    print(f"Bester Kompromiss ueber alle Lochgroessen  (Coverage >= {100 * min_coverage:.0f}%)")
    print("=" * 82)
    cand = [r for r in overall if r["radius"] > 0 and r["coverage"] >= min_coverage
            and np.isfinite(r["absrel"])]
    cand.sort(key=lambda r: r["absrel"])
    if not cand:
        print("keine Konfiguration erfuellt die Coverage-Schwelle")
    for j, r in enumerate(cand[:max(top, 10)]):
        print(f"{j + 1:2d}. AbsRel {100 * r['absrel']:5.2f}%  MAE {r['mae_cm']:6.1f}cm  "
              f"d1 {100 * r['delta1']:5.1f}%  cov {100 * r['coverage']:5.1f}%  "
              f"{r['ms']:5.1f}ms | {cfg_str(r)}")
    # Variante-Effekt (base vs one, abs vs rel)
    print("\n" + "=" * 82)
    print("Variante-Effekt  (gemittelt ueber alle Loecher, radius > 0)")
    print("=" * 82)
    for cmode in ("base", "one"):
        for emode in ("abs", "rel"):
            rows_v = [r for r in overall if r["radius"] > 0
                      and r["conf_mode"] == cmode and r["eps_mode"] == emode
                      and np.isfinite(r["absrel"])]
            if rows_v:
                m_abs = float(np.mean([r["absrel"] for r in rows_v]))
                m_cov = float(np.mean([r["coverage"] for r in rows_v]))
                m_ms = float(np.mean([r["ms"] for r in rows_v]))
                print(f"  conf={cmode:<4}  epsm={emode:<3}  "
                      f"AbsRel {100 * m_abs:5.2f}%  cov {100 * m_cov:5.1f}%  "
                      f"{m_ms:5.1f}ms  (n={len(rows_v)})")


# ---------------------------------------------------------------- capture

def cmd_capture(a):
    import rclpy
    from message_filters import ApproximateTimeSynchronizer, Subscriber
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    class Capture(Node):
        def __init__(self):
            super().__init__("moge_sweep_capture")
            self.rgb, self.dep = [], []
            self.fx, self.w0 = None, None
            subs = [Subscriber(self, Image, a.color, qos_profile=qos_profile_sensor_data),
                    Subscriber(self, Image, a.depth, qos_profile=qos_profile_sensor_data)]
            self.sync = ApproximateTimeSynchronizer(subs, queue_size=5, slop=a.slop)
            self.sync.registerCallback(self.cb)
            self.create_subscription(CameraInfo, a.info, self.info_cb,
                                     qos_profile=qos_profile_sensor_data)
            self.get_logger().info(f"capture: {a.color} + {a.depth} -> {a.frames} Frames")

        def info_cb(self, msg):
            if self.fx is None and msg.k[0] > 0:
                self.fx, self.w0 = float(msg.k[0]), int(msg.width)

        def cb(self, cmsg, dmsg):
            if len(self.rgb) >= a.frames:
                return
            try:
                rgb = color_to_rgb(cmsg)
                dep = depth_to_m(dmsg)
            except ValueError as e:
                self.get_logger().warn(str(e))
                return
            if dep.shape != rgb.shape[:2]:
                dep = cv2.resize(dep, (rgb.shape[1], rgb.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            self.rgb.append(rgb)
            self.dep.append(dep)

    rclpy.init()
    node = Capture()
    t0 = time.perf_counter()
    try:
        while rclpy.ok() and len(node.rgb) < a.frames and time.perf_counter() - t0 < a.timeout:
            rclpy.spin_once(node, timeout_sec=0.05)
        n = len(node.rgb)
        if n < a.min_frames:
            raise SystemExit(f"nur {n} von {a.frames} Frames in {a.timeout:.0f}s "
                             f"-> sind die Topics live?")
        if node.fx is None:
            print("[warn] keine camera_info bekommen -> fx=NaN, MoGe schatzt FOV")
        fov_x = (math.degrees(2 * math.atan(node.w0 / (2 * node.fx)))
                 if node.fx else float("nan"))
        dmin = float(np.min(np.concatenate([d[d > 0] for d in node.dep if (d > 0).any()]))) \
            if any((d > 0).any() for d in node.dep) else float("nan")
        dmax = float(np.max(np.concatenate([d[d > 0] for d in node.dep if (d > 0).any()]))) \
            if any((d > 0).any() for d in node.dep) else float("nan")
        hole_frac = float(np.mean([float((d <= 0).mean()) for d in node.dep]))
        print(f"capture: {n} Frames, HxW={node.rgb[0].shape[0]}x{node.rgb[0].shape[1]}, "
              f"Tiefe {dmin:.2f}-{dmax:.2f}m, Loecher {hole_frac:.0%}")
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        np.savez(a.out,
                 rgb=np.stack(node.rgb),
                 depth=np.stack(node.dep),
                 fov_x=np.full(n, fov_x, dtype=np.float64))
        print(f"-> {a.out}")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


# ---------------------------------------------------------------- infer

class MoGeRunner:
    """Kleine Kapselung um MoGe-2, ohne dass rclpy importiert werden muss."""

    def __init__(self, model_id=MODEL, num_tokens=None, fp16=True):
        import torch
        from moge.model.v2 import MoGeModel
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MoGeModel.from_pretrained(model_id).to(self.device).eval()
        self.num_tokens = num_tokens
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        self._use_fp16_arg = True

    def __call__(self, rgb, fov_x=None):
        torch = self.torch
        with torch.inference_mode():
            img = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float() / 255.0
            kw = dict(fov_x=fov_x, num_tokens=self.num_tokens)
            out = None
            if self._use_fp16_arg:
                try:
                    out = self.model.infer(img, use_fp16=self.fp16, **kw)
                except TypeError:
                    self._use_fp16_arg = False
            if out is None:
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                    out = self.model.infer(img, **kw)
            d = out["depth"].float().cpu().numpy()
            if "mask" in out:
                d[~out["mask"].cpu().numpy().astype(bool)] = np.inf
        return d


def cmd_infer(a):
    print(f"lade {a.model} ...")
    runner = MoGeRunner(a.model, a.num_tokens or None, fp16=not a.no_fp16)
    print(f"Gerät: {runner.device} | fp16={runner.fp16} | num_tokens={a.num_tokens or 'max'}")

    data = np.load(a.frames)
    if "rgb" not in data or "depth" not in data:
        raise SystemExit(f"{a.frames}: braucht 'rgb' und 'depth'")
    rgb = data["rgb"]
    depth = data["depth"].astype(np.float32)
    fov = data["fov_x"].astype(np.float64) if "fov_x" in data else np.full(len(rgb), np.nan)
    n = len(rgb)
    if a.limit:
        n = min(n, int(a.limit))

    preds = []
    t0 = time.perf_counter()
    for i in range(n):
        fx = float(fov[i]) if np.isfinite(fov[i]) else None
        p = runner(rgb[i], fx)
        if p.shape != depth[i].shape:
            H, W = depth[i].shape
            p = cv2.resize(p, (W, H), interpolation=cv2.INTER_NEAREST)
        preds.append(p)
        if (i + 1) % 10 == 0 or i + 1 == n:
            dt = time.perf_counter() - t0
            print(f"  {i + 1:4d}/{n}  {1000 * dt / (i + 1):6.1f} ms/Frame")
    pred = np.stack(preds)
    if a.pred_dtype == "float16":
        pred = pred.astype(np.float16)
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(a.out, pred=pred, depth=depth[:n], fov_x=fov[:n])
    print(f"-> {a.out}  ({os.path.getsize(a.out) / 1e6:.0f} MB)")


# ---------------------------------------------------------------- sweep

def cmd_sweep(a):
    if getattr(a, "preset", "none") != "none":
        for k, v in PRESETS[a.preset].items():
            setattr(a, k, v)
    preds_path = getattr(a, "preds", None)
    if not preds_path:
        raise SystemExit("sweep: --preds fehlt (zuerst 'infer' oder 'all' laufen lassen)")
    data = np.load(preds_path)
    if "pred" not in data or "depth" not in data:
        raise SystemExit(f"{preds_path}: braucht 'pred' und 'depth'")
    pred_all = data["pred"].astype(np.float32)
    depth_all = data["depth"].astype(np.float32)
    n = len(pred_all)
    if getattr(a, "limit", 0):
        n = min(n, int(a.limit))
    if n < 1:
        raise SystemExit("keine Frames")
    pred_all, depth_all = pred_all[:n], depth_all[:n]
    if pred_all.shape[1:] != depth_all.shape[1:]:
        H, W = depth_all.shape[1:]
        pred_all = np.stack([cv2.resize(p, (W, H), interpolation=cv2.INTER_NEAREST)
                             for p in pred_all])
        print(f"[warn] pred auf {W}x{H} skaliert")

    cfgs = build_configs(a)
    blocks = int_list(a.blocks)
    reps = max(int(a.reps), 1)
    total = n * len(blocks) * reps * len(cfgs)
    print(f"Frames: {n} | Lochgroessen: {blocks} | reps: {reps} | Konfigs: {len(cfgs)}")
    print(f"Messungen gesamt: {total:,}".replace(",", "."))
    if len(cfgs) > a.max_combos and not a.force:
        raise SystemExit(f"{len(cfgs)} Konfigs > --max-combos {a.max_combos} "
                         f"(--force zur Uebersteuerung)")
    if total > 400_000:
        print("[warn] > 400k Messungen, wird etwas dauern")

    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    t_start = time.perf_counter()
    done = 0
    step = max(1, total // 25)

    for bi, block in enumerate(blocks):
        for rep in range(reps):
            samples = []
            for i in range(n):
                depth = depth_all[i]
                valid = depth > 0
                rng = np.random.default_rng((int(a.seed), bi, rep, i))
                ho = holdout_mask(valid, rng, block=block, frac=a.holdout)
                d_train = np.where(ho, 0.0, depth).astype(np.float32)
                samples.append((d_train, ho))

            # Stage 1 je ema-Wert (deterministisch, sequentiell ueber Frames)
            est_cache = {}
            for ema in sorted({c["ema"] for c in cfgs}):
                prev = None
                arr = []
                for i in range(n):
                    d_train = samples[i][0]
                    pred = pred_all[i]
                    pvalid = np.isfinite(pred) & (pred > 0)
                    est, prev = stage1_scale(d_train, pred, pvalid, ema, prev)
                    arr.append((est, pvalid))
                est_cache[ema] = arr

            for cfg in cfgs:
                for i in range(n):
                    d_train, ho = samples[i]
                    est, pvalid = est_cache[cfg["ema"]][i]
                    t = time.perf_counter()
                    filled = fill_frame(d_train, est, pvalid, cfg)
                    ms = 1000.0 * (time.perf_counter() - t)
                    m = eval_sample(filled, depth_all[i], ho)
                    if m is None:
                        continue
                    row = dict(cfg)
                    row.update(block=block, rep=rep, frame=i, ms=ms, **m)
                    rows.append(row)
                    done += 1
                if (i + 1) == n and (done % step) < len(cfgs):
                    el = time.perf_counter() - t_start
                    eta = el * (total - done) / max(done, 1)
                    print(f"  {100.0 * done / total:5.1f}%  {el:7.1f}s  ETA {eta:6.1f}s")

    if not rows:
        raise SystemExit("keine Messung erzeugt (zu wenig valide Tiefe?)")

    # --- CSVs
    raw_path = os.path.join(a.out_dir, "sweep_rows.csv")
    FIELDS = BASE_COLS + ["block", "rep", "frame",
                          "absrel", "mae_cm", "rmse", "delta1", "coverage", "ms", "bins"]
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            rr = {k: r.get(k) for k in FIELDS}
            rr["bins"] = _fmt_bins(r["bins"])
            w.writerow(rr)

    per_block = aggregate(rows, ["block"] + BASE_COLS)
    overall = aggregate(rows, BASE_COLS)
    _write_agg(os.path.join(a.out_dir, "sweep_summary_per_block.csv"), per_block)
    _write_agg(os.path.join(a.out_dir, "sweep_summary_overall.csv"), overall)
    print(f"\nCSVs in: {a.out_dir}")

    print_summary(per_block, overall, blocks, a.min_coverage, a.top)

    el = time.perf_counter() - t_start
    print(f"\nfertig in {el:.0f}s  ->  CSVs + oben: {a.out_dir}")


def cmd_all(a):
    cmd_infer(a)
    a.preds = a.out
    cmd_sweep(a)


# ---------------------------------------------------------------- CLI

def add_infer_args(p):
    p.add_argument("--frames", required=True, help="npz aus 'capture'")
    p.add_argument("--out", required=True, help="Ziel .npz fuer pred+depth")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--num-tokens", type=int, default=1200,
                   help="MoGe-Details (0 = Maximum, 1200 Standard)")
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="nur erste N Frames (0 = alle)")
    p.add_argument("--pred-dtype", choices=["float16", "float32"], default="float16")


def add_sweep_args(p, with_limit=True):
    p.add_argument("--preds", default=None, help="npz aus 'infer'")
    p.add_argument("--out-dir", default="moge_sweep_out")
    p.add_argument("--preset", choices=["none", "quick", "full"], default="none",
                   help="vorgefertigte Parametergitter (sonst unten)")
    # Parametergitter (Kommata)
    p.add_argument("--radius", default="0,32",
                   help="Fenster Stufe 2 [px], 0 = nur global")
    p.add_argument("--eps", default="0.01")
    p.add_argument("--local-max", default="0", help="Stufe 2 nur bis X m (0 = aus)")
    p.add_argument("--band", default="1.0", help="Uebergangsbreite zu global [m]")
    p.add_argument("--min-frac", default="0.05")
    p.add_argument("--down", default="4", help="Downsampling Faktor fuer Fenster")
    p.add_argument("--clip-lo", default="0.5")
    p.add_argument("--clip-hi", default="2.0")
    p.add_argument("--conf-mode", default="base", help="base (orig) | one (fix)")
    p.add_argument("--eps-mode", default="abs", help="abs (orig) | rel (fix)")
    p.add_argument("--ema", default="0", help="EMA fuer log_s (0 = aus, deterministisch)")
    p.add_argument("--max-range", default="0", help="Schaeztung > X m verwerfen")
    # Auswertung
    p.add_argument("--blocks", default="16,32,64",
                   help="Holdout-Lochgroessen [px], Komma-Liste")
    p.add_argument("--reps", type=int, default=1,
                   help="unabhaengige Holdout-Proben je Frame/Lochgroesse")
    p.add_argument("--holdout", type=float, default=0.15,
                   help="Anteil Blöcke, die ausgeblendet werden")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-coverage", type=float, default=0.95)
    p.add_argument("--top", type=int, default=5)
    if with_limit:
        p.add_argument("--limit", type=int, default=0, help="nur erste N Frames")
    p.add_argument("--max-combos", type=int, default=5000)
    p.add_argument("--force", action="store_true")


def main():
    ap = argparse.ArgumentParser(
        description="Reproduzierbarer Parameter-Sweep fuer moge_depth.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("capture", help="Frames von ROS-Topics aufnehmen")
    p.add_argument("--color", default="/camera/color/image_raw")
    p.add_argument("--depth", default="/camera/depth/image_raw")
    p.add_argument("--info", default="/camera/color/camera_info")
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--min-frames", type=int, default=10)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--slop", type=float, default=0.05)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("infer", help="MoGe einmal pro Frame rechnen und cachen")
    add_infer_args(p)
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("sweep", help="Parameter auf dem Cache durchprobieren")
    add_sweep_args(p, with_limit=True)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("all", help="infer + sweep in einem Lauf")
    add_infer_args(p)
    add_sweep_args(p, with_limit=False)
    p.set_defaults(func=cmd_all)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
