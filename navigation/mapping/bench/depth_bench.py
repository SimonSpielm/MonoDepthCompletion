#!/usr/bin/env python3
"""
Offline benchmark for the depth completion (no ROS, runs on a laptop).

Runs EXACTLY the production algorithm from navigation/mapping/depth_fusion.py
(global scale, gradient fusion, edge filters) and depth_models.py on recorded or
synthetic frames and reports accuracy AND edge bleeding, so parameters and code
changes can be compared deterministically.

Data sources
------------
  --frames F.npz    recorded frames (moge_sweep.py capture): rgb (N,H,W,3) uint8,
                    depth (N,H,W) float [m], fov_x (N,)
  --preds  P.npz    cached predictions for these frames: pred (N,H,W) [m]
                    (moge_sweep.py infer, or --predict below)
  --predict MODEL   run the monocular model locally (CUDA / Apple MPS / CPU) and
                    cache its predictions, any name of depth_models.MODELS:
                    moge, moge2-s, moge2-b, moge3-l, depthpro2-s / -b / -l / -h
  --synthetic N     N synthetic scenes (floor, wall, boxes, thin poles) with a
                    ToF-like sensor (holes at edges, dark blobs, range drop-out,
                    flying pixels) and a MoGe-like prediction (soft edge ramps,
                    smooth scale error). Needs neither data nor MoGe.

Metrics (each frame is evaluated with a fresh fusion -> deterministic)
------------------------------------------------------------------------
  recorded frames, ground truth = sensor measurements that are hidden on purpose:
    hold_*  : random 32 px blocks (--hold-px) of measured pixels hidden (protocol of the sweeps)
    edge_*  : a 6 px band around measured depth edges hidden (what the ToF really
              loses at edges); evaluated on that band
  synthetic frames, ground truth = known scene depth:
    hold_*  : all real sensor holes away from edges
    edge_*  : all real sensor holes within 6 px of a depth edge

    absrel  : mean |est - gt| / gt over filled pixels
    cov     : fraction of the evaluated pixels that got a value at all
    bleed   : fraction of the evaluated pixels filled with > 10 % error
              (= floating points / "comet tails" in the map). Lower is better;
              read it together with cov (filling nothing gives bleed 0, cov 0).

Examples
--------
  python depth_bench.py --synthetic 8
  python depth_bench.py --synthetic 8 --sweep ramp_dmin=0,0.5 support_px=0,12
  python depth_bench.py --frames frames.npz --preds preds.npz --set down=8
  python depth_bench.py --frames a.npz b.npz --predict moge2-b --num-tokens 600 --hold-px 128
"""
import argparse
import csv
import itertools
import math
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from depth_fusion import GradientFusion, fill  # noqa: E402
from depth_models import MOGE, MODELS, load_model  # noqa: E402


FUSION_KEYS = ("down", "lam", "mu_near", "mu_far", "far_band", "sigma_edge", "z_ref",
               "outlier", "iters", "tol")
FILL_KEYS = ("ema", "ramp_dmin", "support_px", "support_rtol", "support_edge")
DEFAULTS = {}            # sonst gelten die Standardwerte aus depth_fusion.py
NEAR_M = 3.0           # Nahbereich für die near-Metriken
BLEED_REL = 0.10        # Fehler, ab dem ein gefülltes Pixel als Ausreißer zählt
EDGE_BAND_PX = 6
MEAS_EDGE_RTOL = 0.10   # Kante in der Messung: Spanne im 3x3 > 10 %


# ---------------------------------------------------------------- Hilfen

def edge_mask_np(d, rtol):
    """Pixel, deren Tiefenspanne im 3x3 > rtol·Tiefe ist (Löcher ignoriert)."""
    v = d > 0
    k = np.ones((3, 3), np.uint8)
    dmax = cv2.dilate(np.where(v, d, 0).astype(np.float32), k)
    dmin = cv2.erode(np.where(v, d, np.inf).astype(np.float32), k)
    return v & ((dmax - dmin) > rtol * d)


def dilate(mask, px):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def score(filled, gt, mask):
    n = int(mask.sum())
    if n == 0:
        return dict(absrel=np.nan, cov=np.nan, bleed=np.nan, cm=np.nan)
    f, g = filled[mask], gt[mask]
    hit = f > 0
    err = np.abs(f[hit] - g[hit])
    rel = err / g[hit]
    return dict(absrel=float(rel.mean()) if hit.any() else np.nan,
                cov=float(hit.mean()),
                bleed=float((rel > BLEED_REL).sum() / n),
                cm=float(100 * err.mean()) if hit.any() else np.nan)


def score2(filled, gt, mask, prefix):
    """Gesamt und Nahbereich (gt < NEAR_M)."""
    out = {f"{prefix}_{k}": v for k, v in score(filled, gt, mask).items()}
    out.update({f"{prefix}_near_{k}": v for k, v in score(filled, gt, mask & (gt < NEAR_M)).items()})
    return out


# ---------------------------------------------------------------- synthetische Szene

def _smooth_noise(rng, H, W, cells=(6, 10)):
    return cv2.resize(rng.standard_normal(cells).astype(np.float32), (W, H),
                      interpolation=cv2.INTER_CUBIC)


def synth_frame(rng, H=720, W=1280, fx=610.0, cam_h=0.58):
    """(gt, sensor, pred) in Metern. Kamera 0.58 m hoch, leicht nach unten geneigt."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    x, y = (u - W / 2) / fx, (v - H / 2) / fx                  # Strahl mit z = 1
    p = math.radians(rng.uniform(8, 18))
    g = np.cos(p) * y + np.sin(p)                              # Boden: g·X = cam_h
    depth = np.where(g > 1e-3, cam_h / np.maximum(g, 1e-3), np.inf)
    a = math.radians(rng.uniform(-20, 20))                     # schräge Rückwand
    zw = rng.uniform(3.0, 5.5)
    nr = np.sin(a) * x + np.cos(a)
    depth = np.minimum(depth, np.where(nr > 1e-3, zw / nr, np.inf))
    for _ in range(rng.integers(3, 6)):                        # Kisten (Frontflächen)
        z = rng.uniform(0.7, 2.6)
        w, h = rng.uniform(0.25, 0.7), rng.uniform(0.25, 0.9)
        cx, top = rng.uniform(-1.0, 1.0) * z * 0.6, rng.uniform(-0.4, 0.2)
        inside = (np.abs(x * z - cx) < w / 2) & (y * z > top) & (y * z < top + h)
        depth = np.where(inside, np.minimum(depth, z), depth)
    for _ in range(rng.integers(1, 3)):                        # dünne Stangen
        z, cx = rng.uniform(0.8, 2.0), rng.uniform(-0.8, 0.8)
        inside = np.abs(x * z - cx) < rng.uniform(0.01, 0.03)
        depth = np.where(inside, np.minimum(depth, z), depth)
    gt = np.where(np.isfinite(depth), depth, 0).astype(np.float32)

    # Sensor (ToF-artig)
    sensor = gt * (1 + 0.002 * gt * rng.standard_normal(gt.shape).astype(np.float32))
    edges = edge_mask_np(gt, 0.05)
    holes = dilate(edges, 2) & (rng.random(gt.shape) < 0.85)   # Kanten-Löcher
    fly = edges & ~holes                                        # verbleibende Mischpixel
    sensor[fly] = cv2.blur(gt, (5, 5))[fly]
    for _ in range(rng.integers(2, 6)):                         # dunkle Flecken
        c = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        ax = (int(rng.integers(20, 90)), int(rng.integers(15, 60)))
        blob = np.zeros(gt.shape, np.uint8)
        cv2.ellipse(blob, c, ax, float(rng.uniform(0, 180)), 0, 360, 1, -1)
        holes |= blob > 0
    holes |= (gt > 4.5) & (rng.random(gt.shape) < np.clip((gt - 4.5) / 1.5, 0, 1))
    holes[:, :25] = holes[:, -25:] = True                      # FoV-Rand
    sensor[holes | (gt <= 0)] = 0

    # MoGe-artige Vorhersage: weiche Kanten, glatter Skalenfehler
    lg = cv2.GaussianBlur(np.log(np.where(gt > 0, gt, 1)), (0, 0), 2.5)
    pred = np.exp(lg + 0.06 * _smooth_noise(rng, H, W)) * rng.uniform(0.9, 1.1)
    pred *= 1 + 0.005 * rng.standard_normal(gt.shape)
    pred = np.where(gt > 0, pred, np.inf).astype(np.float32)
    return gt, sensor.astype(np.float32), pred


# ---------------------------------------------------------------- Daten laden

def load_recorded(a, frames_path):
    data = np.load(frames_path)
    depth = data["depth"].astype(np.float32)
    n = min(len(depth), a.limit) if a.limit else len(depth)
    fov = data["fov_x"].astype(np.float64) if "fov_x" in data else np.full(len(depth), np.nan)
    tag = f"{a.predict}{a.num_tokens}" if a.predict in MOGE else a.predict
    preds_path = a.preds or (os.path.splitext(frames_path)[0] + f"_preds_{tag}.npz"
                             if a.predict else None)
    if preds_path and os.path.exists(preds_path):
        pred = np.load(preds_path)["pred"][:n].astype(np.float32)
    elif a.predict:
        model = load_model(a.predict, a.num_tokens, a.model_device)
        print(f"{a.predict} auf {model.dev} ...")
        preds = []
        for i in range(n):
            fx = float(fov[i]) if np.isfinite(fov[i]) else None
            preds.append(model(data["rgb"][i], fx).float().cpu().numpy())
            print(f"\r  {i + 1}/{n}", end="", flush=True)
        pred = np.stack(preds)
        np.savez(preds_path, pred=pred, fov_x=fov[:n])
        print(f"\n-> Cache {preds_path}")
    else:
        raise SystemExit(f"--preds fehlt (oder --predict {'|'.join(MODELS)}, um lokal zu rechnen)")
    f_px = [0.5 * depth.shape[2] / math.tan(math.radians(fov[i]) / 2) if np.isfinite(fov[i]) else None
            for i in range(n)]
    return [(None, depth[i], pred[i], f_px[i]) for i in range(n)]


# ---------------------------------------------------------------- Auswertung

def run_fill(depth, pred, cfg, dev, f_px=None):
    fusion = GradientFusion(**{k: cfg[k] for k in FUSION_KEYS if k in cfg})
    kw = {k: cfg[k] for k in FILL_KEYS if k in cfg}
    kw["f_px"] = f_px
    if "support_px" in kw:
        kw["support_px"] = int(kw["support_px"])
    t0 = time.perf_counter()
    out, _ = fill(torch.from_numpy(depth).to(dev), torch.from_numpy(pred).to(dev), fusion, **kw)
    out = out.cpu().numpy()
    return out, 1000 * (time.perf_counter() - t0)


def evaluate(frames, cfg, dev, seed=0, hold_px=32):
    rows, ms = [], []
    for i, (gt, depth, pred, f_px) in enumerate(frames):
        rng = np.random.default_rng(seed + i)
        if gt is not None:                                   # synthetisch: echte Löcher
            filled, t = run_fill(depth, pred, cfg, dev, f_px)
            holes = (depth <= 0) & (gt > 0)
            near = dilate(edge_mask_np(gt, 0.05), EDGE_BAND_PX)
            hold, edge = score2(filled, gt, holes & ~near, "hold"), score2(filled, gt, holes & near, "edge")
            ms.append(t)
        else:                                                # Aufnahme: Messungen ausblenden
            valid = depth > 0
            H, W = depth.shape
            pick = rng.random((-(-H // hold_px), -(-W // hold_px))) < 0.15
            blocks = np.kron(pick, np.ones((hold_px, hold_px), bool))[:H, :W] & valid
            f1, t = run_fill(np.where(blocks, 0, depth).astype(np.float32), pred, cfg, dev, f_px)
            hold = score2(f1, depth, blocks, "hold")
            e = edge_mask_np(depth, MEAS_EDGE_RTOL)
            band = dilate(e, EDGE_BAND_PX) & valid & ~e        # Kantenpixel selbst: unsicher
            f2, _ = run_fill(np.where(band, 0, depth).astype(np.float32), pred, cfg, dev, f_px)
            edge = score2(f2, depth, band, "edge")
            ms.append(t)
        rows.append({**hold, **edge})
    res = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    res["ms"] = float(np.median(ms))
    return res


def parse_kv(items, multi):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        if k not in FUSION_KEYS + FILL_KEYS:
            raise SystemExit(f"unbekannter Parameter '{k}', erlaubt: {FUSION_KEYS + FILL_KEYS}")
        vals = [float(x) for x in v.split(",")]
        vals = [int(x) if k in ("down", "iters") else x for x in vals]
        out[k] = vals if multi else vals[0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", nargs="+", help="eine oder mehrere frames.npz (Mittel über alle)")
    src.add_argument("--synthetic", type=int)
    ap.add_argument("--preds")
    ap.add_argument("--predict", choices=MODELS,
                    help="Modell lokal rechnen (und cachen)")
    ap.add_argument("--num-tokens", type=int, default=1200, help="nur MoGe")
    ap.add_argument("--model-device", default=None, help="cuda / mps / cpu (Standard: automatisch)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None,
                    help="Gerät für die Fusion (cpu / cuda / mps), Standard: cuda, sonst mps, sonst cpu")
    ap.add_argument("--threads", type=int, default=4, help="CPU-Threads (begrenzt, damit der Rechner bedienbar bleibt)")
    ap.add_argument("--set", nargs="*", help="feste Parameter, z. B. down=4 ramp_dmin=0.4")
    ap.add_argument("--sweep", nargs="*", help="Raster, z. B. ramp_dmin=0,0.5 support_px=0,12")
    ap.add_argument("--csv", help="Ergebnisse zusätzlich als CSV")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold-px", type=int, default=32, help="Kantenlänge der ausgeblendeten Blöcke")
    a = ap.parse_args()

    torch.set_num_threads(max(1, a.threads))
    a.device = a.device or ("cuda" if torch.cuda.is_available() else
                            "mps" if torch.backends.mps.is_available() else "cpu")
    if a.synthetic:
        rng = np.random.default_rng(a.seed)
        frames = [(*synth_frame(rng), 610.0) for _ in range(a.synthetic)]
        print(f"{len(frames)} synthetische Frames")
    else:
        if a.preds and len(a.frames) > 1:
            raise SystemExit("--preds geht nur mit einer --frames-Datei")
        frames = [f for p in a.frames for f in load_recorded(a, p)]
        print(f"{len(frames)} aufgenommene Frames aus {len(a.frames)} Datei(en)")

    base = {**DEFAULTS, **parse_kv(a.set, multi=False)}
    grid = parse_kv(a.sweep, multi=True)
    keys = list(grid)
    cfgs = [{**base, **dict(zip(keys, combo))} for combo in itertools.product(*grid.values())] \
        if keys else [base]

    cols = ["hold_absrel", "hold_cov", "hold_near_absrel", "hold_near_cm", "hold_near_cov", "hold_near_bleed",
            "edge_bleed", "edge_cov", "edge_near_bleed", "edge_near_cov", "edge_absrel", "ms"]
    head = " | ".join(f"{k:>6}" for k in keys) + (" | " if keys else "") + \
        "hold AbsRel  cov | <3m AbsRel   cm  cov | edge bleed  cov <3m bleed  cov |    ms"
    print(head + "\n" + "-" * len(head))
    results = []
    for cfg in cfgs:
        r = evaluate(frames, cfg, a.device, a.seed, a.hold_px)
        results.append({**{k: cfg[k] for k in keys}, **r})
        print(" | ".join(f"{cfg[k]:>6g}" for k in keys) + (" | " if keys else "")
              + f"{100 * r['hold_absrel']:10.2f}% {100 * r['hold_cov']:4.0f}% | "
              f"{100 * r['hold_near_absrel']:9.2f}% {r['hold_near_cm']:4.1f} {100 * r['hold_near_cov']:3.0f}% | "
              f"{100 * r['edge_bleed']:9.2f}% {100 * r['edge_cov']:3.0f}% "
              f"{100 * r['edge_near_bleed']:8.2f}% {100 * r['edge_near_cov']:3.0f}% | "
              f"{r['ms']:5.0f}", flush=True)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys + cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        print(f"-> {a.csv}")


if __name__ == "__main__":
    main()
