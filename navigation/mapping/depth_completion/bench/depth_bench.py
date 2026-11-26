#!/usr/bin/env python3
"""
Offline benchmark of the depth completion (no ROS, runs on a laptop).

Runs EXACTLY the production code of ../depth_fusion.py (global scale, gradient
fusion, edge filter) and depth_models.py on recorded or synthetic frames and measures accuracy
AND edge bleeding, so parameters and code changes can be compared deterministically.

Data sources
------------
  --frames F.npz    recorded frames (moge_sweep.py capture or bag_to_frames.py):
                    rgb (N,H,W,3) uint8, depth (N,H,W) float [m], fov_x (N,). Several files
                    are allowed, results are averaged over all of them.
  --preds  P.npz    cached predictions for these frames: pred (N,H,W) [m]
                    (moge_sweep.py infer or --predict)
  --predict MODEL   run the model locally (CUDA / Apple MPS / CPU) and cache its predictions
                    as F_preds_<model>.npz; names from depth_models.MODELS:
                    moge, moge2-s, moge2-b, moge3-l, depthpro2-s / -b / -l / -h
  --synthetic N     N synthetic scenes (floor, wall, boxes, thin poles) with a ToF-like sensor
                    (holes at edges, dark blobs, range drop-out, mixed pixels) and a MoGe-like
                    prediction (soft edge ramps, smooth scale error). Needs no data and no model.

Metrics (each frame with a fresh fusion -> deterministic)
----------------------------------------------------------
  recorded frames, ground truth = sensor measurements hidden on purpose:
    hold_*  : random 32 px blocks (--hold-px) of measured pixels hidden
    edge_*  : a 6 px band around measured depth edges hidden (what the ToF really loses at
              edges); evaluated on that band
  synthetic frames, ground truth = known scene depth:
    hold_*  : all real sensor holes away from edges
    edge_*  : all real sensor holes within 6 px of a depth edge
  *_near_*  : the same, only for pixels below 3 m

    absrel  : mean |est - gt| / gt over the filled pixels
    cm      : mean absolute error [cm]
    cov     : fraction of the evaluated pixels that got a value at all
    bleed   : fraction of the evaluated pixels with > 10 % error (= floating points in the
              map). Lower is better; always read with cov (filling nothing gives bleed 0).

Examples
--------
  python depth_bench.py --synthetic 8
  python depth_bench.py --synthetic 8 --sweep edge_jump=0,0.015,0.02 edge_grow=4,8
  python depth_bench.py --frames frames.npz --preds preds.npz --set down=4
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
FILL_KEYS = ("ema", "edge_jump", "edge_grow")
INT_KEYS = ("down", "iters", "edge_grow")
DEFAULTS = {}            # otherwise the defaults of depth_fusion.py apply
NEAR_M = 3.0            # near range for the near metrics
BLEED_REL = 0.10        # error above which a filled pixel counts as an outlier
EDGE_BAND_PX = 6
MEAS_EDGE_RTOL = 0.10   # edge in the measurement: 3x3 range > 10 %


# ---------------------------------------------------------------- helpers

def edge_mask_np(d, rtol):
    """Pixels whose 3x3 depth range is > rtol·depth (holes ignored)."""
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
    """Overall and near range (gt < NEAR_M)."""
    out = {f"{prefix}_{k}": v for k, v in score(filled, gt, mask).items()}
    out.update({f"{prefix}_near_{k}": v for k, v in score(filled, gt, mask & (gt < NEAR_M)).items()})
    return out


# ---------------------------------------------------------------- synthetic scene

def _smooth_noise(rng, H, W, cells=(6, 10)):
    return cv2.resize(rng.standard_normal(cells).astype(np.float32), (W, H),
                      interpolation=cv2.INTER_CUBIC)


def synth_frame(rng, H=720, W=1280, fx=610.0, cam_h=0.58):
    """(gt, sensor, pred) in metres. Camera 0.58 m high, tilted slightly down."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    x, y = (u - W / 2) / fx, (v - H / 2) / fx                  # ray with z = 1
    p = math.radians(rng.uniform(8, 18))
    g = np.cos(p) * y + np.sin(p)                              # floor: g·X = cam_h
    depth = np.where(g > 1e-3, cam_h / np.maximum(g, 1e-3), np.inf)
    a = math.radians(rng.uniform(-20, 20))                     # slanted back wall
    zw = rng.uniform(3.0, 5.5)
    nr = np.sin(a) * x + np.cos(a)
    depth = np.minimum(depth, np.where(nr > 1e-3, zw / nr, np.inf))
    for _ in range(rng.integers(3, 6)):                        # boxes (front faces)
        z = rng.uniform(0.7, 2.6)
        w, h = rng.uniform(0.25, 0.7), rng.uniform(0.25, 0.9)
        cx, top = rng.uniform(-1.0, 1.0) * z * 0.6, rng.uniform(-0.4, 0.2)
        inside = (np.abs(x * z - cx) < w / 2) & (y * z > top) & (y * z < top + h)
        depth = np.where(inside, np.minimum(depth, z), depth)
    for _ in range(rng.integers(1, 3)):                        # thin poles
        z, cx = rng.uniform(0.8, 2.0), rng.uniform(-0.8, 0.8)
        inside = np.abs(x * z - cx) < rng.uniform(0.01, 0.03)
        depth = np.where(inside, np.minimum(depth, z), depth)
    gt = np.where(np.isfinite(depth), depth, 0).astype(np.float32)

    # sensor (ToF-like)
    sensor = gt * (1 + 0.002 * gt * rng.standard_normal(gt.shape).astype(np.float32))
    edges = edge_mask_np(gt, 0.05)
    holes = dilate(edges, 2) & (rng.random(gt.shape) < 0.85)   # holes at edges
    fly = edges & ~holes                                        # remaining mixed pixels
    sensor[fly] = cv2.blur(gt, (5, 5))[fly]
    for _ in range(rng.integers(2, 6)):                         # dark blobs
        c = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        ax = (int(rng.integers(20, 90)), int(rng.integers(15, 60)))
        blob = np.zeros(gt.shape, np.uint8)
        cv2.ellipse(blob, c, ax, float(rng.uniform(0, 180)), 0, 360, 1, -1)
        holes |= blob > 0
    holes |= (gt > 4.5) & (rng.random(gt.shape) < np.clip((gt - 4.5) / 1.5, 0, 1))
    holes[:, :25] = holes[:, -25:] = True                      # FoV border
    sensor[holes | (gt <= 0)] = 0

    # MoGe-like prediction: soft edges, smooth scale error
    lg = cv2.GaussianBlur(np.log(np.where(gt > 0, gt, 1)), (0, 0), 2.5)
    pred = np.exp(lg + 0.06 * _smooth_noise(rng, H, W)) * rng.uniform(0.9, 1.1)
    pred *= 1 + 0.005 * rng.standard_normal(gt.shape)
    pred = np.where(gt > 0, pred, np.inf).astype(np.float32)
    return gt, sensor.astype(np.float32), pred


# ---------------------------------------------------------------- data loading

def load_recorded(a, frames_path):
    data = np.load(frames_path)
    depth = data["depth"].astype(np.float32)
    n = min(len(depth), a.limit) if a.limit else len(depth)
    fov = data["fov_x"].astype(np.float64) if "fov_x" in data else np.full(len(depth), np.nan)
    tag = f"{a.predict}{a.num_tokens}" if a.predict in MOGE else a.predict
    preds_path = a.preds or (os.path.splitext(frames_path)[0] + f"_preds_{tag}.npz"
                             if a.predict else None)
    cached = np.load(preds_path)["pred"] if preds_path and os.path.exists(preds_path) else None
    if cached is not None and (len(cached) >= n or not a.predict):
        pred = cached[:n].astype(np.float32)
        n = len(pred)
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
        raise SystemExit(f"--preds missing (or --predict {'|'.join(MODELS)} to run a model locally)")
    return [(None, depth[i], pred[i]) for i in range(n)]


# ---------------------------------------------------------------- evaluation

def make_filler(cfg):
    """fill with the parameters of cfg as f(depth_t, pred_t, log_s) -> (out, log_s).
    The fusion object (warm start) lives as long as the returned function."""
    ints = lambda d: {k: (int(v) if k in INT_KEYS else v) for k, v in d.items()}
    fusion = GradientFusion(**ints({k: cfg[k] for k in FUSION_KEYS if k in cfg}))
    kw = ints({k: cfg[k] for k in FILL_KEYS if k in cfg})
    return lambda d, p, s: fill(d, p, fusion, s, **kw)


def run_fill(depth, pred, cfg, dev):
    t0 = time.perf_counter()
    out, _ = make_filler(cfg)(torch.from_numpy(depth).to(dev), torch.from_numpy(pred).to(dev), None)
    out = out.cpu().numpy()
    return out, 1000 * (time.perf_counter() - t0)


def temporal(scenes, cfg, dev):
    """Flicker on static recordings: each scene is run as a sequence with state (smoothed global
    scale, warm start), like the live node. Mean relative change between consecutive frames:
      filled : on sensor holes (in both frames) that are filled in both frames
      model  : model depth x global scale on the same pixels (reference: flicker of the model)
      sensor : on pixels measured in both frames (reference: sensor noise)
    plus the share of hole pixels that get filled (cov)."""
    acc = {k: [] for k in ("filled", "filled_near", "model", "sensor", "cov")}
    for frames in scenes:
        fill_fn, log_s, prev = make_filler(cfg), None, None
        for _, depth, pred in frames:
            d_t, p_t = torch.from_numpy(depth).to(dev), torch.from_numpy(pred).to(dev)
            out, log_s = fill_fn(d_t, p_t, log_s)
            out = out.cpu().numpy()
            model = np.where(np.isfinite(pred), pred * math.exp(log_s), 0)
            hole = depth <= 0
            acc["cov"].append(float((out[hole] > 0).mean()))
            if prev is not None:
                p_depth, p_out, p_model = prev
                m = hole & (p_depth <= 0) & (out > 0) & (p_out > 0)
                rel = np.abs(out - p_out)[m] / p_out[m]
                acc["filled"].append(float(rel.mean()))
                near = (p_out[m] < NEAR_M)
                acc["filled_near"].append(float(rel[near].mean()) if near.any() else np.nan)
                mm = m & (model > 0) & (p_model > 0)
                acc["model"].append(float((np.abs(model - p_model)[mm] / p_model[mm]).mean()))
                ms = (depth > 0) & (p_depth > 0)
                acc["sensor"].append(float((np.abs(depth - p_depth)[ms] / p_depth[ms]).mean()))
            prev = (depth, out, model)
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def evaluate(frames, cfg, dev, seed=0, hold_px=32):
    rows, ms = [], []
    for i, (gt, depth, pred) in enumerate(frames):
        rng = np.random.default_rng(seed + i)
        if gt is not None:                                   # synthetic: real holes
            filled, t = run_fill(depth, pred, cfg, dev)
            holes = (depth <= 0) & (gt > 0)
            near = dilate(edge_mask_np(gt, 0.05), EDGE_BAND_PX)
            hold, edge = score2(filled, gt, holes & ~near, "hold"), score2(filled, gt, holes & near, "edge")
            ms.append(t)
        else:                                                # recording: hide measurements
            valid = depth > 0
            H, W = depth.shape
            pick = rng.random((-(-H // hold_px), -(-W // hold_px))) < 0.15
            blocks = np.kron(pick, np.ones((hold_px, hold_px), bool))[:H, :W] & valid
            f1, t = run_fill(np.where(blocks, 0, depth).astype(np.float32), pred, cfg, dev)
            hold = score2(f1, depth, blocks, "hold")
            e = edge_mask_np(depth, MEAS_EDGE_RTOL)
            band = dilate(e, EDGE_BAND_PX) & valid & ~e        # edge pixels themselves: uncertain
            f2, _ = run_fill(np.where(band, 0, depth).astype(np.float32), pred, cfg, dev)
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
        allowed = FUSION_KEYS + FILL_KEYS
        if k not in allowed:
            raise SystemExit(f"unknown parameter '{k}', allowed: {allowed}")
        vals = [float(x) for x in v.split(",")]
        vals = [int(x) if k in INT_KEYS else x for x in vals]
        out[k] = vals if multi else vals[0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", nargs="+", help="one or more frames.npz (averaged over all)")
    src.add_argument("--synthetic", type=int)
    ap.add_argument("--preds")
    ap.add_argument("--predict", choices=MODELS,
                    help="run the model locally (and cache)")
    ap.add_argument("--num-tokens", type=int, default=1200, help="MoGe only")
    ap.add_argument("--model-device", default=None, help="cuda / mps / cpu (default: automatic)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None,
                    help="device for the fusion (cpu / cuda / mps), default: cuda, else mps, else cpu")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (limited so the machine stays usable)")
    ap.add_argument("--set", nargs="*", help="fixed parameters, e.g. down=4 edge_grow=4")
    ap.add_argument("--sweep", nargs="*", help="grid, e.g. edge_jump=0.01,0.015 edge_grow=4,8")
    ap.add_argument("--csv", help="also write results as CSV")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold-px", type=int, default=32, help="side length of the hidden blocks")
    ap.add_argument("--temporal", action="store_true",
                    help="flicker over time instead of accuracy (each --frames file = one sequence)")
    a = ap.parse_args()

    torch.set_num_threads(max(1, a.threads))
    a.device = a.device or ("cuda" if torch.cuda.is_available() else
                            "mps" if torch.backends.mps.is_available() else "cpu")
    if a.synthetic:
        rng = np.random.default_rng(a.seed)
        frames = [synth_frame(rng) for _ in range(a.synthetic)]
        print(f"{len(frames)} synthetische Frames")
    else:
        if a.preds and len(a.frames) > 1:
            raise SystemExit("--preds only works with a single --frames file")
        scenes = [load_recorded(a, p) for p in a.frames]
        frames = [f for sc in scenes for f in sc]
        print(f"{len(frames)} aufgenommene Frames aus {len(a.frames)} Datei(en)")

    base = {**DEFAULTS, **parse_kv(a.set, multi=False)}
    grid = parse_kv(a.sweep, multi=True)
    keys = list(grid)
    cfgs = [{**base, **dict(zip(keys, combo))} for combo in itertools.product(*grid.values())] \
        if keys else [base]

    if a.temporal:
        if a.synthetic:
            raise SystemExit("--temporal needs recorded --frames")
        head = " | ".join(f"{k:>9}" for k in keys) + (" | " if keys else "") + \
            "flicker filled  <3m | model | sensor | filled cov"
        print(head + "\n" + "-" * len(head))
        for cfg in cfgs:
            r = temporal(scenes, cfg, a.device)
            print(" | ".join(f"{cfg[k]:>9g}" for k in keys) + (" | " if keys else "")
                  + f"{100 * r['filled']:13.2f}% {100 * r['filled_near']:5.2f}% | {100 * r['model']:4.2f}% | "
                  f"{100 * r['sensor']:5.2f}% | {100 * r['cov']:9.0f}%", flush=True)
        return

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
