#!/usr/bin/env python3
"""
Evaluation behind the multi-scene results of the paper (Sec. "Multi-scene evaluation").

Runs on the cached predictions of navigation/mapping/depth_completion/bench (rosbags/frames/
*_preds_<model>.npz, created with depth_bench.py --predict ... --limit 15) and writes
paper/results/eval_<model>.json. No model inference happens here.

    python paper/scripts/eval_paper.py                         # MoGe-2 ViT-B, 600 tokens: all ablations
    python paper/scripts/eval_paper.py --model depthpro2-l --only main

Variants (alignment x edge handling):
    alignment: global  = Stage 1 only (median scale)
               local   = Stage 1 + local mean of the log residual within ~16 px (box filter)
               fusion  = Stage 1 + variational correction (GradientFusion, deployed)
    edges:     none    = no edge handling
               rel3x3  = former filter: relative 3x3 depth range > 4 % on the completed image
               plane   = former filters: plane distance < 0.5 m (ramp) + support check
               jump    = deployed filter: jump in 1/z > 0.015 in 5x5, dilated by 4 px
The two former filters are reimplemented below from the git history for this ablation.

Protocols (as in bench/depth_bench.py): hold-out blocks of 32 and 128 px covering 15 % of the
measured pixels, a 6 px band around measured depth edges, and temporal flicker on the static
recordings (sequence with smoothed scale and warm start). Errors are reported below 3 m.
"""
import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DC = os.path.join(ROOT, "navigation", "mapping", "depth_completion")
sys.path.insert(0, DC)
sys.path.insert(0, os.path.join(DC, "bench"))
import depth_bench as B  # noqa: E402
from depth_fusion import GradientFusion, global_scale, jump_edges  # noqa: E402

FRAMES = os.path.join(ROOT, "rosbags", "frames")
SCENES = ["depth_estimation", "octomap", "azure_kinect", "rosbag", "rosbag2", "moge_szene"]
NEAR = 3.0
DEV = torch.device("mps" if torch.backends.mps.is_available() else
                   "cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- former edge filters

def depth_edge_rel(d, rtol=0.04, k=3):
    v = d > 0
    dmax = F.max_pool2d(torch.where(v, d, torch.zeros_like(d))[None, None], k, 1, k // 2)[0, 0]
    dmin = -F.max_pool2d(-torch.where(v, d, torch.full_like(d, float("inf")))[None, None], k, 1, k // 2)[0, 0]
    return v & ((dmax - dmin) > rtol * d)


def ramp_mask(est, f_px, dmin=0.5, s=2):
    v = est > 0
    q = torch.where(v, 1.0 / torch.where(v, est, torch.ones_like(est)), torch.zeros_like(est))
    g2 = torch.zeros_like(q)
    gu = (q[:, 2 * s:] - q[:, : -2 * s]) / (2 * s)
    gv = (q[2 * s:, :] - q[: -2 * s, :]) / (2 * s)
    g2[:, s:-s] += torch.where(v[:, 2 * s:] & v[:, : -2 * s], gu * gu, torch.zeros_like(gu))
    g2[s:-s, :] += torch.where(v[2 * s:, :] & v[: -2 * s, :], gv * gv, torch.zeros_like(gv))
    return (f_px * f_px) * g2 > 1.0 / (dmin * dmin)


def supported(est, depth_m, r=12, rtol=0.02, edge=0.1):
    v = depth_m > 0
    k = 2 * r + 1

    def mx(a):
        a = F.max_pool2d(a[None, None], (1, k), 1, (0, r))
        return F.max_pool2d(a, (k, 1), 1, (r, 0))[0, 0]

    dmax = mx(torch.where(v, depth_m, torch.zeros_like(depth_m)))
    dmin = -mx(-torch.where(v, depth_m, torch.full_like(depth_m, 1e9)))
    inside = (est >= dmin * (1 - rtol)) & (est <= dmax * (1 + rtol))
    near_side = (torch.abs(est - dmin) <= rtol * dmin) | (torch.abs(est - dmax) <= rtol * dmax)
    is_edge = dmax > dmin * (1 + edge)
    return (dmax <= 0) | (inside & (~is_edge | near_side))


def local_mean(depth_m, est, radius=16, f=4):
    """Local alternative to the fusion: mean log residual of the measurements within ~radius px."""
    H, W = depth_m.shape
    both = (depth_m > 0) & (est > 0)
    r = torch.where(both, torch.log(torch.where(both, depth_m, torch.ones_like(est)) /
                                    torch.where(both, est, torch.ones_like(est))), torch.zeros_like(est))
    k = 2 * max(radius // f, 1) + 1
    pool = lambda a: F.avg_pool2d(F.avg_pool2d(a[None, None], f, f), k, 1, k // 2, count_include_pad=True)
    c = pool(r) / torch.clamp(pool(both.float()), min=0.05)
    c = F.interpolate(c, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    return torch.where(est > 0, est * torch.exp(c), est)


# ---------------------------------------------------------------- one frame of a variant

class Filler:
    """Stateful filler for one sequence (smoothed scale, warm start)."""

    def __init__(self, align, edges, f_px):
        self.align, self.edges, self.f_px = align, edges, f_px
        self.fusion = GradientFusion(iters=0) if align == "global" else GradientFusion()
        self.log_s = None

    @torch.inference_mode()
    def __call__(self, depth_m, pred):
        valid = depth_m > 0
        est, self.log_s = global_scale(depth_m, pred, self.log_s)
        if self.align == "fusion":
            est = self.fusion(depth_m, est)
        elif self.align == "local":
            est = local_mean(depth_m, est)
        keep = est > 0
        if self.edges == "jump":
            keep &= ~jump_edges(est, 0.015, 4)
        elif self.edges == "plane":
            keep &= ~ramp_mask(est, self.f_px) & supported(est, depth_m)
        out = torch.where(valid, depth_m, torch.where(keep, est, torch.zeros_like(est)))
        if self.edges == "rel3x3":
            out = torch.where(depth_edge_rel(out) & ~valid, torch.zeros_like(out), out)
        return out, est


def sync():
    if DEV.type == "mps":
        torch.mps.synchronize()
    elif DEV.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------- data

def load_scenes(model, tokens, n, tag=None):
    tag = tag or (f"{model}{tokens}" if model in B.MOGE else model)
    scenes = []
    for s in SCENES:
        data = np.load(os.path.join(FRAMES, s + ".npz"))
        pred = np.load(os.path.join(FRAMES, f"{s}_preds_{tag}.npz"))["pred"].astype(np.float32)
        m = min(n, len(pred))
        depth = data["depth"][:m].astype(np.float32)
        fov = float(data["fov_x"][0])
        f_px = 0.5 * depth.shape[2] / math.tan(math.radians(fov) / 2)
        scenes.append(SimpleNamespace(name=s, depth=depth, pred=pred[:m], f_px=f_px,
                                      rgb=data["rgb"][:m]))
    return scenes


# ---------------------------------------------------------------- protocols

def holdout(scenes, align, edges, hold_px, frames=2, seed=0, dist_bins=None, last=False):
    """last=True evaluates the last `frames` frames of each sequence (e.g. streaming predictions
    with a full window) with the same hidden blocks as the first ones."""
    rows, ms, dist_err = [], [], []
    for sc in scenes:
        n = min(frames, len(sc.depth))
        for k in range(n):
            i = len(sc.depth) - n + k if last else k
            depth, pred = sc.depth[i], sc.pred[i]
            rng = np.random.default_rng(seed + k)
            valid = depth > 0
            H, W = depth.shape
            pick = rng.random((-(-H // hold_px), -(-W // hold_px))) < 0.15
            blocks = np.kron(pick, np.ones((hold_px, hold_px), bool))[:H, :W] & valid
            hidden = np.where(blocks, 0, depth).astype(np.float32)
            fill = Filler(align, edges, sc.f_px)
            p_t = torch.from_numpy(pred).to(DEV)
            sync(); t0 = time.perf_counter()
            out, _ = fill(torch.from_numpy(hidden).to(DEV), p_t)
            sync(); ms.append(1000 * (time.perf_counter() - t0))
            f1 = out.cpu().numpy()
            hold = B.score2(f1, depth, blocks, "hold")
            e = B.edge_mask_np(depth, B.MEAS_EDGE_RTOL)
            band = B.dilate(e, B.EDGE_BAND_PX) & valid & ~e
            out2, _ = Filler(align, edges, sc.f_px)(torch.from_numpy(np.where(band, 0, depth).astype(np.float32)).to(DEV), p_t)
            edge = B.score2(out2.cpu().numpy(), depth, band, "edge")
            rows.append({**hold, **edge})
            if dist_bins is not None:                   # error over distance to nearest measurement
                d = cv2.distanceTransform((hidden <= 0).astype(np.uint8), cv2.DIST_L2, 5)
                m = blocks & (f1 > 0) & (depth < NEAR)
                rel = np.abs(f1 - depth)[m] / depth[m]
                dist_err.append((d[m], rel))
    res = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    res["ms"] = float(np.median(ms))
    if dist_bins is not None:
        d = np.concatenate([x[0] for x in dist_err]); rel = np.concatenate([x[1] for x in dist_err])
        res["dist_bins"] = dist_bins
        res["dist_absrel"] = [float(rel[(d >= a) & (d < b)].mean()) if ((d >= a) & (d < b)).sum() > 200 else None
                              for a, b in zip(dist_bins[:-1], dist_bins[1:])]
    return res


def temporal(scenes, align, edges, keep_row=None):
    acc = {k: [] for k in ("filled", "filled_near", "model", "sensor", "cov")}
    rows = {}
    for sc in scenes:
        fill, prev = Filler(align, edges, sc.f_px), None
        stack = []
        for depth, pred in zip(sc.depth, sc.pred):
            out, est = fill(torch.from_numpy(depth).to(DEV), torch.from_numpy(pred).to(DEV))
            out = out.cpu().numpy()
            model = np.where(np.isfinite(pred), pred * math.exp(fill.log_s), 0)
            hole = depth <= 0
            acc["cov"].append(float((out[hole] > 0).mean()))
            if prev is not None:
                p_depth, p_out, p_model = prev
                m = hole & (p_depth <= 0) & (out > 0) & (p_out > 0)
                rel = np.abs(out - p_out)[m] / p_out[m]
                acc["filled"].append(float(rel.mean()))
                near = p_out[m] < NEAR
                acc["filled_near"].append(float(rel[near].mean()) if near.any() else np.nan)
                mm = m & (model > 0) & (p_model > 0)
                acc["model"].append(float((np.abs(model - p_model)[mm] / p_model[mm]).mean()))
                ms_ = (depth > 0) & (p_depth > 0)
                acc["sensor"].append(float((np.abs(depth - p_depth)[ms_] / p_depth[ms_]).mean()))
            prev = (depth, out, model)
            if keep_row and sc.name == keep_row[0]:
                stack.append(out[keep_row[1]].copy())
        if stack:
            rows[sc.name] = np.stack(stack).tolist()
    res = {k: float(np.nanmean(v)) for k, v in acc.items()}
    if rows:
        res["xt"] = rows
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="moge2-b")
    ap.add_argument("--tokens", type=int, default=600)
    ap.add_argument("--only", choices=("all", "main"), default="all",
                    help="main = deployed variant only (fusion + jump)")
    ap.add_argument("--xt", default="azure_kinect:300", help="scene:row for the x-t slice")
    ap.add_argument("--tag", help="prediction cache tag, e.g. depthpro2-s-stream16 (default from model)")
    ap.add_argument("--n", type=int, default=15, help="frames per recording")
    ap.add_argument("--last", action="store_true", help="hold-out on the last 2 frames instead of the first")
    a = ap.parse_args()
    torch.set_num_threads(4)

    scenes = load_scenes(a.model, a.tokens, a.n, a.tag)
    variants = [("fusion", "jump")] if a.only == "main" else [
        ("fusion", "none"), ("fusion", "rel3x3"), ("fusion", "plane"), ("fusion", "jump"),
        ("global", "jump"), ("local", "jump")]
    xt_scene, xt_row = a.xt.split(":")
    out = {"model": a.model, "tokens": a.tokens, "device": str(DEV),
           "frames_temporal": {sc.name: len(sc.depth) for sc in scenes}, "variants": {}}
    for align, edges in variants:
        key = f"{align}+{edges}"
        print(f"{key} ...", flush=True)
        r = {"hold32": holdout(scenes, align, edges, 32, last=a.last),
             "hold128": holdout(scenes, align, edges, 128, dist_bins=[0, 8, 16, 32, 64, 128], last=a.last),
             "temporal": temporal(scenes, align, edges, keep_row=(xt_scene, int(xt_row)))}
        out["variants"][key] = r
        h, g, t = r["hold32"], r["hold128"], r["temporal"]
        print(f"  <3m small {100 * h['hold_near_absrel']:.2f}% cov {100 * h['hold_near_cov']:.0f}% | "
              f"large {100 * g['hold_near_absrel']:.2f}% {g['hold_near_cm']:.1f}cm cov {100 * g['hold_near_cov']:.0f}% | "
              f"edge bleed {100 * h['edge_bleed']:.2f}% cov {100 * h['edge_cov']:.0f}% | "
              f"flicker {100 * t['filled']:.2f}% | {h['ms']:.0f} ms", flush=True)
    os.makedirs(os.path.join(ROOT, "paper", "results"), exist_ok=True)
    name = a.tag or f"{a.model}{a.tokens if a.model in B.MOGE else ''}"
    suffix = (f"_n{a.n}" if a.n != 15 else "") + ("_last" if a.last else "")
    path = os.path.join(ROOT, "paper", "results", f"eval_{name}{suffix}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print("->", path)


if __name__ == "__main__":
    main()
