#!/usr/bin/env python3
"""
Figures of the multi-scene evaluation in the paper, written to paper/figures/.

    python paper/scripts/make_figures.py qual      # RGB | sensor | prediction | completed, 3 scenes
    python paper/scripts/make_figures.py cloud     # rendered 3-D point clouds, with/without edge filter
    python paper/scripts/make_figures.py edge1d    # edge filter along one image row (illustration)
    python paper/scripts/make_figures.py plots     # accuracy vs. latency, error vs. distance, x-t slice

qual/cloud run the deployed filler (MoGe-2 ViT-B, 600 tokens, cached predictions); plots read
paper/results/*.json from eval_paper.py. No model inference happens here.
"""
import json
import math
import os
import sys

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DC = os.path.join(ROOT, "navigation", "mapping", "depth_completion")
sys.path.insert(0, DC)
from depth_fusion import GradientFusion, fill, global_scale  # noqa: E402

FRAMES = os.path.join(ROOT, "rosbags", "frames")
FIG = os.path.join(ROOT, "paper", "figures")
RES = os.path.join(ROOT, "paper", "results")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "savefig.dpi": 300})
COL_W, TEXT_W = 3.5, 7.16                               # IEEE column / text width [in]
ORANGE, MAGENTA = np.array([232, 116, 59]), np.array([212, 83, 126])


def load(scene, i=0, tag="moge2-b600"):
    d = np.load(os.path.join(FRAMES, scene + ".npz"))
    pred = np.load(os.path.join(FRAMES, f"{scene}_preds_{tag}.npz"))["pred"][i].astype(np.float32)
    depth = d["depth"][i].astype(np.float32)
    f_px = 0.5 * depth.shape[1] / math.tan(math.radians(float(d["fov_x"][i])) / 2)
    return d["rgb"][i], depth, pred, f_px


def complete(depth, pred, edge_jump=0.015):
    out, log_s = fill(torch.from_numpy(depth), torch.from_numpy(pred), GradientFusion(), edge_jump=edge_jump)
    est = np.where(np.isfinite(pred), pred * math.exp(log_s), 0).astype(np.float32)
    return out.numpy(), est


def turbo(d, lo, hi):
    n = np.clip((d - lo) / (hi - lo), 0, 1)
    img = plt.get_cmap("turbo")(n)[..., :3]
    img[~(d > 0)] = 0
    return img


# ---------------------------------------------------------------- qualitative

def qual():
    scenes = [("azure_kinect", "Laboratory"), ("octomap", "Desk"), ("rosbag2", "Door and arm")]
    fig, axes = plt.subplots(len(scenes), 4, figsize=(TEXT_W, TEXT_W * 0.42))
    for r, (s, name) in enumerate(scenes):
        rgb, depth, pred, _ = load(s)
        out, est = complete(depth, pred)
        lo, hi = np.percentile(depth[depth > 0], 1), np.percentile(est[est > 0], 99)
        disc = (depth <= 0) & (out <= 0) & (est > 0)
        comp = turbo(out, lo, hi)
        comp[disc] = MAGENTA / 255
        panels = [rgb / 255.0, turbo(depth, lo, hi), turbo(est, lo, hi), comp]
        for c, p in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(p, interpolation="antialiased")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(["RGB", "Sensor depth", "Prediction (global scale)", "Completed"][c], fontsize=8)
        axes[r, 0].set_ylabel(name, fontsize=8)
        print(f"{s}: holes {100 * (depth <= 0).mean():.1f} %, completed {100 * (out > 0).mean():.1f} %, "
              f"discarded {100 * disc.mean():.1f} %")
    fig.subplots_adjust(left=0.03, right=0.995, top=0.93, bottom=0.005, wspace=0.02, hspace=0.03)
    fig.savefig(os.path.join(FIG, "qualitative.png"), dpi=300)
    print("-> qualitative.png")


# ---------------------------------------------------------------- 3-D point clouds

def render(points, colors, size=(1400, 900), yaw=-20.0, pitch=50.0, dist=1.2, target=(0.0, 0.3, 3.0), f=900):
    """Perspective splat rendering with a z-buffer (painter's order), white background."""
    ty, tp = math.radians(yaw), math.radians(pitch)
    fwd = np.array([math.sin(ty) * math.cos(tp), math.sin(tp), math.cos(ty) * math.cos(tp)])
    tgt = np.array(target)
    eye = tgt - fwd * (tgt[2] + dist)                   # behind and above the camera
    right = np.cross(np.array([0.0, -1.0, 0.0]), fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    R = np.stack([right, -up, fwd])                     # camera frame: x right, y down, z forward
    pc = (points - eye) @ R.T
    m = pc[:, 2] > 0.1
    pc, col = pc[m], colors[m]
    W, H = size
    u = (f * pc[:, 0] / pc[:, 2] + W / 2).astype(int)
    v = (f * pc[:, 1] / pc[:, 2] + H / 2).astype(int)
    ok = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    u, v, z, col = u[ok], v[ok], pc[ok, 2], col[ok]
    order = np.argsort(-z)                              # far first, near overwrites
    img = np.ones((H, W, 3))
    for du in (0, 1):
        for dv in (0, 1):
            img[v[order] + dv, u[order] + du] = col[order]
    return img


def backproject(depth, rgb, f_px, mask, stride=1, max_depth=7.0):
    H, W = depth.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[::stride, ::stride]
    m = mask[::stride, ::stride] & (d > 0) & (d < max_depth)
    z = d[m]
    pts = np.stack([(u[m] - W / 2) / f_px * z, (v[m] - H / 2) / f_px * z, z], 1)
    return pts, rgb[::stride, ::stride][m] / 255.0


def cloud():
    rgb, depth, pred, f_px = load("azure_kinect")
    raw, _ = complete(depth, pred, edge_jump=0)
    out, _ = complete(depth, pred)
    meas = depth > 0
    views = [("Sensor", depth, None), ("Completed, no edge filter", raw, raw), ("Completed, edge filter", out, out)]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, TEXT_W * 0.23))
    for ax, (title, d, comp) in zip(axes, views):
        p1, c1 = backproject(depth, rgb, f_px, meas)
        pts, cols = [p1], [c1]
        if comp is not None:
            p2, c2 = backproject(comp, rgb, f_px, ~meas)
            pts.append(p2); cols.append(0.35 * c2 + 0.65 * ORANGE / 255)
        img = render(np.concatenate(pts), np.concatenate(cols))
        ax.imshow(img[60:-40, 90:-90])
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.01, wspace=0.02)
    fig.savefig(os.path.join(FIG, "clouds.png"), dpi=300)
    print("-> clouds.png")


# ---------------------------------------------------------------- edge filter along one row

def edge1d():
    n, E = 64, 32
    i = np.arange(n)
    truth = np.where(i < E, 1.5, 3.0)
    hole = (i >= E - 18) & (i < E + 18)
    model = 1.5 + 1.5 / (1 + np.exp(-(i - E + 0.5) / 1.8))
    q = 1 / model
    rng_q = np.array([np.ptp(q[max(0, k - 2):k + 3]) for k in range(n)])
    edge = rng_q > 0.015
    zone = np.array([edge[max(0, k - 4):k + 5].any() for k in range(n)])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 2.5), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    for ax in (a1, a2):
        ax.axvspan(E - 18.5, E + 17.5, color="0.92", lw=0)
        ax.axvspan(np.where(zone)[0][0] - 0.5, np.where(zone)[0][-1] + 0.5, color="#F2C28B", alpha=0.45, lw=0)
    a1.plot(i, truth, "--", color="0.45", lw=0.8, label="true depth")
    a1.plot(i, model, color="#7F77DD", lw=1.2, label="aligned prediction")
    a1.scatter(i[~hole], truth[~hole], s=5, color="#1D9E75", label="measured", zorder=3)
    keep, drop = hole & ~zone, hole & zone
    a1.scatter(i[keep], model[keep], s=7, color="#378ADD", label="filled", zorder=3)
    a1.scatter(i[drop], model[drop], s=9, marker="x", color="#D4537E", lw=0.8, label="discarded", zorder=3)
    a1.set_ylabel("depth [m]")
    a1.legend(ncol=3, fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.32))
    a2.plot(i, rng_q, color="0.2", lw=1.0)
    a2.axhline(0.015, color="#D85A30", lw=0.8, ls=":")
    a2.text(1, 0.03, r"$\kappa=0.015\,\mathrm{m^{-1}}$", color="#D85A30", fontsize=6)
    a2.set_ylabel(r"$\Delta(1/z)$ [m$^{-1}$]")
    a2.set_xlabel("image column [px]")
    for ax in (a1, a2):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.86, bottom=0.15, hspace=0.12)
    fig.savefig(os.path.join(FIG, "edge_filter.pdf"))
    print("-> edge_filter.pdf")


# ---------------------------------------------------------------- plots from eval results

def plots():
    main = json.load(open(os.path.join(RES, "eval_moge2-b600.json")))
    lat = json.load(open(os.path.join(RES, "latency.json")))
    # accuracy vs. latency
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    names = {"moge2-s600": ("S/600", (4, 2)), "moge2-s1200": ("S/1200", (-6, 6)), "moge2-b600": ("B/600", (5, 2)),
             "moge2-b1200": ("B/1200", (4, 3)), "moge1200": ("L/1200", (-16, 6)), "depthpro2-s": ("S", (4, -7)),
             "depthpro2-b": ("B", (4, -8)), "depthpro2-l": ("L", (-8, 4))}
    for tag, (label, off) in names.items():
        p = os.path.join(RES, f"eval_{tag}.json")
        if not os.path.exists(p) or tag not in lat:
            continue
        r = json.load(open(p))["variants"]["fusion+jump"]["hold128"]
        ms = lat[tag]["total_ms"]
        col = "#D85A30" if tag.startswith("depthpro") else "#2F6DB5"
        ax.scatter(ms, 100 * r["hold_near_absrel"], s=14, color=col, zorder=3,
                   edgecolor="k" if tag == "moge2-b600" else "none", lw=0.8)
        ax.annotate(label, (ms, 100 * r["hold_near_absrel"]), fontsize=6, xytext=off, textcoords="offset points")
    ax.axvline(125, color="0.5", lw=0.7, ls="--")
    ax.text(127, ax.get_ylim()[1] * 0.97, "8 Hz", fontsize=6, color="0.4", va="top")
    ax.set_xscale("log")
    ax.set_xlim(70, 1400)
    ax.scatter([], [], s=14, color="#2F6DB5", label="MoGe-2 (backbone/tokens)")
    ax.scatter([], [], s=14, color="#D85A30", label="Depth Pro 2 (size)")
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.set_xlabel("latency per frame, model + completion [ms]")
    ax.set_ylabel("AbsRel < 3 m, 128 px holes [%]")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.2)
    fig.savefig(os.path.join(FIG, "accuracy_latency.pdf"))
    print("-> accuracy_latency.pdf")

    # error vs. distance to the nearest measurement
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    style = {"global+jump": ("Stage 1 only", "0.5", "--"), "local+jump": ("local mean", "#D85A30", "-."),
             "fusion+jump": ("variational (ours)", "#2F6DB5", "-")}
    for key, (label, col, ls) in style.items():
        r = main["variants"][key]["hold128"]
        b = r["dist_bins"]
        x = [math.sqrt(max(lo, 1) * hi) for lo, hi in zip(b[:-1], b[1:])]
        y = [100 * v if v is not None else np.nan for v in r["dist_absrel"]]
        ax.plot(x, y, ls, color=col, marker="o", ms=3, lw=1.1, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("distance to the nearest remaining measurement [px]")
    ax.set_ylabel("AbsRel < 3 m [%]")
    ax.legend(fontsize=6, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.96, bottom=0.22)
    fig.savefig(os.path.join(FIG, "err_vs_distance.pdf"))
    print("-> err_vs_distance.pdf")

    # x-t slice: deviation of the completed depth from its temporal median in one row
    keys = ["global+jump", "local+jump", "fusion+jump"]
    scene = next(iter(main["variants"]["fusion+jump"]["temporal"]["xt"]))
    depth = np.load(os.path.join(FRAMES, scene + ".npz"))["depth"][:15].astype(np.float32)
    row = 300                                           # eval_paper.py --xt default
    measured = depth[:, row] > 0
    short = {"global+jump": "Stage 1", "local+jump": "local mean", "fusion+jump": "ours"}
    fig, axes = plt.subplots(len(keys), 1, figsize=(COL_W, 2.0), sharex=True)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad((0, 0, 0, 0))
    for ax, key in zip(axes, keys):
        xt = np.array(main["variants"][key]["temporal"]["xt"][scene])
        med = np.median(np.where(xt > 0, xt, np.nan), axis=0)
        dev = np.where((xt > 0) & ~measured, 100 * (xt - med) / med, np.nan)
        ax.imshow(np.where(measured, 0.8, 1.0), aspect="auto", cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        im = ax.imshow(dev, aspect="auto", cmap=cmap, vmin=-4, vmax=4, interpolation="nearest")
        ax.set_xlim(350, dev.shape[1] - 1)
        ax.set_yticks([]); ax.set_ylabel(short[key], fontsize=6)
    axes[-1].set_xlabel("image column [px]")
    cb = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
    cb.set_label("deviation [%]", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.08, right=0.84, top=0.97, bottom=0.18, hspace=0.12)
    fig.savefig(os.path.join(FIG, "xt_slice.pdf"))
    print("-> xt_slice.pdf")


if __name__ == "__main__":
    {"qual": qual, "cloud": cloud, "edge1d": edge1d, "plots": plots}[sys.argv[1]]()
