#!/usr/bin/env python3
"""
Run several monocular depth models on the same RGB image and compare them
(default: Apple Depth Pro 2 S and B).

Prints runtime per model and an edge-sharpness indicator, and writes a PNG with
RGB | one panel per model (shared turbo colormap), below each model's depth edges.

    python compare_models.py image.jpg [--fov-x 90] [--models depthpro2-s depthpro2-b]
    python compare_models.py --frames frames.npz --index 0

Edge sharpness: fraction of pixels whose 3x3 relative depth range lies between 2 % and 10 %.
Real depth edges are steps (range > 10 %) and flat surfaces stay below 2 %, so a high value
means many "ramp" pixels, i.e. soft, smeared edges.
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_bench import MODELS, edge_mask_np, load_model  # noqa: E402


def colorize(d, lo, hi):
    n = np.clip((np.log(np.maximum(d, 1e-3)) - math.log(lo)) / (math.log(hi) - math.log(lo)), 0, 1)
    img = cv2.applyColorMap((255 * (1 - n)).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~np.isfinite(d) | (d <= 0)] = 0
    return img


def ramp_fraction(d):
    d = np.where(np.isfinite(d), d, 0).astype(np.float32)
    soft = edge_mask_np(d, 0.02) & ~edge_mask_np(d, 0.10)
    return float(soft[d > 0].mean())


def label(img, text):
    for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?")
    ap.add_argument("--frames")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--fov-x", type=float, default=None, help="horizontal field of view [deg]")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=["depthpro2-s", "depthpro2-b"])
    ap.add_argument("--num-tokens", type=int, default=1200, help="MoGe only")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="compare_models.png")
    a = ap.parse_args()

    if a.frames:
        data = np.load(a.frames)
        rgb = data["rgb"][a.index]
        fov = a.fov_x or (float(data["fov_x"][a.index]) if "fov_x" in data else None)
    else:
        rgb = cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB)
        fov = a.fov_x
    print(f"Bild {rgb.shape[1]}x{rgb.shape[0]}, FoV_x {fov if fov else 'vom Modell geschätzt'}")

    runs = [((("MoGe-2-L" if m == "moge" else m) + f" {a.num_tokens} Tok.") if m.startswith("moge")
             else "DepthPro2-" + m.split("-")[1].upper(), m, a.num_tokens) for m in a.models]
    results = []
    for name, kind, tok in runs:
        model = load_model(kind, tok, a.device)
        model(rgb, fov)                                    # warm-up
        if model.dev.type == "mps":
            torch.mps.synchronize()
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            d = model(rgb, fov)
            if model.dev.type == "mps":
                torch.mps.synchronize()
            elif model.dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append(1000 * (time.perf_counter() - t0))
        d = d.float().cpu().numpy()
        if d.shape != rgb.shape[:2]:
            d = cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        results.append((name, d, float(np.median(ts))))
        v = d[np.isfinite(d) & (d > 0)]
        print(f"{name:16s} {model.dev.type:4s} {np.median(ts):7.0f} ms | Tiefe {np.percentile(v, 2):.2f}-"
              f"{np.percentile(v, 98):.2f} m | Rampen-Pixel {100 * ramp_fraction(d):5.2f} %")
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    allv = np.concatenate([d[np.isfinite(d) & (d > 0)] for _, d, _ in results])
    lo, hi = np.percentile(allv, 2), np.percentile(allv, 98)
    W = 640
    H = int(rgb.shape[0] * W / rgb.shape[1])
    top = [cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (W, H))]
    bottom = [np.zeros((H, W, 3), np.uint8)]
    for name, d, ms in results:
        p = cv2.resize(colorize(d, lo, hi), (W, H))
        label(p, f"{name}  {ms:.0f} ms")
        top.append(p)
        e = np.where(np.isfinite(d), d, 0).astype(np.float32)
        soft = edge_mask_np(e, 0.02) & ~edge_mask_np(e, 0.10)
        hard = edge_mask_np(e, 0.10)
        vis = np.zeros(d.shape + (3,), np.uint8)
        vis[soft] = (0, 165, 255)      # orange: ramp (soft edge)
        vis[hard] = (255, 255, 255)    # white: hard edge
        vis = cv2.resize(vis, (W, H), interpolation=cv2.INTER_AREA)
        label(vis, f"Kanten: weiss hart, orange Rampe {100 * ramp_fraction(d):.1f} %")
        bottom.append(vis)
    label(top[0], "RGB")
    cv2.imwrite(a.out, np.vstack([np.hstack(top), np.hstack(bottom)]))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
