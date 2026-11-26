#!/usr/bin/env python3
"""
Runtime of the whole depth completion per frame: model + fill (fusion and filters), on the
same device as the ROS node. One frame of a frames.npz, median over --reps.

    python speed.py --frames f.npz --models moge2-s:600 moge2-b:1200 depthpro2-s
    python speed.py --frames f.npz --fill-only --set down=2

model:num_tokens applies to MoGe; Depth Pro 2 always runs at full resolution.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from depth_bench import FILL_KEYS, FUSION_KEYS, parse_kv  # noqa: E402
from depth_fusion import GradientFusion, fill  # noqa: E402
from depth_models import load_model  # noqa: E402

PROD = {}   # defaults of depth_fusion.py (= production), --set overrides


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, dev, reps):
    fn()
    sync(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        sync(dev)
        ts.append(1000 * (time.perf_counter() - t0))
    return out, float(np.median(ts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=["moge2-s:1200", "depthpro2-s"])
    ap.add_argument("--device", default=None, help="default: cuda, else mps, else cpu")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--set", nargs="*", help="fill/fusion parameters, override PROD")
    ap.add_argument("--fill-only", action="store_true", help="fill only, with a dummy prediction")
    a = ap.parse_args()

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else
                                    "mps" if torch.backends.mps.is_available() else "cpu"))
    data = np.load(a.frames)
    rgb, depth = data["rgb"][a.index], data["depth"][a.index].astype(np.float32)
    fov = float(data["fov_x"][a.index])
    cfg = {**PROD, **parse_kv(a.set, multi=False)}
    fus = {k: cfg[k] for k in FUSION_KEYS if k in cfg}
    kw = {k: (int(cfg[k]) if k == "edge_grow" else cfg[k]) for k in FILL_KEYS if k in cfg}
    d_t = torch.from_numpy(depth).to(dev)
    print(f"Gerät {dev}, Bild {rgb.shape[1]}x{rgb.shape[0]}, fill: {cfg}")

    def run_fill(pred):
        return fill(d_t, pred, GradientFusion(**fus), **kw)[0]

    if a.fill_only:
        pred = torch.from_numpy(np.where(depth > 0, depth, 2.0).astype(np.float32)).to(dev)
        _, ms = timed(lambda: run_fill(pred), dev, a.reps)
        print(f"fill {ms:6.0f} ms")
        return

    print(f"{'Modell':18s} {'Modell':>8s} {'fill':>6s} {'gesamt':>7s} {'Hz':>5s}")
    for spec in a.models:
        name, _, tok = spec.partition(":")
        model = load_model(name, int(tok or 1200), str(dev))
        pred, t_model = timed(lambda: model(rgb, fov), dev, a.reps)
        pred = pred.float().to(dev)
        _, t_fill = timed(lambda: run_fill(pred), dev, a.reps)
        tot = t_model + t_fill
        print(f"{spec:18s} {t_model:6.0f}ms {t_fill:4.0f}ms {tot:5.0f}ms {1000 / tot:5.1f}", flush=True)
        del model
        if dev.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
