#!/usr/bin/env python3
"""
Test suite for the depth completion on the robot PC (Linux, NVIDIA GPU). Runs every model
configuration on recorded scenes with the deployed completion (fusion + edge filter) and writes
a report folder that can be sent for analysis. No ROS needed.

    python run_suite.py --bags ~/Desktop/rosbags/depth_estimation ~/Desktop/rosbags/octomap/octomap
    python run_suite.py --frames a.npz b.npz              # recordings already converted
    python run_suite.py --bags ... --quick                # small grid, first check (~10 min)
    python run_suite.py --bags ... --only moge2-b:600 depthpro2-s:stream4
    python run_suite.py --bags ... --resume reports/<run>  # continue an interrupted run

Configurations (default grid, see --list):
    MoGe-2  S, B, L, L-normal          x tokens 600, 1200, 2400
    MoGe-3  L                          x tokens 600, 1200, 2400 (refinement, needs flex_gemm)
            L without refinement, 1200 tokens; G with refinement, 600 and 1200 tokens
    Depth Pro 2  S, B, L (H with --with-h)  x per frame, stream window 4, stream window 16
A configuration that fails (out of memory, missing package) is logged and skipped.

Per configuration and recording:
    latency    model and completion per frame (median after 2 warm-up frames), GPU memory peak
    accuracy   hidden blocks of 32 and 128 px and a 6 px band around measured depth edges on the
               last 2 frames (streaming has a full window there): AbsRel, MAE, coverage and
               gross errors (> 10 %), below 3 m and over all depths
    stability  frame-to-frame change of the completed depth on holes (static camera), of the
               prediction alone and of the sensor; fraction of real holes that get filled

Report (<out>/): env.json (GPU, driver, versions, git commit), results.jsonl (one line per
configuration incl. per-recording values), summary.csv, report.md, suite.log.

Do not run the ROS pipeline at the same time: it shares the GPU and distorts the timings.
"""
import argparse
import csv
import datetime
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", message="Mean of empty slice")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import depth_bench as B  # noqa: E402
from depth_fusion import GradientFusion, fill  # noqa: E402
from depth_models import load_model  # noqa: E402

NEAR = 3.0
WARMUP = 2

MOGE2 = [("moge2-s", "MoGe-2 ViT-S"), ("moge2-b", "MoGe-2 ViT-B"), ("moge", "MoGe-2 ViT-L"),
         ("moge2-l-normal", "MoGe-2 ViT-L normal")]
TOKENS = [600, 1200, 2400]


def default_grid(with_h=False):
    g = [f"{m}:{t}" for m, _ in MOGE2 for t in TOKENS]
    g += [f"moge3-l:{t}:refine3" for t in TOKENS] + ["moge3-l:1200:refine0"]
    g += [f"moge3-g:{t}:refine3" for t in (600, 1200)]
    sizes = ["s", "b", "l"] + (["h"] if with_h else [])
    g += [f"depthpro2-{s}:{mode}" for s in sizes for mode in ("frame", "stream4", "stream16")]
    return g


QUICK = ["moge2-b:600", "moge:1200", "moge3-l:1200:refine3", "depthpro2-s:frame", "depthpro2-s:stream4"]


def parse_spec(spec):
    """'moge2-b:600', 'moge3-l:1200:refine3', 'depthpro2-s:stream4' -> load_model kwargs."""
    p = spec.split(":")
    name, kw = p[0], {}
    if name.startswith("depthpro2"):
        mode = p[1] if len(p) > 1 else "stream4"
        kw["stream"] = mode != "frame"
        kw["window"] = int(mode[6:]) if mode.startswith("stream") else 4
    else:
        kw["num_tokens"] = int(p[1]) if len(p) > 1 else 1200
        if len(p) > 2 and p[2].startswith("refine"):
            kw["refine"] = int(p[2][6:])
    return name, kw


# ---------------------------------------------------------------- data

def load_recordings(a, log):
    recs = []
    if a.bags:
        from bag_to_frames import read_bag
        os.makedirs(a.data, exist_ok=True)
        for bag in a.bags:
            bag = os.path.abspath(os.path.expanduser(bag))
            name = os.path.basename(bag.rstrip("/")).replace(".bag", "")
            cache = os.path.join(a.data, f"{name}_n{a.n}_s{a.step}.npz")
            if not os.path.exists(cache):
                log(f"converting {bag} ...")
                rgb, depth, fov, _, _ = read_bag(bag, a.n, a.step)
                np.savez(cache, rgb=rgb, depth=depth, fov_x=np.full(len(rgb), fov))
            recs.append((name, cache))
    for f in a.frames or []:
        recs.append((os.path.splitext(os.path.basename(f))[0], f))
    out = []
    for name, path in recs:
        d = np.load(path)
        n = min(a.n, len(d["depth"]))
        fov = float(d["fov_x"][0])
        out.append(dict(name=name, rgb=d["rgb"][:n], depth=d["depth"][:n].astype(np.float32),
                        fov=fov if math.isfinite(fov) else None))
        log(f"recording {name}: {n} frames, {d['rgb'].shape[2]}x{d['rgb'].shape[1]}, "
            f"holes {100 * (out[-1]['depth'] <= 0).mean():.1f} %, FoV {fov:.1f}")
    if not out:
        raise SystemExit("no recordings: use --bags or --frames")
    return out


# ---------------------------------------------------------------- metrics

def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def holdout(depth, pred, dev, hold_px, seed):
    """Hide blocks (and, for 32 px, the edge band), complete with a fresh fusion, score."""
    H, W = depth.shape
    valid = depth > 0
    rng = np.random.default_rng(seed)
    pick = rng.random((-(-H // hold_px), -(-W // hold_px))) < 0.15
    blocks = np.kron(pick, np.ones((hold_px, hold_px), bool))[:H, :W] & valid
    p_t = torch.from_numpy(pred).to(dev)
    out, _ = fill(torch.from_numpy(np.where(blocks, 0, depth).astype(np.float32)).to(dev), p_t, GradientFusion())
    r = B.score2(out.cpu().numpy(), depth, blocks, "hold")
    if hold_px == 32:
        e = B.edge_mask_np(depth, B.MEAS_EDGE_RTOL)
        band = B.dilate(e, B.EDGE_BAND_PX) & valid & ~e
        out2, _ = fill(torch.from_numpy(np.where(band, 0, depth).astype(np.float32)).to(dev), p_t, GradientFusion())
        r.update(B.score2(out2.cpu().numpy(), depth, band, "edge"))
    return r


def evaluate_recording(rec, preds, dev):
    depth = rec["depth"]
    n = len(depth)
    acc = {}
    for hold_px in (32, 128):
        rows = [holdout(depth[i], preds[i], dev, hold_px, seed=k) for k, i in enumerate(range(max(0, n - 2), n))]
        for key in rows[0]:
            acc[f"{key}_{hold_px}"] = float(np.nanmean([r[key] for r in rows]))
    # stability: causal sequence with smoothed scale and warm start, as in the node
    fusion, log_s, prev = GradientFusion(), None, None
    fl = {k: [] for k in ("filled", "filled_near", "model", "sensor", "cov")}
    fill_ms = []
    for i in range(n):
        d_t, p_t = torch.from_numpy(depth[i]).to(dev), torch.from_numpy(preds[i]).to(dev)
        sync(dev); t0 = time.perf_counter()
        out, log_s = fill(d_t, p_t, fusion, log_s)
        sync(dev); fill_ms.append(1000 * (time.perf_counter() - t0))
        out = out.cpu().numpy()
        model = np.where(np.isfinite(preds[i]), preds[i] * math.exp(log_s), 0)
        hole = depth[i] <= 0
        fl["cov"].append(float((out[hole] > 0).mean()) if hole.any() else np.nan)
        if prev is not None:
            p_depth, p_out, p_model = prev
            m = hole & (p_depth <= 0) & (out > 0) & (p_out > 0)
            if m.any():
                rel = np.abs(out - p_out)[m] / p_out[m]
                fl["filled"].append(float(rel.mean()))
                near = p_out[m] < NEAR
                fl["filled_near"].append(float(rel[near].mean()) if near.any() else np.nan)
                mm = m & (model > 0) & (p_model > 0)
                fl["model"].append(float((np.abs(model - p_model)[mm] / p_model[mm]).mean()) if mm.any() else np.nan)
            ms_ = (depth[i] > 0) & (p_depth > 0)
            fl["sensor"].append(float((np.abs(depth[i] - p_depth)[ms_] / p_depth[ms_]).mean()))
        prev = (depth[i], out, model)
    acc.update({f"flicker_{k}" if k != "cov" else "real_holes_filled": float(np.nanmean(v)) if v else None
                for k, v in fl.items()})
    acc["fill_ms"] = float(np.median(fill_ms[WARMUP:] if n > WARMUP else fill_ms))
    return acc


def run_config(spec, recs, dev, log):
    name, kw = parse_spec(spec)
    res = {"config": spec, "model": name, **{k: v for k, v in kw.items()}, "ok": False}
    model = None
    try:
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model = load_model(name, device=str(dev), **kw)
        res["load_s"] = round(time.perf_counter() - t0, 1)
        per, model_ms = {}, []
        for rec in recs:
            if hasattr(model, "stream"):
                model.stream = None                      # new stream per recording
            preds, times = [], []
            for rgb in rec["rgb"]:
                sync(dev); t = time.perf_counter()
                p = model(rgb, rec["fov"])
                sync(dev); times.append(1000 * (time.perf_counter() - t))
                preds.append(p.float().cpu().numpy())
            model_ms += times[WARMUP:]
            per[rec["name"]] = evaluate_recording(rec, preds, dev)
            per[rec["name"]]["model_ms"] = float(np.median(times[WARMUP:]))
            log(f"  {rec['name']}: model {per[rec['name']]['model_ms']:.0f} ms, "
                f"128px <3m {100 * per[rec['name']]['hold_near_absrel_128']:.2f} %, "
                f"flicker {100 * (per[rec['name']]['flicker_filled'] or 0):.2f} %")
        keys = next(iter(per.values())).keys()
        res.update({k: float(np.nanmean([p[k] for p in per.values() if p[k] is not None])) for k in keys})
        res["model_ms"] = float(np.median(model_ms))
        res["total_ms"] = res["model_ms"] + res["fill_ms"]
        res["hz"] = 1000.0 / res["total_ms"]
        if dev.type == "cuda":
            res["gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        res["per_recording"] = per
        res["ok"] = True
    except Exception as e:                                  # noqa: BLE001 - log and continue
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
        log(f"  FAILED: {res['error']}")
    finally:
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------- report

def environment(dev):
    env = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
           "platform": platform.platform(), "python": sys.version.split()[0], "torch": torch.__version__,
           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "device": str(dev)}
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        env.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 2 ** 30, 1))
        try:
            env["nvidia_smi"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm",
                 "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:                                   # noqa: BLE001
            pass
    for pkg in ("numpy", "cv2", "moge", "depth_pro_2", "flex_gemm", "utils3d_moge"):
        try:
            m = __import__(pkg)
            env[f"pkg_{pkg}"] = getattr(m, "__version__", "installed")
        except Exception:                                   # noqa: BLE001
            env[f"pkg_{pkg}"] = None
    try:
        env["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                           capture_output=True, text=True).stdout.strip()
    except Exception:                                       # noqa: BLE001
        pass
    return env


COLS = [("config", "config", "{}"), ("load_s", "load s", "{:.0f}"), ("model_ms", "model ms", "{:.0f}"),
        ("fill_ms", "fill ms", "{:.0f}"), ("total_ms", "total ms", "{:.0f}"), ("hz", "Hz", "{:.1f}"),
        ("gpu_mem_mb", "GPU MB", "{:.0f}"),
        ("hold_near_absrel_32", "32px AbsRel<3m %", "{:.2%}"), ("hold_near_cm_32", "32px MAE cm", "{:.1f}"),
        ("hold_near_cov_32", "32px cov", "{:.0%}"),
        ("hold_near_absrel_128", "128px AbsRel<3m %", "{:.2%}"), ("hold_near_cm_128", "128px MAE cm", "{:.1f}"),
        ("hold_near_cov_128", "128px cov", "{:.0%}"), ("hold_near_bleed_128", "128px gross", "{:.2%}"),
        ("hold_absrel_128", "128px AbsRel all %", "{:.2%}"),
        ("edge_bleed_32", "edge bleed", "{:.2%}"), ("edge_cov_32", "edge cov", "{:.0%}"),
        ("flicker_filled", "flicker", "{:.2%}"), ("flicker_filled_near", "flicker<3m", "{:.2%}"),
        ("flicker_model", "flicker model", "{:.2%}"), ("real_holes_filled", "real holes", "{:.0%}")]


def write_report(out, env, results, recs):
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in COLS] + ["error"])
        for r in results:
            w.writerow([r.get(c[0]) for c in COLS] + [r.get("error", "")])
    fmt = lambda r, k, f: (f.format(r[k]) if r.get(k) is not None else "–")
    lines = [f"# Depth completion suite: {env.get('gpu', env['device'])}", "",
             f"{env['time']} · host {env['host']} · torch {env['torch']} · CUDA {env['cuda']} · "
             f"commit {env.get('git_commit')}", "",
             "Recordings: " + ", ".join(f"{r['name']} ({len(r['depth'])} frames)" for r in recs), "",
             "| " + " | ".join(c[1] for c in COLS) + " |", "|" + "---|" * len(COLS)]
    for r in results:
        if r["ok"]:
            lines.append("| " + " | ".join(fmt(r, k, f) if k != "config" else r["config"] for k, _, f in COLS) + " |")
    failed = [r for r in results if not r["ok"]]
    if failed:
        lines += ["", "## Failed", ""] + [f"- `{r['config']}`: {r['error']}" for r in failed]
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bags", nargs="*", help="ROS 1/2 bag folders or files (converted once into --data)")
    ap.add_argument("--frames", nargs="*", help="frames.npz files (bag_to_frames.py)")
    ap.add_argument("--n", type=int, default=30, help="frames per recording")
    ap.add_argument("--step", type=int, default=1, help="use every k-th frame of a bag (1 = camera rate)")
    ap.add_argument("--data", default=os.path.expanduser("~/depth_suite_data"), help="cache of converted bags")
    ap.add_argument("--out", default=None, help="report folder (default: reports/<host>_<time>)")
    ap.add_argument("--only", nargs="*", help="configurations, e.g. moge2-b:600 moge3-l:1200:refine3 depthpro2-s:stream4")
    ap.add_argument("--quick", action="store_true", help=f"small grid: {' '.join(QUICK)}")
    ap.add_argument("--with-h", action="store_true", help="include Depth Pro 2 H (3.5 GB)")
    ap.add_argument("--resume", help="report folder of an interrupted run: skip finished configurations")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--list", action="store_true", help="print the configurations and exit")
    a = ap.parse_args()

    grid = a.only or (QUICK if a.quick else default_grid(a.with_h))
    if a.list:
        print("\n".join(grid))
        return
    out = a.resume or a.out or os.path.join(HERE, "reports",
                                            f"{socket.gethostname()}_{datetime.datetime.now():%Y%m%d_%H%M}")
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "suite.log"), "a")

    def log(msg):
        line = f"{datetime.datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    torch.set_num_threads(4)
    dev = torch.device(a.device)
    env = environment(dev)
    env.update(n_frames=a.n, step=a.step, grid=grid)
    with open(os.path.join(out, "env.json"), "w") as f:
        json.dump(env, f, indent=1)
    log(f"device {dev} {env.get('gpu', '')}, {len(grid)} configurations -> {out}")
    recs = load_recordings(a, log)

    res_path = os.path.join(out, "results.jsonl")
    done = {}
    if os.path.exists(res_path):
        for line in open(res_path):
            r = json.loads(line)
            done[r["config"]] = r
    results = []
    for k, spec in enumerate(grid):
        if spec in done and done[spec]["ok"]:
            log(f"[{k + 1}/{len(grid)}] {spec}: done before, skipped")
            results.append(done[spec])
            continue
        log(f"[{k + 1}/{len(grid)}] {spec}")
        r = run_config(spec, recs, dev, log)
        if r["ok"]:
            log(f"  -> {r['total_ms']:.0f} ms ({r['hz']:.1f} Hz), 128px <3m {100 * r['hold_near_absrel_128']:.2f} %, "
                f"gross {100 * r['hold_near_bleed_128']:.2f} %, flicker {100 * (r['flicker_filled'] or 0):.2f} %")
        with open(res_path, "a") as f:
            f.write(json.dumps(r) + "\n")
        results.append(r)
        write_report(out, env, results, recs)
    log(f"finished -> {out}/report.md")


if __name__ == "__main__":
    main()
