#!/usr/bin/env python3
"""
Parameter-Sweep fuer moge_depth_optim.py (Gradienten-Fusion / Korrekturfeld c).

Nutzt den bestehenden MoGe-Predictions-Cache von moge_sweep.py (gleiche Modell,
gleiche num_tokens=1200, fp16, camera_info) -> kein erneutes Inferieren noetig:

  1) moge_sweep.py capture + infer  (einmalig, ergibt moge_sweep_out/preds.npz)
  2) dieses Skript:
     python moge_sweep_optim.py sweep --preds moge_sweep_out/preds.npz \
         --out-dir moge_sweep_optim_out --down 4,8,16 --lam 1,4,16 \
         --sigma-edge 0.02,0.05,0.1 --z-ref 1,2,4 --blocks 32 --reps 1

Auswertung wie bei moge_sweep.py: pro Frame zufaellige Blöcke GEMESSENER Pixel werden
(fix Seed) ausgeblendet, gefuellt und mit der echten Messung verglichen.
GradientFusion wird pro Frame frisch erzeugt (kein Warmstart) -> deterministisch,
Frame-Reihenfolge irrelevant. Baseline (nur globaler Faktor, ohne Fusion) mit
--baseline (Standard an).
"""
import os
import sys

sys.path = [p for p in sys.path if "conan_cache" not in p]
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import argparse
import csv
import itertools
import time
from collections import defaultdict

import cv2
import numpy as np

BINS = [0, 1, 2, 3, 4, 6, 100]

COLS = ["down", "lam", "mu_near", "mu_far", "far_band",
        "sigma_edge", "z_ref", "outlier", "iters", "tol", "ema"]


# ---------------------------------------------------------------- Gradienten-Fusion
# 1:1-Kopie aus moge_depth_optim.py

def _block_stats(z, mask, f):
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
    def __init__(self, down=8, lam=4.0, mu_near=1e-3, mu_far=0.05, far_band=0.15,
                 sigma_edge=0.05, z_ref=2.0, outlier=0.1, iters=80, tol=1e-4):
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

        wm = frac * np.minimum(1.0, np.exp(2 * (np.log(self.z_ref) - zlog)))
        wm *= np.exp(-(np.sqrt(gMvar) + np.sqrt(rvar)) / self.sigma_edge)
        wm[frac < 0.2] = 0
        has = wm > 0
        if has.any():
            wm *= 1.0 / (1.0 + ((r - np.median(r[has])) / self.outlier) ** 2)
            t = np.clip((gM - np.percentile(zlog[has], 98)) / self.far_band, 0, 1)
        else:
            t = np.ones_like(gM)
        mu = self.mu_near + (self.mu_far - self.mu_near) * t

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
        wm2 = wm * np.minimum(1.0, self.outlier / np.maximum(np.abs(r - x), 1e-9))
        x = self._cg(wm2 * r, wm2 + mu + deg, wx, wy, x)
        self._c_prev = x

        H2, W2 = H // 2, W // 2
        logM2 = cv2.resize(logM, (W2, H2), interpolation=cv2.INTER_NEAREST)
        c_half = self._guided_upsample(x.astype(np.float32), gM, logM2, H2, W2)
        out = est * np.exp(cv2.resize(c_half, (W, H), interpolation=cv2.INTER_LINEAR))
        out[~pv] = 0
        return out.astype(np.float32)

    def _guided_upsample(self, c, gM, logM, H, W):
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


# ---------------------------------------------------------------- Metrik / Aggregation

def holdout_mask(valid, rng, block=32, frac=0.15):
    H, W = valid.shape
    gh, gw = -(-H // block), -(-W // block)
    pick = rng.random((gh, gw)) < frac
    m = np.kron(pick, np.ones((block, block), bool))[:H, :W]
    return m & valid


def stage1_scale(depth_train, pred, pvalid, ema=0.0, prev_log_s=None):
    import math
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


def float_list(s):
    return [float(v) for v in str(s).split(",") if v.strip()]


def int_list(s):
    return [int(float(v)) for v in str(s).split(",") if v.strip()]


def build_configs(a):
    grids = {c: (int_list(getattr(a, c)) if c in ("down", "iters") else float_list(getattr(a, c)))
             for c in COLS if c != "ema"}
    grids["ema"] = float_list(a.ema)
    seen = {}
    for vals in itertools.product(*(grids[c] for c in COLS)):
        cfg = dict(zip(COLS, vals))
        seen[tuple(cfg[c] for c in COLS)] = cfg
    cfgs = list(seen.values())
    cfgs.sort(key=lambda c: (c["down"], c["lam"], c["sigma_edge"], c["z_ref"], c["ema"]))
    return cfgs


def cfg_str(r):
    return (f"down={r['down']:g} lam={r['lam']:g} s_edge={r['sigma_edge']:g} zref={r['z_ref']:g} "
            f"out={r['outlier']:g} mun={r['mu_near']:g} muf={r['mu_far']:g} "
            f"band={r['far_band']:g} it={r['iters']:g} tol={r['tol']:g} ema={r['ema']:g}")


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
    fields = (["block"] if has_block else []) + COLS + \
             ["absrel", "mae_cm", "rmse", "delta1", "coverage", "ms", "n", "bins"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in agg_rows:
            rr = dict(r)
            rr["bins"] = _fmt_bins(r["bins"])
            w.writerow(rr)


def print_summary(per_block, overall, base_agg, blocks, min_coverage, top):
    print("\n" + "=" * 88)
    print("Ergebnis je Holdout-Lochgroesse  (AbsRel = mittlerer relativer Fehler)")
    print("=" * 88)
    for block in blocks:
        b = [r for r in base_agg if r["block"] == block]
        if b and np.isfinite(b[0]["absrel"]):
            r = b[0]
            print(f"\nLoch {block:>3}px   global, OHNE Fusion: "
                  f"AbsRel {100 * r['absrel']:5.2f}%  MAE {r['mae_cm']:6.1f}cm  "
                  f"RMSE {r['rmse']:.2f}m  d1 {100 * r['delta1']:5.1f}%  cov {100 * r['coverage']:5.1f}%")
        rows_b = [r for r in per_block if r["block"] == block]
        cand = [r for r in rows_b if r["coverage"] >= min_coverage and np.isfinite(r["absrel"])]
        cand.sort(key=lambda r: r["absrel"])
        for j, r in enumerate(cand[:top]):
            print(f"  #{j + 1} {cfg_str(r)}")
            print(f"        -> AbsRel {100 * r['absrel']:5.2f}%  MAE {r['mae_cm']:6.1f}cm  "
                  f"RMSE {r['rmse']:.2f}m  d1 {100 * r['delta1']:5.1f}%  "
                  f"cov {100 * r['coverage']:5.1f}%  {r['ms']:5.1f}ms")

    print("\n" + "=" * 88)
    print(f"Bester Kompromiss ueber alle Lochgroessen  (Coverage >= {100 * min_coverage:.0f}%)")
    print("=" * 88)
    cand = [r for r in overall if r["coverage"] >= min_coverage and np.isfinite(r["absrel"])]
    cand.sort(key=lambda r: r["absrel"])
    if not cand:
        print("keine Konfiguration erfuellt die Coverage-Schwelle")
    for j, r in enumerate(cand[:max(top, 10)]):
        print(f"{j + 1:2d}. AbsRel {100 * r['absrel']:5.2f}%  MAE {r['mae_cm']:6.1f}cm  "
              f"d1 {100 * r['delta1']:5.1f}%  cov {100 * r['coverage']:5.1f}%  "
              f"{r['ms']:5.1f}ms | {cfg_str(r)}")


# ---------------------------------------------------------------- sweep

def cmd_sweep(a):
    if not a.preds:
        raise SystemExit("sweep: --preds fehlt (aus moge_sweep.py infer)")
    data = np.load(a.preds)
    if "pred" not in data or "depth" not in data:
        raise SystemExit(f"{a.preds}: braucht 'pred' und 'depth'")
    pred_all = data["pred"].astype(np.float32)
    depth_all = data["depth"].astype(np.float32)
    n = len(pred_all)
    if a.limit:
        n = min(n, int(a.limit))
    if n < 1:
        raise SystemExit("keine Frames")
    pred_all, depth_all = pred_all[:n], depth_all[:n]
    if pred_all.shape[1:] != depth_all.shape[1:]:
        H, W = depth_all.shape[1:]
        pred_all = np.stack([cv2.resize(p, (W, H), interpolation=cv2.INTER_NEAREST)
                             for p in pred_all])

    cfgs = build_configs(a)
    blocks = int_list(a.blocks)
    reps = max(int(a.reps), 1)
    total = n * len(blocks) * reps * (len(cfgs) + (1 if a.baseline else 0))
    print(f"Frames: {n} | Loecher: {blocks} | reps: {reps} | Konfigs: {len(cfgs)}"
          + (" + Baseline" if a.baseline else ""))
    print(f"Messungen gesamt: {total:,}".replace(",", "."))
    if len(cfgs) > a.max_combos and not a.force:
        raise SystemExit(f"{len(cfgs)} Konfigs > --max-combos {a.max_combos} (--force)")

    os.makedirs(a.out_dir, exist_ok=True)
    rows, base_rows = [], []
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

            est_cache = {}
            for ema in sorted({c["ema"] for c in cfgs}):
                prev = None
                arr = []
                for i in range(n):
                    d_train = samples[i][0]
                    pvalid = np.isfinite(pred_all[i]) & (pred_all[i] > 0)
                    est, prev = stage1_scale(d_train, pred_all[i], pvalid, ema, prev)
                    arr.append(est)
                est_cache[ema] = arr

            if a.baseline:
                ema0 = sorted({c["ema"] for c in cfgs})[0]
                for i in range(n):
                    d_train, ho = samples[i]
                    filled = np.where(d_train > 0, d_train, est_cache[ema0][i])
                    m = eval_sample(filled, depth_all[i], ho)
                    if m:
                        m = dict(m, **{c: (0.0 if c in ("down", "iters") else 0.0) for c in COLS})
                        m["down"] = -1
                        base_rows.append(dict(m, block=block, rep=rep, frame=i, ms=0.0))

            for cfg in cfgs:
                for i in range(n):
                    d_train, ho = samples[i]
                    gf = GradientFusion(down=cfg["down"], lam=cfg["lam"],
                                        mu_near=cfg["mu_near"], mu_far=cfg["mu_far"],
                                        far_band=cfg["far_band"], sigma_edge=cfg["sigma_edge"],
                                        z_ref=cfg["z_ref"], outlier=cfg["outlier"],
                                        iters=cfg["iters"], tol=cfg["tol"])
                    t = time.perf_counter()
                    out = gf(d_train, est_cache[cfg["ema"]][i])
                    ms = 1000.0 * (time.perf_counter() - t)
                    filled = np.where(d_train > 0, d_train, out)
                    m = eval_sample(filled, depth_all[i], ho)
                    if m is None:
                        continue
                    row = dict(cfg)
                    row.update(block=block, rep=rep, frame=i, ms=ms, **m)
                    rows.append(row)
                    done += 1
                    if done % step == 0:
                        el = time.perf_counter() - t_start
                        eta = el * (total - done) / max(done, 1)
                        print(f"  {100.0 * done / total:5.1f}%  {el:7.1f}s  ETA {eta:6.1f}s",
                              flush=True)

    if not rows:
        raise SystemExit("keine Messung erzeugt (zu wenig valide Tiefe?)")

    FIELDS = COLS + ["block", "rep", "frame",
                     "absrel", "mae_cm", "rmse", "delta1", "coverage", "ms", "bins"]
    with open(os.path.join(a.out_dir, "sweep_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            rr = {k: r.get(k) for k in FIELDS}
            rr["bins"] = _fmt_bins(r["bins"])
            w.writerow(rr)

    per_block = aggregate(rows, ["block"] + COLS)
    overall = aggregate(rows, COLS)
    base_agg = aggregate(base_rows, ["block"] + COLS) if base_rows else []
    _write_agg(os.path.join(a.out_dir, "sweep_summary_per_block.csv"), per_block)
    _write_agg(os.path.join(a.out_dir, "sweep_summary_overall.csv"), overall)
    if base_agg:
        _write_agg(os.path.join(a.out_dir, "sweep_baseline.csv"), base_agg)
    print(f"\nCSVs in: {a.out_dir}")

    print_summary(per_block, overall, base_agg, blocks, a.min_coverage, a.top)
    el = time.perf_counter() - t_start
    print(f"\nfertig in {el:.0f}s")


def main():
    ap = argparse.ArgumentParser(
        description="Parameter-Sweep fuer moge_depth_optim.py (Gradienten-Fusion)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--preds", required=True, help="npz aus moge_sweep.py infer")
    ap.add_argument("--out-dir", default="moge_sweep_optim_out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--blocks", default="32", help="Holdout-Lochgroessen [px]")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-coverage", type=float, default=0.95)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--baseline", action="store_true", default=True)
    ap.add_argument("--no-baseline", dest="baseline", action="store_false")
    ap.add_argument("--max-combos", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    # Parametergitter (Kommata)
    ap.add_argument("--down", default="8")
    ap.add_argument("--lam", default="4")
    ap.add_argument("--mu-near", default="0.001")
    ap.add_argument("--mu-far", default="0.05")
    ap.add_argument("--far-band", default="0.15")
    ap.add_argument("--sigma-edge", default="0.05")
    ap.add_argument("--z-ref", default="2")
    ap.add_argument("--outlier", default="0.1")
    ap.add_argument("--iters", default="80")
    ap.add_argument("--tol", default="0.0001")
    ap.add_argument("--ema", default="0")
    a = ap.parse_args()
    cmd_sweep(a)


if __name__ == "__main__":
    main()
