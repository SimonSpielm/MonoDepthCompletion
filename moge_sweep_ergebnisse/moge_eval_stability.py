#!/usr/bin/env python3
"""
Zusatz-Auswertung für das Paper: große Löcher, einseitige Löcher, zeitliche Stabilität.

Warum: Der Holdout aus moge_sweep*.py blendet nur Blöcke <= 64 px aus, die ringsum von
Messungen umgeben sind. Das bevorzugt die lokale affine Angleichung (Variante A). Dieses
Skript misst die Fälle, in denen A laut Beobachtung schwächelt:

  1) large   : Holdout mit großen Blöcken (Standard 96/128/192/256 px)
  2) grow    : einseitige Löcher. Echte Sensorlöcher werden um w px "vergrößert", d. h. die
               gemessenen Pixel im Ring der Breite w um echte Löcher werden ausgeblendet.
               Auf der einen Seite liegt das echte Loch, auf der anderen Messungen.
  3) dist    : bei 1) und 2) zusätzlich AbsRel je Abstand zur nächsten (Trainings-)Messung
               [px], sodass sichtbar wird, wo A auf den globalen Faktor zurückfällt
  4) temporal: zeitliche Stabilität über die Frame-Folge (statische Szene vorausgesetzt).
               Pro Pixel std/mean der gefüllten Tiefe über die Zeit in
                 - real : Pixel, die in ALLEN Frames ein echtes Loch sind
                 - ho   : ein FESTES großes Holdout-Muster (gleich in allen Frames)
               plus mittlere relative Frame-zu-Frame-Änderung und Flicker-Anteil
               (|Δ|/d > --flicker-thr). Referenz: dieselben Größen auf immer gültigen
               Messpixeln (Sensorrauschen) und für Stage 1 (MoGe-Eigenrauschen).
               Läuft sequenziell wie in Produktion: EMA im globalen Faktor (0.3) und
               Warmstart der Fusion (B) zwischen Frames, zusätzlich B ohne Warmstart.

Methoden (tuned Defaults wie in moge_depth.py / moge_depth_optim.py):
  global : nur Stage 1
  A      : local_affine(r=16, eps=1e-3, down=2, clip 0.25..4, min_frac 0.05)
  B      : GradientFusion(down=4, lam=1, mu_near=1e-4, mu_far=5e-3, far_band=0.15,
                          sigma_edge=0.02, z_ref=4, outlier=1.0)

Start (gleicher Cache wie die Sweeps, keine Inferenz nötig):
  python moge_eval_stability.py --preds moge_sweep_out/preds.npz --out-dir moge_eval_out
  python moge_eval_stability.py --preds ... --only temporal --plots
Ausgabe: CSVs, summary.md, optional PNGs (--plots, braucht matplotlib).
"""
import os
import sys

sys.path = [p for p in sys.path if "conan_cache" not in p]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv
import time

import cv2
import numpy as np

from moge_sweep import local_affine, stage1_scale          # noqa: E402  (1:1 wie Sweep A)
from moge_sweep_optim import GradientFusion                # noqa: E402  (1:1 wie Sweep B)

METHODS = ("global", "A", "B")
CFG_A = dict(radius_px=16, eps=0.001, min_frac=0.05, down=2, clip_lo=0.25, clip_hi=4.0,
             conf_mode="base", eps_mode="abs")
CFG_B = dict(down=4, lam=1.0, mu_near=1e-4, mu_far=0.005, far_band=0.15, sigma_edge=0.02,
             z_ref=4.0, outlier=1.0, iters=80, tol=1e-4)
DIST_BINS = [0, 8, 16, 32, 64, 128, 256, 10_000]   # Abstand zur nächsten Messung [px]


# ---------------------------------------------------------------- Füllen

def fill(method, d_train, pred, prev_log_s, ema, fusion=None):
    """Gibt (gefüllt, log_s) zurück. Gemessene Pixel bleiben unverändert."""
    pvalid = np.isfinite(pred) & (pred > 0)
    est, log_s = stage1_scale(d_train, pred, pvalid, ema, prev_log_s)
    if method == "A":
        est = np.where(pvalid, local_affine(d_train, est, **CFG_A), 0.0).astype(np.float32)
    elif method == "B":
        est = fusion(d_train, est)
    return np.where(d_train > 0, d_train, est).astype(np.float32), log_s


def new_fusion():
    return GradientFusion(**CFG_B)


# ---------------------------------------------------------------- Holdout-Muster

def block_holdout(valid, rng, block, frac):
    H, W = valid.shape
    pick = rng.random((-(-H // block), -(-W // block))) < frac
    return np.kron(pick, np.ones((block, block), bool))[:H, :W] & valid


def grow_holdout(valid, width, min_area=400):
    """Gültige Pixel im Abstand <= width zu einem großen echten Loch (einseitiges Loch).
    Nur zusammenhängende Löcher >= min_area px zählen; verstreute Einzelpixel-Löcher
    (bei der Kinect überall) würden sonst fast das ganze Bild ausblenden.
    Der Bildrand zählt nicht als Loch."""
    hole = (~valid).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hole, connectivity=8)
    big = np.zeros(n, bool)
    big[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    big_hole = big[lab]
    if not big_hole.any():
        return np.zeros_like(valid)
    dist = cv2.distanceTransform((~big_hole).astype(np.uint8), cv2.DIST_L2, 5)
    return valid & (dist <= width)


def dist_to_measurement(d_train):
    return cv2.distanceTransform((d_train <= 0).astype(np.uint8), cv2.DIST_L2, 5)


# ---------------------------------------------------------------- Metriken

def errors(filled, gt, mask, dist):
    sel = mask & (filled > 0)
    if int(sel.sum()) < 50:
        return None
    p, g, dd = filled[sel], gt[sel], dist[sel]
    rel = np.abs(p - g) / g
    out = {"absrel": float(rel.mean()), "mae_cm": 100 * float(np.abs(p - g).mean()),
           "delta1": float(np.mean(np.maximum(p / g, g / p) < 1.25)),
           "coverage": float(sel.sum()) / max(int(mask.sum()), 1), "n_px": int(sel.sum())}
    sums = []
    for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:]):
        s = (dd >= lo) & (dd < hi)
        sums.append((float(rel[s].sum()), int(s.sum())))
    out["dist_sums"] = sums
    return out


def agg(rows):
    """Pixelgewichtete AbsRel je Distanzklasse + Mittel der Frame-Metriken."""
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in ("absrel", "mae_cm", "delta1",
                                                                  "coverage")}
    nb = len(DIST_BINS) - 1
    s = np.zeros(nb)
    n = np.zeros(nb)
    for r in rows:
        for j, (a, b) in enumerate(r["dist_sums"]):
            s[j] += a
            n[j] += b
    out["dist_absrel"] = [float(s[j] / n[j]) if n[j] >= 200 else float("nan") for j in range(nb)]
    out["dist_n"] = n.astype(int).tolist()
    return out


# ---------------------------------------------------------------- Experimente 1-3

def run_spatial(pred_all, depth_all, a):
    """large + grow. Jeder Frame unabhängig (kein EMA, Fusion ohne Warmstart),
    wie in den Sweeps, damit die Zahlen vergleichbar sind."""
    cases = [("large", b) for b in a.large_blocks] + [("grow", w) for w in a.grow_widths]
    results = []
    n = len(pred_all)
    for ci, (kind, size) in enumerate(cases):
        for rep in range(a.reps if kind == "large" else 1):
            per = {m: [] for m in METHODS}
            t0 = time.perf_counter()
            for i in range(n):
                gt = depth_all[i]
                valid = gt > 0
                if kind == "large":
                    rng = np.random.default_rng((a.seed, 100 + ci, rep, i))
                    ho = block_holdout(valid, rng, size, a.holdout)
                else:
                    ho = grow_holdout(valid, size, a.grow_min_area)
                d_train = np.where(ho, 0, gt).astype(np.float32)
                dist = dist_to_measurement(d_train)
                for m in METHODS:
                    f, _ = fill(m, d_train, pred_all[i], None, 0.0,
                                new_fusion() if m == "B" else None)
                    e = errors(f, gt, ho, dist)
                    if e:
                        per[m].append(e)
            for m in METHODS:
                if per[m]:
                    r = agg(per[m])
                    r.update(kind=kind, size=size, rep=rep, method=m, frames=len(per[m]))
                    results.append(r)
            print(f"  {kind:5s} {size:4d}px rep {rep}: "
                  + "  ".join(f"{m} {100 * agg(per[m])['absrel']:.2f}%" for m in METHODS
                              if per[m])
                  + f"   ({time.perf_counter() - t0:.0f}s)", flush=True)
    return results


def collapse_reps(results):
    """Mittelt über reps (gleiche kind/size/method)."""
    out = {}
    for r in results:
        k = (r["kind"], r["size"], r["method"])
        out.setdefault(k, []).append(r)
    rows = []
    for (kind, size, m), rs in out.items():
        d = dict(kind=kind, size=size, method=m, frames=sum(x["frames"] for x in rs))
        for key in ("absrel", "mae_cm", "delta1", "coverage"):
            d[key] = float(np.mean([x[key] for x in rs]))
        d["dist_absrel"] = list(np.nanmean(np.array([x["dist_absrel"] for x in rs]), axis=0))
        d["dist_n"] = list(np.sum(np.array([x["dist_n"] for x in rs]), axis=0))
        rows.append(d)
    return rows


# ---------------------------------------------------------------- Experiment 4

def temporal_stats(stack, region):
    """stack: (T,H,W) gefüllte Tiefe. region: (H,W) bool.
    Nur Pixel, die in allen Frames > 0 sind."""
    ok = region & np.all(stack > 0, axis=0)
    if int(ok.sum()) < 50:
        return None
    v = stack[:, ok]                                     # (T, N)
    mean = v.mean(axis=0)
    cv = v.std(axis=0) / mean                            # zeitl. Variationskoeffizient
    dv = np.abs(np.diff(v, axis=0)) / mean[None]         # rel. Frame-zu-Frame-Änderung
    return {"tstd_rel": float(np.mean(cv)), "tstd_rel_p95": float(np.percentile(cv, 95)),
            "dframe_rel": float(np.mean(dv)), "n_px": int(ok.sum()), "_dv": dv}


def run_temporal(pred_all, depth_all, a):
    n = len(pred_all)
    H, W = depth_all.shape[1:]
    valid_all = depth_all > 0
    real_hole = np.all(~valid_all, axis=0)                 # in allen Frames Loch
    always_valid = np.all(valid_all, axis=0)
    rng = np.random.default_rng((a.seed, 999))
    pick = rng.random((-(-H // a.temporal_block), -(-W // a.temporal_block))) < a.holdout
    ho_fixed = np.kron(pick, np.ones((a.temporal_block,) * 2, bool))[:H, :W]

    variants = [("global", None), ("A", None), ("B", "warm"), ("B", "cold")]
    stacks = {}
    for scen in ("real", "ho"):
        for m, mode in variants:
            name = m if mode is None else f"B_{mode}"
            prev, fusion = None, new_fusion()
            out = []
            for i in range(n):
                d = depth_all[i] if scen == "real" else np.where(ho_fixed, 0, depth_all[i])
                fu = fusion if (m == "B" and mode == "warm") else (new_fusion() if m == "B" else None)
                f, prev = fill(m, d.astype(np.float32), pred_all[i], prev, a.ema, fu)
                out.append(f)
            stacks[(scen, name)] = np.stack(out)
            print(f"  temporal {scen:4s} {name:8s} fertig", flush=True)

    rows = []
    sensor = temporal_stats(depth_all, always_valid)
    if sensor:
        rows.append(dict(scenario="sensor (always-valid)", method="measured",
                         **{k: v for k, v in sensor.items() if not k.startswith("_")},
                         flicker=float(np.mean(sensor["_dv"] > a.flicker_thr))))
    for (scen, name), st in stacks.items():
        region = real_hole if scen == "real" else (ho_fixed & np.all(valid_all, axis=0))
        s = temporal_stats(st, region)
        if s is None:
            continue
        row = dict(scenario=scen, method=name,
                   **{k: v for k, v in s.items() if not k.startswith("_")},
                   flicker=float(np.mean(s["_dv"] > a.flicker_thr)))
        if scen == "ho":   # zusätzlich Genauigkeit im festen Holdout über die Zeit
            sel = region[None] & (st > 0)
            g = np.broadcast_to(depth_all, st.shape)[sel]
            row["absrel"] = float(np.mean(np.abs(st[sel] - g) / g))
        rows.append(row)
    return rows, stacks, real_hole, ho_fixed


# ---------------------------------------------------------------- Ausgabe

def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (";".join(f"{x:.5g}" for x in r[k]) if isinstance(r.get(k), list)
                            else r.get(k)) for k in fields})


def fmt(v, pct=True):
    if v is None or not np.isfinite(v):
        return "–"
    return f"{100 * v:.2f}" if pct else f"{v:.2f}"


def summary_md(spatial, temporal, a, n_frames):
    L = [f"# MoGe-Stabilitäts-Auswertung\n", f"Frames: {n_frames} | Preds: `{a.preds}`\n"]
    if spatial:
        L.append("## Große und einseitige Löcher (AbsRel %, MAE cm)\n")
        L.append("| Fall | Größe [px] | global | A | B | MAE A | MAE B |")
        L.append("|---|---|---|---|---|---|---|")
        keys = sorted({(r["kind"], r["size"]) for r in spatial})
        get = {(r["kind"], r["size"], r["method"]): r for r in spatial}
        for kind, size in keys:
            g, A, B = (get.get((kind, size, m)) for m in METHODS)
            L.append(f"| {kind} | {size} | {fmt(g and g['absrel'])} | {fmt(A and A['absrel'])} | "
                     f"{fmt(B and B['absrel'])} | {fmt(A and A['mae_cm'], False)} | "
                     f"{fmt(B and B['mae_cm'], False)} |")
        L.append("\n## AbsRel % je Abstand zur nächsten Messung (alle Fälle gepoolt)\n")
        hdr = [f"{lo}–{hi}" if hi < 10_000 else f">{lo}" for lo, hi in
               zip(DIST_BINS[:-1], DIST_BINS[1:])]
        L.append("| Methode | " + " | ".join(hdr) + " |")
        L.append("|---|" + "---|" * len(hdr))
        for m in METHODS:
            rs = [r for r in spatial if r["method"] == m]
            n = np.sum([np.array(r["dist_n"]) for r in rs], axis=0)
            s = np.sum([np.nan_to_num(np.array(r["dist_absrel"])) * np.array(r["dist_n"])
                        for r in rs], axis=0)
            vals = [s[j] / n[j] if n[j] >= 200 else float("nan") for j in range(len(n))]
            L.append(f"| {m} | " + " | ".join(fmt(v) for v in vals) + " |")
    if temporal:
        L.append("\n## Zeitliche Stabilität (statische Szene vorausgesetzt)\n")
        L.append(f"tstd = mittlerer zeitlicher Variationskoeffizient std/mean pro Pixel, "
                 f"dframe = mittlere relative Frame-zu-Frame-Änderung, "
                 f"flicker = Anteil Frame-Übergänge mit |Δ|/d > {100 * a.flicker_thr:.0f} %.\n")
        L.append("| Region | Methode | tstd % | tstd p95 % | dframe % | flicker % | AbsRel % | Pixel |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in temporal:
            L.append(f"| {r['scenario']} | {r['method']} | {fmt(r['tstd_rel'])} | "
                     f"{fmt(r['tstd_rel_p95'])} | {fmt(r['dframe_rel'])} | {fmt(r['flicker'])} | "
                     f"{fmt(r.get('absrel', float('nan')))} | {r['n_px']} |")
    L.append("\n## Methoden-Parameter\n")
    L.append(f"- A: `{CFG_A}`\n- B: `{CFG_B}`\n- Temporal: EMA {a.ema}, Holdout-Block "
             f"{a.temporal_block}px, B warm = Warmstart wie in Produktion")
    return "\n".join(L) + "\n"


def make_plots(spatial, stacks, real_hole, ho_fixed, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"global": "#888888", "A": "#e07b39", "B": "#3a78c2", "B_warm": "#3a78c2",
            "B_cold": "#7fb2e5"}

    if spatial:   # Fehler über Abstand zur Messung
        fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=200)
        mids = [0.5 * (lo + min(hi, 512)) for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:])]
        for m in METHODS:
            rs = [r for r in spatial if r["method"] == m]
            n = np.sum([np.array(r["dist_n"]) for r in rs], axis=0)
            s = np.sum([np.nan_to_num(np.array(r["dist_absrel"])) * np.array(r["dist_n"])
                        for r in rs], axis=0)
            v = np.where(n >= 200, s / np.maximum(n, 1), np.nan)
            ax.plot(mids, 100 * v, "o-", ms=3, lw=1.5, color=cols[m], label=m)
        ax.set_xscale("log")
        ax.set_xlabel("distance to nearest measurement [px]")
        ax.set_ylabel("AbsRel [%]")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "err_vs_distance.png"))
        plt.close(fig)

    if stacks:    # x-t-Schnitt: Zeile mit den meisten Lochpixeln
        for scen, region in (("real", real_hole), ("ho", ho_fixed)):
            names = [k[1] for k in stacks if k[0] == scen and k[1] in ("global", "A", "B_warm")]
            if not names:
                continue
            cnt = region.sum(axis=1)
            rows_max = np.flatnonzero(cnt == cnt.max())
            row = int(rows_max[len(rows_max) // 2])        # mittlere Zeile, nicht der Rand
            fig, axes = plt.subplots(len(names), 1, figsize=(3.5, 1.1 * len(names) + 0.4),
                                     dpi=200, sharex=True)
            axes = np.atleast_1d(axes)
            ref = stacks[(scen, names[-1])][:, row, :][:, region[row]]   # Farbskala: Lochregion
            vmin, vmax = np.percentile(ref[ref > 0], [1, 99]) if (ref > 0).any() else (0, 5)
            for ax, nm in zip(axes, names):
                img = stacks[(scen, nm)][:, row, :].astype(float)
                img[img <= 0] = np.nan
                ax.imshow(img, aspect="auto", cmap="turbo", vmin=vmin, vmax=vmax,
                          interpolation="nearest")
                ax.contour(np.broadcast_to(region[row], img.shape), levels=[0.5],
                           colors="w", linewidths=0.4)
                ax.set_ylabel(nm, fontsize=7)
                ax.set_yticks([])
            axes[-1].set_xlabel(f"column (image row {row})")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"xt_slice_{scen}.png"))
            plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--preds", required=True, help="npz aus moge_sweep.py infer")
    ap.add_argument("--out-dir", default="moge_eval_out")
    ap.add_argument("--only", choices=("all", "spatial", "temporal"), default="all")
    ap.add_argument("--limit", type=int, default=0, help="nur die ersten N Frames")
    ap.add_argument("--large-blocks", default="96,128,192,256")
    ap.add_argument("--grow-widths", default="16,32,64,128")
    ap.add_argument("--grow-min-area", type=int, default=400,
                    help="Mindestfläche [px] echter Löcher, die bei 'grow' vergrößert werden")
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temporal-block", type=int, default=128)
    ap.add_argument("--ema", type=float, default=0.3, help="EMA des globalen Faktors (temporal)")
    ap.add_argument("--flicker-thr", type=float, default=0.02)
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()
    a.large_blocks = [int(v) for v in a.large_blocks.split(",") if v]
    a.grow_widths = [int(v) for v in a.grow_widths.split(",") if v]

    data = np.load(a.preds)
    pred_all = data["pred"].astype(np.float32)
    depth_all = data["depth"].astype(np.float32)
    if a.limit:
        pred_all, depth_all = pred_all[: a.limit], depth_all[: a.limit]
    if pred_all.shape[1:] != depth_all.shape[1:]:
        H, W = depth_all.shape[1:]
        pred_all = np.stack([cv2.resize(p, (W, H), interpolation=cv2.INTER_NEAREST)
                             for p in pred_all])
    n = len(pred_all)
    print(f"Frames: {n} | {depth_all.shape[2]}x{depth_all.shape[1]} | "
          f"Löcher {100 * np.mean(depth_all <= 0):.0f} %")
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.perf_counter()

    spatial, temporal, stacks, real_hole, ho_fixed = [], [], {}, None, None
    if a.only in ("all", "spatial"):
        print("Räumlich (large / grow) ...")
        spatial = collapse_reps(run_spatial(pred_all, depth_all, a))
        write_csv(os.path.join(a.out_dir, "spatial.csv"), spatial,
                  ["kind", "size", "method", "frames", "absrel", "mae_cm", "delta1", "coverage",
                   "dist_absrel", "dist_n"])
    if a.only in ("all", "temporal"):
        print("Zeitlich ...")
        temporal, stacks, real_hole, ho_fixed = run_temporal(pred_all, depth_all, a)
        write_csv(os.path.join(a.out_dir, "temporal.csv"), temporal,
                  ["scenario", "method", "tstd_rel", "tstd_rel_p95", "dframe_rel", "flicker",
                   "absrel", "n_px"])

    md = summary_md(spatial, temporal, a, n)
    with open(os.path.join(a.out_dir, "summary.md"), "w") as f:
        f.write(md)
    if a.plots:
        try:
            make_plots(spatial, stacks, real_hole, ho_fixed, a.out_dir)
        except ImportError:
            print("matplotlib fehlt -> keine Plots")
    print("\n" + md)
    print(f"fertig in {time.perf_counter() - t0:.0f}s -> {a.out_dir}")


if __name__ == "__main__":
    main()
