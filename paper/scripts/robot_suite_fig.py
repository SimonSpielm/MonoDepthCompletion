"""Accuracy, gross errors and flicker against latency for all models of the robot GPU suite.

Usage (from paper/): python scripts/robot_suite_fig.py <report dir> [figures/robot_tradeoff.pdf]
"""
import json, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

D = sys.argv[1]  # report folder of bench/run_suite.py
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures/robot_tradeoff.pdf"
RR = {r["config"]: r for r in map(json.loads, open(os.path.join(D, "results.jsonl")))}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
C = dict(S="#2a78d6", B="#eb6834", L="#1baf7a", LN="#eda100", BIG="#6250d6")
REC, CUR = "moge:600", "moge2-b:600"
T = ["600", "1200", "2400"]
V = ["frame", "stream4"]  # window 16 is in the report, not in the figure
# family -> list of (color, [(config, label)])
FAMS = [
    ("MoGe-2", [("S", [(f"moge2-s:{t}", f"S-{t}") for t in T]), ("B", [(f"moge2-b:{t}", f"B-{t}") for t in T]),
                ("L", [(f"moge:{t}", f"L-{t}") for t in T]), ("LN", [(f"moge2-l-normal:{t}", f"L-normal-{t}") for t in T])]),
    ("MoGe-3", [("L", [(f"moge3-l:{t}:refine3", f"L-{t}") for t in T]), ("BIG", [(f"moge3-g:{t}:refine3", f"G-{t}") for t in T[:2]])]),
    ("Depth Pro 2", [(k, [(f"depthpro2-{s.lower()}:{v}", f"{s}-{v}") for v in V]) for s, k in [("S", "S"), ("B", "B"), ("L", "L"), ("H", "BIG")]]),
]
METRICS = [("hold_near_absrel_128", "Error", "AbsRel [%]"), ("hold_near_bleed_128", "Gross errors", "> 10 % off [%]"),
           ("flicker_filled", "Flicker", "frame-to-frame [%]")]
SIZES = [16, 8, 3]

plt.rcParams.update({"font.size": 6, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF})

CANDS = [(7, 0, "left", "center"), (-7, 0, "right", "center"), (0, 8, "center", "bottom"), (0, -8, "center", "top"),
         (7, 7, "left", "bottom"), (-7, 7, "right", "bottom"), (7, -7, "left", "top"), (-7, -7, "right", "top"),
         (16, 0, "left", "center"), (-16, 0, "right", "center"), (0, 18, "center", "bottom"), (0, -18, "center", "top"),
         (14, 16, "left", "bottom"), (-14, 16, "right", "bottom"), (14, -16, "left", "top"), (-14, -16, "right", "top"),
         (0, 30, "center", "bottom"), (0, -30, "center", "top"), (26, 22, "left", "bottom"), (-26, 22, "right", "bottom"),
         (26, -22, "left", "top"), (-26, -22, "right", "top"), (34, 0, "left", "center"), (-34, 0, "right", "center")]
CANDS = [(round(dx * 0.45, 1), round(dy * 0.45, 1), ha, va) for dx, dy, ha, va in CANDS]


def place(fig, ax, pts, lines, rings, labels):
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    obst, labs = [], []
    for x, y in pts:
        px, py = ax.transData.transform((x, y)); obst.append(Bbox([[px - 2.5, py - 2.5], [px + 2.5, py + 2.5]]))
    for x, y in rings:
        px, py = ax.transData.transform((x, y)); obst.append(Bbox([[px - 5, py - 5], [px + 5, py + 5]]))
    for xs, ys in lines:
        P = ax.transData.transform(list(zip(xs, ys)))
        for (x0, y0), (x1, y1) in zip(P[:-1], P[1:]):
            for t in np.linspace(0, 1, 14):
                px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0); obst.append(Bbox([[px - 2, py - 2], [px + 2, py + 2]]))
    axbb = ax.get_window_extent()
    for x, y, text, col in labels:
        best = None
        for i, (dx, dy, ha, va) in enumerate(CANDS):
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=4.6)
            bb = t.get_window_extent(rend).expanded(1.03, 1.1); t.remove()
            inside = axbb.contains(bb.x0, bb.y0) and axbb.contains(bb.x1, bb.y1)
            score = sum(bb.overlaps(o) for o in obst) + 25 * sum(bb.overlaps(o) for o in labs) + (0 if inside else 1000) + i * 0.05
            if best is None or score < best[0]: best = (score, i, bb)
        _, i, bb = best; dx, dy, ha, va = CANDS[i]
        far = abs(dx) > 5 or abs(dy) > 5
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=4.6, color=INK,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.35, shrinkA=0, shrinkB=1.5) if far else None)
        obst.append(bb); labs.append(bb)


fig, axs = plt.subplots(3, 3, figsize=(7.16, 5.6), sharex=True)
for r, (key, title, unit) in enumerate(METRICS):
    top = max(100 * v[key] for v in RR.values()) * 1.1
    for c, (fam, models) in enumerate(FAMS):
        ax = axs[r, c]
        ax.set_xscale("log"); ax.set_xlim(18, 330); ax.set_ylim(0, top)
        ax.set_xticks([20, 30, 50, 100, 200, 300]); ax.set_xticklabels(["20", "30", "50", "100", "200", "300"]); ax.minorticks_off()
        ax.axvspan(125, 330, color="#f4f3ef", zorder=0); ax.axvline(125, color=MUTED, lw=0.5, ls="--", zorder=1)
        ax.text(129, 0.985, "8 Hz", transform=ax.get_xaxis_transform(), va="top", fontsize=5, color=MUTED)
        pts, lines, labels, rings = [], [], [], []
        for col, items in models:
            x = [RR[k]["total_ms"] for k, _ in items]; y = [100 * RR[k][key] for k, _ in items]
            if len(x) > 1: ax.plot(x, y, color=C[col], lw=0.8, alpha=0.85, zorder=2)
            ax.scatter(x, y, s=SIZES[:len(x)], color=C[col], edgecolors=SURF, linewidths=0.4, zorder=3)
            pts += list(zip(x, y)); lines.append((x, y))
            labels += [(xi, yi, lab, C[col]) for xi, yi, (_, lab) in zip(x, y, items)]
            for k, ls in [(REC, "-"), (CUR, "--")]:
                if k in [i[0] for i in items]:
                    xr, yr = RR[k]["total_ms"], 100 * RR[k][key]; rings.append((xr, yr))
                    ax.scatter([xr], [yr], s=60, facecolors="none", edgecolors=INK, linewidths=0.6, linestyles=ls, zorder=4)
        ax.grid(axis="y", color=GRID, lw=0.4); ax.set_axisbelow(True)
        if r == 0: ax.set_title(fam, fontsize=7.5, color=INK, weight="bold", loc="left", pad=4)
        if c == 0: ax.set_ylabel(f"{title}\n{unit}", color=INK, fontsize=6.5)
        if r == 2: ax.set_xlabel("Latency [ms]")
        labels.sort(key=lambda t: t[0])
        place(fig, ax, pts, lines, rings, labels)
fig.tight_layout(rect=(0, 0, 1, 1))
fig.savefig(OUT)
print("ok")
