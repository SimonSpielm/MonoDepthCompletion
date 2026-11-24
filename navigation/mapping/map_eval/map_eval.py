#!/usr/bin/env python3
"""
Schritt 3/3 der Karten-Auswertung: Metriken, Kurven und Differenzkarten (eine oder mehrere Szenen).

Vergleicht die Karten jeder Variante (EVAL-Segment, letzter Schnappschuss) mit der Referenzkarte
(REF-Nah-Scan). Alle Größen auf Voxelebene (2 cm), k = Toleranz in Voxeln (Standard 2 = 4 cm):

  completeness@k  Anteil Referenz-Hindernisvoxel, die in der Karte (±k) belegt sind     (↑)
  accuracy@k      Anteil belegter Kartenvoxel, die (±k) ein Referenz-Hindernis haben    (↑)
  fscore@k        harmonisches Mittel der beiden                                        (↑)
  phantom         Anteil belegter Kartenvoxel in Raum, den die Referenz als FREI gesehen hat
                  (weiter als k von jedem Referenz-Hindernis)                           (↓)
  missed_unknown  Anteil Referenz-Hindernisse, die in der Karte UNBEKANNT sind. Gefährlich,
                  weil unbekannt im Betrieb als frei gilt                               (↓)
  missed_free     Anteil Referenz-Hindernisse, die die Karte als FREI beobachtet hat    (↓)
  ws_unknown      Anteil unbekannter Voxel im Arbeitsraum (Kugel --ws-radius um die
                  mittlere Kameraposition)                                              (↓)
  flip_rate       mittlerer Anteil der Voxel, deren Status (belegt/frei) zwischen zwei
                  Schnappschüssen wechselt (nur im statischen Teil sinnvoll)            (↓)
  t90             Zeit [s], bis completeness 90 % des Endwerts erreicht                 (↓)

Differenzkarten (pro Szene, Ordner <scene>/eval/):
  diff_<var>_vs_ref.png / .ply   grau = beide belegt, rot = nur Karte (Phantom),
                                 cyan = Hindernis unbekannt, blau = Hindernis als frei gesehen
  diff_A_vs_B.png / .ply         grau = beide, orange = nur A, blau = nur B
  --rviz veröffentlicht dieselben Wolken als PointCloud2 (/map_eval/<name>, frame map) für
  Screenshots in RViz.

Start:
  python map_eval.py --scenes ~/map_eval_scenes/s01_* ~/map_eval_scenes/s02_* --out ~/map_eval_summary
"""
import os
import argparse
import csv
import glob

import numpy as np

COLORS = {"both": (170, 170, 170), "phantom": (230, 40, 40), "miss_unknown": (0, 200, 220),
          "miss_free": (40, 70, 230), "onlyA": (240, 140, 30), "onlyB": (40, 110, 230)}
METRICS = ["completeness", "accuracy", "fscore", "phantom", "missed_unknown", "missed_free",
           "ws_unknown", "flip_rate", "t90", "n_occ"]
LOWER_BETTER = {"phantom", "missed_unknown", "missed_free", "ws_unknown", "flip_rate", "t90"}


# ---------------------------------------------------------------- Hilfen

def dilate(m, k):
    """Würfel-Dilatation (Chebyshev-Radius k), separabel."""
    out = m
    for ax in range(3):
        src = np.moveaxis(out, ax, 0)
        acc = src.copy()
        for s in range(1, k + 1):
            acc[s:] |= src[:-s]
            acc[:-s] |= src[s:]
        out = np.moveaxis(acc, 0, ax)
    return out


def load_map(path, dims):
    d = np.load(path)
    n = int(np.prod(dims))
    unpack = lambda a: np.unpackbits(a, count=n).astype(bool).reshape(dims)
    return {"t": d["snap_t"], "frames": d["snap_frames"],
            "occ": [unpack(x) for x in d["occ"]], "known": [unpack(x) for x in d["known"]]}


def voxel_xyz(mask, origin, voxel):
    idx = np.argwhere(mask)
    return idx.astype(np.float32) * voxel + origin + 0.5 * voxel


def write_ply(path, layers, origin, voxel):
    pts, cols = [], []
    for mask, col in layers:
        p = voxel_xyz(mask, origin, voxel)
        pts.append(p)
        cols.append(np.tile(np.array(col, np.uint8), (len(p), 1)))
    pts = np.concatenate(pts) if pts else np.zeros((0, 3), np.float32)
    cols = np.concatenate(cols) if cols else np.zeros((0, 3), np.uint8)
    rec = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["r"], rec["g"], rec["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(rec)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
                .encode())
        f.write(rec.tobytes())
    return pts, cols


def topdown_png(path, layers, title, origin, voxel, labels):
    """Draufsicht: pro Säule die Klasse mit höchster Priorität (Reihenfolge in layers)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    dims = layers[0][0].shape
    img = np.full((dims[1], dims[0], 3), 255, np.uint8)       # (y, x)
    for mask, col in reversed(layers):                         # erste Ebene gewinnt
        col_any = mask.any(axis=2).T
        img[col_any] = col
    fig, ax = plt.subplots(figsize=(5, 5 * dims[1] / max(dims[0], 1) + 0.6), dpi=200)
    ext = [origin[0], origin[0] + dims[0] * voxel, origin[1], origin[1] + dims[1] * voxel]
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=9)
    ax.legend(handles=[Patch(color=np.array(c) / 255, label=l) for (_, c), l in zip(layers, labels)],
              fontsize=6, loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- Metriken

def scene_metrics(scene, k, ws_radius, static_until):
    mdir = os.path.join(scene, "maps")
    g = np.load(os.path.join(mdir, "grid.npz"))
    origin, dims, voxel = g["origin"], tuple(int(v) for v in g["dims"]), float(g["voxel"])
    ref = load_map(os.path.join(mdir, "map_ref.npz"), dims)
    r_occ, r_known = ref["occ"][-1], ref["known"][-1]
    r_free = r_known & ~r_occ
    r_occ_d = dilate(r_occ, k)

    cam = g["cam_pos"]
    cam = cam[~np.isnan(cam[:, 0])]
    c = cam.mean(0)
    ii = np.indices(dims).reshape(3, -1).T.astype(np.float32) * voxel + origin + 0.5 * voxel
    ws = (np.linalg.norm(ii - c, axis=1) <= ws_radius).reshape(dims)
    del ii

    variants = sorted(os.path.basename(p)[4:-4] for p in glob.glob(os.path.join(mdir, "map_*.npz"))
                      if not p.endswith("map_ref.npz"))
    rows, maps, curves = [], {}, {}
    n_ref = max(int(r_occ.sum()), 1)
    for v in variants:
        m = load_map(os.path.join(mdir, f"map_{v}.npz"), dims)
        occ, known = m["occ"][-1], m["known"][-1]
        maps[v] = (occ, known)
        occ_d = dilate(occ, k)
        n_occ = max(int(occ.sum()), 1)
        comp = float((r_occ & occ_d).sum()) / n_ref
        acc = float((occ & r_occ_d).sum()) / n_occ
        miss = r_occ & ~occ_d
        row = dict(scene=os.path.basename(scene.rstrip("/")), variant=v, n_occ=int(occ.sum()),
                   completeness=comp, accuracy=acc,
                   fscore=2 * comp * acc / max(comp + acc, 1e-9),
                   phantom=float((occ & r_free & ~r_occ_d).sum()) / n_occ,
                   missed_unknown=float((miss & ~known).sum()) / n_ref,
                   missed_free=float((miss & known).sum()) / n_ref,
                   ws_unknown=float((ws & ~known).sum()) / max(int(ws.sum()), 1))
        # Zeitverlauf
        ts = m["t"] - m["t"][0]
        comp_t = [float((r_occ & dilate(o, k)).sum()) / n_ref for o in m["occ"]]
        flips = []
        for j in range(1, len(m["occ"])):
            if static_until is not None and ts[j] > static_until:
                break
            both = m["known"][j] & m["known"][j - 1]
            ch = (m["occ"][j] ^ m["occ"][j - 1]) & both
            base = (m["occ"][j] | m["occ"][j - 1]) & both
            flips.append(float(ch.sum()) / max(int(base.sum()), 1))
        row["flip_rate"] = float(np.mean(flips)) if flips else float("nan")
        fin = comp_t[-1] if comp_t else 0
        hit = [t for t, cv in zip(ts, comp_t) if cv >= 0.9 * fin] if fin > 0 else []
        row["t90"] = float(hit[0]) if hit else float("nan")
        curves[v] = (ts, np.array(comp_t), np.array(flips))
        rows.append(row)
    return rows, maps, curves, (r_occ, r_known, r_free, r_occ_d), (origin, dims, voxel)


# ---------------------------------------------------------------- Ausgabe je Szene

def scene_outputs(scene, maps, ref, grid, curves, k, rviz_pub):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(scene, "eval")
    os.makedirs(out, exist_ok=True)
    origin, dims, voxel = grid
    r_occ, r_known, r_free, r_occ_d = ref
    for v, (occ, known) in maps.items():
        occ_d = dilate(occ, k)
        layers = [((r_occ & ~occ_d & ~known), COLORS["miss_unknown"]),
                  ((r_occ & ~occ_d & known), COLORS["miss_free"]),
                  ((occ & r_free & ~r_occ_d), COLORS["phantom"]),
                  ((occ & r_occ_d), COLORS["both"])]
        labels = ["Hindernis unbekannt", "Hindernis als frei", "Phantom (nur Karte)", "beide belegt"]
        name = f"diff_{v}_vs_ref"
        pts, cols = write_ply(os.path.join(out, name + ".ply"), layers, origin, voxel)
        topdown_png(os.path.join(out, name + ".png"), layers, f"{v} vs. Referenz", origin, voxel,
                    labels)
        if rviz_pub:
            rviz_pub(name, pts, cols)
    if "A" in maps and "B" in maps:
        (oa, _), (ob, _) = maps["A"], maps["B"]
        layers = [(oa & ~dilate(ob, k), COLORS["onlyA"]), (ob & ~dilate(oa, k), COLORS["onlyB"]),
                  (oa & ob, COLORS["both"])]
        labels = ["nur A", "nur B", "beide"]
        pts, cols = write_ply(os.path.join(out, "diff_A_vs_B.ply"), layers, origin, voxel)
        topdown_png(os.path.join(out, "diff_A_vs_B.png"), layers, "A vs. B", origin, voxel, labels)
        if rviz_pub:
            rviz_pub("diff_A_vs_B", pts, cols)
    # Kurven
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6), dpi=200)
    for v, (ts, comp_t, flips) in curves.items():
        ax[0].plot(ts, 100 * comp_t, label=v, lw=1.5)
        if len(flips):
            ax[1].plot(ts[1:1 + len(flips)], 100 * flips, label=v, lw=1.2)
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel(f"completeness@{k} [%]")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("voxel flips [%]")
    for x in ax:
        x.grid(alpha=0.3)
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "map_over_time.png"))
    plt.close(fig)


def make_rviz_publisher():
    import rclpy
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    rclpy.init()
    node = rclpy.create_node("map_eval_diff")
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)
    pubs = {}

    def pub(name, pts, cols):
        if name not in pubs:
            pubs[name] = node.create_publisher(PointCloud2, f"/map_eval/{name}", qos)
        rgb = ((cols[:, 0].astype(np.uint32) << 16) | (cols[:, 1].astype(np.uint32) << 8)
               | cols[:, 2].astype(np.uint32)).view(np.float32)
        data = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
        data["x"], data["y"], data["z"], data["rgb"] = pts[:, 0], pts[:, 1], pts[:, 2], rgb
        msg = PointCloud2(header=Header(frame_id="map", stamp=node.get_clock().now().to_msg()),
                          height=1, width=len(data), is_bigendian=False, is_dense=True,
                          point_step=16, row_step=16 * len(data), data=data.tobytes(),
                          fields=[PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32,
                                             count=1) for i, n in enumerate(("x", "y", "z", "rgb"))])
        pubs[name].publish(msg)
        print(f"  RViz: /map_eval/{name} ({len(data)} Punkte)")
    return pub, node


# ---------------------------------------------------------------- Zusammenfassung

def fmt(v, m):
    if v is None or not np.isfinite(v):
        return "–"
    if m in ("t90",):
        return f"{v:.1f}"
    if m == "n_occ":
        return f"{v:.0f}"
    return f"{100 * v:.1f}"


def write_summary(rows, out, k):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "map_metrics_all.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "variant"] + METRICS)
        w.writeheader()
        for r in rows:
            w.writerow({kk: r.get(kk) for kk in ["scene", "variant"] + METRICS})
    variants = [v for v in ("raw", "global", "A", "B") if any(r["variant"] == v for r in rows)]
    variants += sorted({r["variant"] for r in rows} - set(variants))
    scenes = sorted({r["scene"] for r in rows})
    agg = {}
    for v in variants:
        for m in METRICS:
            vals = np.array([r[m] for r in rows if r["variant"] == v], float)
            vals = vals[np.isfinite(vals)]
            agg[(v, m)] = (vals.mean() if len(vals) else np.nan, vals.std() if len(vals) > 1 else 0.0)

    L = [f"# Karten-Auswertung ({len(scenes)} Szenen, k = {k} Voxel)\n",
         "Mittelwert ± Std über Szenen. % außer t90 [s] und n_occ.\n",
         "| Metrik | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
    for m in METRICS:
        arrow = "↓" if m in LOWER_BETTER else ("" if m == "n_occ" else "↑")
        L.append(f"| {m} {arrow} | " + " | ".join(
            f"{fmt(agg[(v, m)][0], m)} ± {fmt(agg[(v, m)][1], m)}" for v in variants) + " |")
    L.append("\n## Je Szene\n")
    for s in scenes:
        L.append(f"### {s}\n")
        L.append("| Metrik | " + " | ".join(variants) + " |")
        L.append("|---|" + "---|" * len(variants))
        for m in METRICS:
            vals = {r["variant"]: r[m] for r in rows if r["scene"] == s}
            L.append(f"| {m} | " + " | ".join(fmt(vals.get(v, np.nan), m) for v in variants) + " |")
        L.append("")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    # LaTeX-Tabelle fürs Paper
    names = {"raw": "Raw", "global": "Global", "A": "Var.~A", "B": "Var.~B"}
    show = ["completeness", "accuracy", "fscore", "phantom", "missed_unknown", "flip_rate", "t90"]
    head = {"completeness": "Compl.", "accuracy": "Acc.", "fscore": "F", "phantom": "Phant.",
            "missed_unknown": "Miss.\\,unk.", "flip_rate": "Flips", "t90": "\\(t_{90}\\)"}
    T = ["\\begin{tabular}{@{}l" + "c" * len(show) + "@{}}", "\\toprule",
         "Method & " + " & ".join(head[m] + ("\\,\\(\\downarrow\\)" if m in LOWER_BETTER
                                             else "\\,\\(\\uparrow\\)") for m in show) + " \\\\",
         " & " + " & ".join("[s]" if m == "t90" else "[\\%]" for m in show) + " \\\\", "\\midrule"]
    for v in variants:
        cells = []
        for m in show:
            mu, sd = agg[(v, m)]
            cells.append(f"{fmt(mu, m)}\\,\\(\\pm\\)\\,{fmt(sd, m)}" if len(scenes) > 1
                         else fmt(mu, m))
        T.append(f"{names.get(v, v)} & " + " & ".join(cells) + " \\\\")
    T += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(out, "map_table.tex"), "w") as f:
        f.write("\n".join(T) + "\n")
    print("\n".join(L[:len(METRICS) + 4]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="Ordner für die Zusammenfassung über alle Szenen")
    ap.add_argument("--k", type=int, default=2, help="Toleranz in Voxeln")
    ap.add_argument("--ws-radius", type=float, default=1.2, help="Arbeitsraum-Radius [m]")
    ap.add_argument("--static-until", type=float, default=None,
                    help="flip_rate nur bis zu dieser Zeit [s] ab Segmentstart (statischer Teil)")
    ap.add_argument("--rviz", action="store_true", help="Differenzwolken für RViz veröffentlichen")
    a = ap.parse_args()

    rviz_pub, node = (make_rviz_publisher() if a.rviz else (None, None))
    rows = []
    for s in a.scenes:
        print(f"== {s}")
        r, maps, curves, ref, grid = scene_metrics(s, a.k, a.ws_radius, a.static_until)
        scene_outputs(s, maps, ref, grid, curves, a.k, rviz_pub)
        rows += r
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.scenes[0].rstrip("/"))),
                                "map_eval_summary")
    write_summary(rows, out, a.k)
    print(f"\nZusammenfassung: {out}  (summary.md, map_metrics_all.csv, map_table.tex)")
    if node is not None:
        import rclpy
        print("RViz-Wolken sind latched veröffentlicht. Strg+C beendet.")
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
