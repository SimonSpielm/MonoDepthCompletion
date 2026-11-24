#!/usr/bin/env python3
"""
Schritt 2/3 der Karten-Auswertung: gefüllte Tiefe -> nvblox-TSDF-Karten mit Schnappschüssen.

Für jede Variante wird eine eigene nvblox-Karte aus dem EVAL-Segment gebaut, und einmal eine
Referenzkarte aus dem REF-Segment (Nah-Scan, Standard: rohe Tiefe). Integration und
Belegt-Regel sind identisch zu start_nvblox.py:
  TSDF, voxel 2 cm, jedes Tiefenbild genau einmal
  belegt    = (Gewicht > w_min) & (Distanz < occ_band * voxel)
  bekannt   = Gewicht > 0          (unbekannt zählt in start_nvblox.py als frei!)
Abgefragt wird ein festes, weltfestes Gitter (gleich für alle Karten der Szene), alle
--snapshot-every Frames und am Ende.

Segmente: aus <scene>/scene_log.yaml (segments.eval / segments.ref, Sekunden ab Aufnahmestart)
oder per --eval / --ref "start:end" in Sekunden.

Ausgabe <scene>/maps/:
  grid.npz            origin, dims, voxel, bounds
  map_<name>.npz      snap_frames, snap_t, occ (S, packbits), known (S, packbits)
                      name = Variante (eval) bzw. ref

Start (Env mit nvblox_torch, wie start_nvblox.py):
  python map_build.py --scene ~/map_eval_scenes/s01_monitor_tisch
"""
import os
import argparse
import time

import numpy as np


# ---------------------------------------------------------------- Segmente

def parse_seg(s):
    if s is None or s == "":
        return None
    a, b = s.split(":")
    return float(a), float(b)


def load_segments(scene):
    p = os.path.join(scene, "scene_log.yaml")
    if not os.path.exists(p):
        return None, None
    try:
        import yaml
    except ImportError:
        print("pyyaml fehlt -> Segmente per --eval/--ref angeben")
        return None, None
    with open(p) as f:
        d = yaml.safe_load(f) or {}
    seg = d.get("segments") or {}

    def get(k):
        s = seg.get(k) or {}
        if s.get("start") is None or s.get("end") is None:
            return None
        return float(s["start"]), float(s["end"])
    return get("eval"), get("ref")


def frames_in(rel_s, seg, valid):
    if seg is None:
        return np.flatnonzero(valid)
    return np.flatnonzero(valid & (rel_s >= seg[0]) & (rel_s <= seg[1]))


# ---------------------------------------------------------------- Gitter

def make_grid(cam_pos, voxel, margin_xy, z_range, bounds=None):
    if bounds is not None:
        lo = np.array(bounds[0::2], float)
        hi = np.array(bounds[1::2], float)
    else:
        lo = np.array([cam_pos[:, 0].min() - margin_xy, cam_pos[:, 1].min() - margin_xy, z_range[0]])
        hi = np.array([cam_pos[:, 0].max() + margin_xy, cam_pos[:, 1].max() + margin_xy, z_range[1]])
    lo = np.floor(lo / voxel) * voxel                     # auf nvblox-Voxelraster einrasten
    dims = np.ceil((hi - lo) / voxel).astype(int)
    return lo, dims


def grid_points(origin, dims, voxel, k0, k1):
    """Voxelzentren der flachen Indizes [k0, k1) (C-Reihenfolge x, y, z)."""
    k = np.arange(k0, k1)
    iz = k % dims[2]
    iy = (k // dims[2]) % dims[1]
    ix = k // (dims[2] * dims[1])
    return np.stack([ix, iy, iz], 1).astype(np.float32) * voxel + origin + 0.5 * voxel


# ---------------------------------------------------------------- nvblox

class MapRunner:
    def __init__(self, K, H, W, voxel, w_min, occ_band, chunk):
        import torch
        from nvblox_torch.mapper import Mapper
        from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
        from nvblox_torch.sensor import Sensor
        from nvblox_torch.mapper import QueryType
        self.torch, self.QueryType = torch, QueryType
        self.mapper = Mapper(voxel_sizes_m=[voxel], integrator_types=ProjectiveIntegratorType.TSDF)
        self.sensor = Sensor.from_camera(fu=float(K[0, 0]), fv=float(K[1, 1]), cu=float(K[0, 2]),
                                         cv=float(K[1, 2]), width=int(W), height=int(H))
        self.voxel, self.w_min, self.occ = voxel, w_min, occ_band * voxel
        self.chunk = chunk
        self.depth_gpu = torch.empty((H, W), dtype=torch.float32, device="cuda")

    def integrate(self, depth_mm, T_wc):
        torch = self.torch
        self.depth_gpu.copy_(torch.from_numpy(depth_mm.astype(np.float32) * 0.001))
        self.mapper.add_depth_frame(self.depth_gpu, torch.from_numpy(T_wc.astype(np.float32)),
                                    self.sensor, mapper_id=0)

    def query(self, origin, dims, voxel):
        torch = self.torch
        n = int(np.prod(dims))
        occ = np.zeros(n, bool)
        known = np.zeros(n, bool)
        for k0 in range(0, n, self.chunk):
            k1 = min(n, k0 + self.chunk)
            pts = torch.from_numpy(grid_points(origin, dims, voxel, k0, k1)).to("cuda")
            out = self.mapper.query_layer(self.QueryType.TSDF, pts, mapper_id=0)
            d, w = out[:, 0], out[:, 1]
            occ[k0:k1] = ((w > self.w_min) & (d < self.occ)).cpu().numpy()
            known[k0:k1] = (w > 0).cpu().numpy()
        return occ, known


def build(name, frames, depth, poses, K, H, W, grid, a, rel_s, out_dir):
    origin, dims = grid
    runner = MapRunner(K, H, W, a.voxel, a.w_min, a.occ_band, a.chunk)
    snaps_f, snaps_t, occs, knowns = [], [], [], []
    t0 = time.perf_counter()
    for j, i in enumerate(frames):
        runner.integrate(np.asarray(depth[i]), poses[i])
        last = j == len(frames) - 1
        if (j + 1) % a.snapshot_every == 0 or last:
            occ, known = runner.query(origin, dims, a.voxel)
            snaps_f.append(int(i))
            snaps_t.append(float(rel_s[i]))
            occs.append(np.packbits(occ))
            knowns.append(np.packbits(known))
            print(f"  {name:6s} {j + 1:5d}/{len(frames)}  belegt {occ.sum():8d}  "
                  f"bekannt {known.sum():9d}  {time.perf_counter() - t0:6.1f}s", flush=True)
    np.savez_compressed(os.path.join(out_dir, f"map_{name}.npz"), snap_frames=np.array(snaps_f),
                        snap_t=np.array(snaps_t), occ=np.stack(occs), known=np.stack(knowns),
                        n_vox=int(np.prod(dims)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--fill", default=None, help="Standard: <scene>/fill")
    ap.add_argument("--out", default=None, help="Standard: <scene>/maps")
    ap.add_argument("--variants", default=None, help="Standard: alle aus fill/meta.npz")
    ap.add_argument("--ref-variant", default="raw")
    ap.add_argument("--eval", default=None, help="Sekunden start:end (sonst scene_log.yaml)")
    ap.add_argument("--ref", default=None, help="Sekunden start:end (sonst scene_log.yaml)")
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--w-min", type=float, default=0.5)
    ap.add_argument("--occ-band", type=float, default=0.5, help="in Voxeln, wie start_nvblox.py")
    ap.add_argument("--snapshot-every", type=int, default=15, help="Frames (15 = 1 s)")
    ap.add_argument("--margin-xy", type=float, default=2.0, help="Gitter um Kamerapfad [m]")
    ap.add_argument("--z-range", default="-0.5,2.0", help="absolut im map-Frame [m]; mit = angeben, z. B. --z-range=-1,3")
    ap.add_argument("--bounds", default=None, help="xmin,xmax,ymin,ymax,zmin,zmax (statt auto)")
    ap.add_argument("--chunk", type=int, default=2_000_000)
    a = ap.parse_args()

    fill_dir = a.fill or os.path.join(a.scene, "fill")
    out_dir = a.out or os.path.join(a.scene, "maps")
    os.makedirs(out_dir, exist_ok=True)
    meta = np.load(os.path.join(fill_dir, "meta.npz"))
    stamps, poses, K = meta["stamps_ns"], meta["poses"], meta["K"]
    H, W = int(meta["H"]), int(meta["W"])
    variants = a.variants.split(",") if a.variants else [str(v) for v in meta["variants"]]
    valid = ~np.isnan(poses[:, 0, 0])

    start_file = os.path.join(a.scene, "record_start_unix.txt")
    t_start = float(open(start_file).read()) if os.path.exists(start_file) else stamps[0] * 1e-9
    rel_s = stamps * 1e-9 - t_start
    seg_eval, seg_ref = load_segments(a.scene)
    seg_eval = parse_seg(a.eval) or seg_eval
    seg_ref = parse_seg(a.ref) or seg_ref
    f_eval = frames_in(rel_s, seg_eval, valid)
    f_ref = frames_in(rel_s, seg_ref, valid)
    print(f"Frames gesamt {len(stamps)}, mit Pose {valid.sum()} | eval {seg_eval} -> {len(f_eval)} "
          f"| ref {seg_ref} -> {len(f_ref)}")
    if seg_ref is None and not a.no_ref:
        print("WARNUNG: kein REF-Segment -> Referenz = alle Frames (nur für Tests sinnvoll)")

    cam = poses[valid][:, :3, 3]
    zr = [float(v) for v in a.z_range.split(",")]
    bounds = [float(v) for v in a.bounds.split(",")] if a.bounds else None
    origin, dims = make_grid(cam, a.voxel, a.margin_xy, zr, bounds)
    print(f"Gitter: Ursprung {origin.round(2)}, {dims} = {np.prod(dims) / 1e6:.1f} Mio. Voxel")
    np.savez(os.path.join(out_dir, "grid.npz"), origin=origin, dims=dims, voxel=a.voxel,
             eval_frames=f_eval, ref_frames=f_ref, cam_pos=poses[:, :3, 3], rel_s=rel_s,
             seg_eval=np.array(seg_eval or (np.nan, np.nan)),
             seg_ref=np.array(seg_ref or (np.nan, np.nan)))

    for v in variants:
        depth = np.load(os.path.join(fill_dir, f"depth_{v}.npy"), mmap_mode="r")
        build(v, f_eval, depth, poses, K, H, W, (origin, dims), a, rel_s, out_dir)
    if not a.no_ref:
        depth = np.load(os.path.join(fill_dir, f"depth_{a.ref_variant}.npy"), mmap_mode="r")
        build("ref", f_ref, depth, poses, K, H, W, (origin, dims), a, rel_s, out_dir)
    print(f"fertig: {out_dir}")


if __name__ == "__main__":
    main()
