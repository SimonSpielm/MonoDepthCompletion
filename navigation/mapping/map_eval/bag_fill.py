#!/usr/bin/env python3
"""
Schritt 1/3 der Karten-Auswertung: ROS-2-Bag -> gefüllte Tiefe je Variante.

Liest RGB, Tiefe, camera_info, TF (und optional ignore_mask) aus dem Bag, rechnet MoGe-2 genau
einmal pro Frame und erzeugt daraus kausal (Frame für Frame, wie im Betrieb) die Varianten
  raw    : Sensortiefe unverändert
  global : nur Stufe 1 (globaler Faktor, EMA 0.3)            -> moge_depth.align_and_fill(radius 0)
  A      : lokale affine Angleichung, getunt (r=16, eps=1e-3) -> moge_depth.align_and_fill
  B      : Gradienten-Fusion, getunt, Warmstart (GPU wenn da)  -> moge_depth_optim.fill(_torch)
Gefüllt wird VOR dem Maskieren; Pixel aus ignore_mask werden danach wieder 0 (wie im Paper).

Ausgabe in --out (Standard: <szene>/fill):
  meta.npz           stamps_ns, poses (T,4,4 map<-camera, NaN wenn TF fehlt), K, H, W, variants
  depth_<var>.npy    uint16 [mm], Form (T,H,W), memmap
  pred.npy           float16 MoGe-Tiefe [m] (nur mit --save-pred, für spätere Neu-Füllungen)
  fill_log.txt       Zeiten, Fehlende Posen, Skalierungsfaktoren

Start (conda-Env "moge", ROS gesourct):
  python bag_fill.py --scene ~/map_eval_scenes/s01_monitor_tisch
"""
import os
import sys
import argparse
import math
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
VARIANTS_ALL = ("raw", "global", "A", "B")


# ---------------------------------------------------------------- TF offline

def quat_to_mat(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class OfflineTF:
    """Minimaler TF-Puffer: Kanten parent->child mit Zeitreihe, Lookup per nächstem Stempel."""

    def __init__(self, tol_ns=50_000_000):
        self.dyn = defaultdict(list)      # (parent, child) -> [(stamp_ns, T)]
        self.static = {}                  # (parent, child) -> T
        self.parent = {}                  # child -> parent
        self.tol_ns = tol_ns

    def add(self, parent, child, stamp_ns, T, static=False):
        parent, child = parent.lstrip("/"), child.lstrip("/")
        self.parent[child] = parent
        if static:
            self.static[(parent, child)] = T
        else:
            self.dyn[(parent, child)].append((stamp_ns, T))

    def finalize(self):
        for k, v in self.dyn.items():
            v.sort(key=lambda e: e[0])
            self.dyn[k] = (np.array([e[0] for e in v], np.int64), [e[1] for e in v])

    def _edge(self, parent, child, t):
        if (parent, child) in self.static:
            return self.static[(parent, child)]
        ts, Ts = self.dyn[(parent, child)]
        i = int(np.searchsorted(ts, t))
        best = min((j for j in (i - 1, i) if 0 <= j < len(ts)), key=lambda j: abs(ts[j] - t))
        return Ts[best] if abs(int(ts[best]) - t) <= self.tol_ns else None

    def lookup(self, target, source, t):
        """T_target_source (Punkte aus source nach target) zum Zeitpunkt t, oder None."""
        T = np.eye(4)
        f = source
        for _ in range(32):
            if f == target:
                return T
            p = self.parent.get(f)
            if p is None:
                return None
            E = self._edge(p, f, t)
            if E is None:
                return None
            T = E @ T
            f = p
        return None


def tf_msg_to_T(tr):
    T = np.eye(4)
    q = tr.transform.rotation
    p = tr.transform.translation
    T[:3, :3] = quat_to_mat(q.x, q.y, q.z, q.w)
    T[:3, 3] = (p.x, p.y, p.z)
    return T


def stamp_ns(h):
    return int(h.stamp.sec) * 1_000_000_000 + int(h.stamp.nanosec)


# ---------------------------------------------------------------- Bag lesen

def open_reader(bag):
    import rosbag2_py
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""),
           rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    return r, types


def iter_bag(bag, topics):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    r, types = open_reader(bag)
    cls = {t: get_message(types[t]) for t in topics if t in types}
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic in cls:
            yield topic, deserialize_message(data, cls[topic])


def to_rgb(msg):
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}[msg.encoding]
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * ch]
    img = img.reshape(msg.height, msg.width, ch)[..., :3]
    return np.ascontiguousarray(img[..., ::-1] if msg.encoding.startswith("bgr") else img)


def to_depth_m(msg):
    if msg.encoding == "32FC1":
        d = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.step // 4)[:, : msg.width]
        return np.nan_to_num(d.copy(), nan=0.0, posinf=0.0)
    d = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[:, : msg.width]
    return d.astype(np.float32) * 0.001


def to_mask(msg):
    m = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width]
    return m > 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--scene", required=True, help="Szenen-Ordner aus record_scene.sh")
    ap.add_argument("--bag", default=None, help="Standard: <scene>/bag")
    ap.add_argument("--out", default=None, help="Standard: <scene>/fill")
    ap.add_argument("--variants", default=",".join(VARIANTS_ALL))
    ap.add_argument("--stride", type=int, default=1, help="nur jeden N-ten Frame")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--color", default="/camera/color/image_raw")
    ap.add_argument("--depth", default="/camera/depth/image_raw")
    ap.add_argument("--info", default="/camera/color/camera_info")
    ap.add_argument("--mask", default="/camera/depth/ignore_mask")
    ap.add_argument("--map-frame", default="map")
    ap.add_argument("--cam-frame", default="camera")
    ap.add_argument("--save-pred", action="store_true")
    ap.add_argument("--cpu-fusion", action="store_true", help="B auf der CPU statt GPU")
    a = ap.parse_args()

    bag = a.bag or os.path.join(a.scene, "bag")
    out = a.out or os.path.join(a.scene, "fill")
    variants = [v for v in a.variants.split(",") if v]
    assert all(v in VARIANTS_ALL for v in variants), variants
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "fill_log.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")

    # ---- Pass 1: TF, camera_info, Frame-Stempel
    say(f"Pass 1: TF + Stempel aus {bag}")
    tf = OfflineTF()
    K = H = W = None
    color_st, depth_st, has_mask = set(), set(), False
    for topic, msg in iter_bag(bag, ["/tf", "/tf_static", a.info, a.color, a.depth, a.mask]):
        if topic in ("/tf", "/tf_static"):
            for tr in msg.transforms:
                tf.add(tr.header.frame_id, tr.child_frame_id, stamp_ns(tr.header),
                       tf_msg_to_T(tr), static=(topic == "/tf_static"))
        elif topic == a.info and K is None:
            K, H, W = np.array(msg.k, float).reshape(3, 3), int(msg.height), int(msg.width)
        elif topic == a.color:
            color_st.add(stamp_ns(msg.header))
        elif topic == a.depth:
            depth_st.add(stamp_ns(msg.header))
        elif topic == a.mask:
            has_mask = True
    tf.finalize()
    if K is None:
        raise SystemExit(f"kein {a.info} im Bag")
    stamps = sorted(color_st & depth_st)[:: max(a.stride, 1)]
    if a.max_frames:
        stamps = stamps[: a.max_frames]
    poses = np.full((len(stamps), 4, 4), np.nan)
    for i, t in enumerate(stamps):
        T = tf.lookup(a.map_frame, a.cam_frame, t)
        if T is not None:
            poses[i] = T
    n_nopose = int(np.isnan(poses[:, 0, 0]).sum())
    say(f"  Frames (RGB+Tiefe gleicher Stempel): {len(stamps)} | ohne Pose: {n_nopose} | "
        f"ignore_mask: {'ja' if has_mask else 'NEIN'} | {W}x{H}")
    if not has_mask:
        say("  WARNUNG: keine ignore_mask -> maskierte Pixel (Mensch/Arm) wären von MoGe wieder "
            "gefüllt. Nur ok, wenn bei der Aufnahme kein Mensch/Arm im Bild war.")
    if len(stamps) == 0:
        raise SystemExit("keine Frames")

    # ---- Modelle
    sys.path.insert(0, REPO)
    import torch
    from navigation.mapping.moge_depth import align_and_fill
    from navigation.mapping.moge_depth_optim import (MoGe, GradientFusion, GradientFusionTorch,
                                                     fill, fill_torch)
    need_moge = any(v != "raw" for v in variants)
    moge = MoGe() if need_moge else None
    use_gpu = moge is not None and moge.dev.type == "cuda" and not a.cpu_fusion
    fusion = (GradientFusionTorch() if use_gpu else GradientFusion()) if "B" in variants else None
    fov_x = math.degrees(2 * math.atan(W / (2 * K[0, 0])))
    say(f"  MoGe: {'ja' if need_moge else 'nein'} | B auf {'GPU' if use_gpu else 'CPU'} | "
        f"FoV_x {fov_x:.1f} Grad")

    T = len(stamps)
    outs = {v: np.lib.format.open_memmap(os.path.join(out, f"depth_{v}.npy"), "w+",
                                         np.uint16, (T, H, W)) for v in variants}
    pred_mm = (np.lib.format.open_memmap(os.path.join(out, "pred.npy"), "w+", np.float16,
                                         (T, H, W)) if a.save_pred and need_moge else None)
    state = {v: None for v in variants}          # log_s je Variante (EMA wie im Betrieb)
    idx = {t: i for i, t in enumerate(stamps)}
    pending = defaultdict(dict)
    skipped = []
    tsum = defaultdict(float)
    done = 0
    t0 = time.perf_counter()

    def to_mm(d):
        return np.clip(np.rint(d * 1000.0), 0, 65535).astype(np.uint16)

    def process(i, rgb, depth, mask):
        nonlocal done
        if mask is not None:
            depth = np.where(mask, 0, depth).astype(np.float32)
        pred_t = pred = None
        if need_moge:
            ts = time.perf_counter()
            pred_t = moge(rgb, fov_x)
            if use_gpu:
                torch.cuda.synchronize()
            pred = pred_t.float().cpu().numpy()
            tsum["moge"] += time.perf_counter() - ts
            if pred_mm is not None:
                pred_mm[i] = np.where(np.isfinite(pred), pred, 0).astype(np.float16)
        for v in variants:
            ts = time.perf_counter()
            if v == "raw":
                d = depth
            elif v in ("global", "A"):
                r = 0 if v == "global" else 16
                d, _, state[v] = align_and_fill(depth, pred, state[v], radius_px=r, eps=0.001)
            else:
                if use_gpu:
                    d, _, state[v] = fill_torch(torch.from_numpy(depth).to(pred_t.device),
                                                pred_t, fusion, state[v])
                else:
                    d, _, state[v] = fill(depth, pred, fusion, state[v])
            if mask is not None:
                d = np.where(mask, 0, d)
            outs[v][i] = to_mm(d)
            tsum[v] += time.perf_counter() - ts
        done += 1
        if done % 50 == 0 or done == T:
            el = time.perf_counter() - t0
            say(f"  {done}/{T}  {el:.0f}s  ETA {el / done * (T - done):.0f}s  "
                + " ".join(f"{k} {1000 * s / done:.0f}ms" for k, s in tsum.items())
                + "".join(f" | s_{v}={math.exp(state[v]):.3f}" for v in ("global", "A", "B")
                          if v in state and state[v] is not None))

    # ---- Pass 2: Frames verarbeiten (kausal, in Stempel-Reihenfolge)
    say("Pass 2: füllen")
    next_i = 0
    for topic, msg in iter_bag(bag, [a.color, a.depth, a.mask]):
        t = stamp_ns(msg.header)
        if t not in idx:
            continue
        key = {a.color: "rgb", a.depth: "depth", a.mask: "mask"}[topic]
        pending[t][key] = msg
        # verarbeiten, sobald der nächste Frame vollständig ist (Reihenfolge bleibt kausal)
        while next_i < T:
            tn = stamps[next_i]
            p = pending.get(tn, {})
            if "rgb" in p and "depth" in p and (not has_mask or "mask" in p):
                rgb, depth = to_rgb(p["rgb"]), to_depth_m(p["depth"])
                if depth.shape != (H, W):
                    import cv2
                    depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
                mask = to_mask(p["mask"]) if "mask" in p else None
                process(next_i, rgb, depth, mask)
                pending.pop(tn, None)
                next_i += 1
            else:
                break
        # hängt ein Frame (z. B. Maske fehlt), nach 30 neueren Frames überspringen
        while next_i < T and len(pending) > 30 and stamps[next_i] < min(pending):
            skipped.append(next_i)
            poses[next_i] = np.nan
            next_i += 1
        if len(pending) > 30 and next_i < T and stamps[next_i] in pending:
            pending.pop(stamps[next_i])
            skipped.append(next_i)
            poses[next_i] = np.nan
            next_i += 1
    while next_i < T:                      # Rest am Ende unvollständig
        skipped.append(next_i)
        poses[next_i] = np.nan
        next_i += 1
    if skipped:
        say(f"  WARNUNG: {len(skipped)} Frames unvollständig -> übersprungen (Pose = NaN)")

    for m in outs.values():
        m.flush()
    np.savez(os.path.join(out, "meta.npz"), stamps_ns=np.array(stamps, np.int64), poses=poses,
             K=K, H=H, W=W, variants=np.array(variants), skipped=np.array(skipped, int),
             has_mask=has_mask, fov_x=fov_x)
    say(f"fertig: {out}")


if __name__ == "__main__":
    main()
