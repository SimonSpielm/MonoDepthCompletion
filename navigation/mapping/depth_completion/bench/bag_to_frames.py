#!/usr/bin/env python3
"""
Extract RGB/depth pairs from a ROS 1 or ROS 2 bag into the frames.npz format of
depth_bench.py (rgb uint8, depth float32 [m], fov_x [deg]). No ROS install needed,
only `pip install rosbags`.

    python bag_to_frames.py ../../../../rosbags/depth_estimation --out still1.npz --max 60

Pairs are matched by identical header.stamp (as published by ln_ros_camera_bridge);
`--step` keeps every k-th pair. fov_x comes from the color camera_info (fx, width).
"""
import argparse
import math
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

COLOR, DEPTH, INFO = "/camera/color/image_raw", "/camera/depth/image_raw", "/camera/color/camera_info"


def stamp(m):
    return (m.header.stamp.sec, m.header.stamp.nanosec)


def image(m):
    a = np.frombuffer(m.data, np.uint8)
    if m.encoding in ("16UC1", "mono16"):
        return a.view(np.uint16).reshape(m.height, m.width).astype(np.float32) / 1000.0
    if m.encoding == "32FC1":
        return a.view(np.float32).reshape(m.height, m.width).copy()
    img = a.reshape(m.height, m.width, -1)[..., :3]
    return np.ascontiguousarray(img[..., ::-1] if m.encoding.startswith("bgr") else img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=60, help="maximum number of pairs")
    ap.add_argument("--step", type=int, default=1, help="keep every k-th pair")
    ap.add_argument("--color", default=COLOR)
    ap.add_argument("--depth", default=DEPTH)
    ap.add_argument("--info", default=INFO)
    a = ap.parse_args()

    rgbs, depths, fov = {}, {}, None
    with AnyReader([Path(a.bag)], default_typestore=get_typestore(Stores.ROS2_JAZZY)) as r:
        conns = [c for c in r.connections if c.topic in (a.color, a.depth, a.info)]
        for c, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            if c.topic == a.info:
                if fov is None:
                    k = getattr(m, "k", None)
                    fx = float((k if k is not None else m.K)[0])
                    fov = math.degrees(2 * math.atan(0.5 * m.width / fx))
            elif c.topic == a.color:
                rgbs[stamp(m)] = image(m)
            else:
                depths[stamp(m)] = image(m)

    keys = sorted(set(rgbs) & set(depths))[:: a.step][: a.max]
    if not keys:
        raise SystemExit("keine RGB/Tiefe-Paare mit gleichem Zeitstempel gefunden")
    rgb = np.stack([rgbs[k] for k in keys])
    depth = np.stack([depths[k] for k in keys])
    if depth.shape[1:] != rgb.shape[1:3]:
        raise SystemExit(f"Tiefe {depth.shape[1:]} nicht auf RGB {rgb.shape[1:3]} registriert")
    np.savez(a.out, rgb=rgb, depth=depth, fov_x=np.full(len(keys), fov if fov else np.nan))
    print(f"{len(keys)} Paare ({len(rgbs)} RGB, {len(depths)} Tiefe), {rgb.shape[2]}x{rgb.shape[1]}, "
          f"FoV_x {fov:.1f} Grad, gueltig {100 * (depth > 0).mean():.1f} % -> {a.out}")


if __name__ == "__main__":
    main()
