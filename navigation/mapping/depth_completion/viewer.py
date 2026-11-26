#!/usr/bin/env python3
"""
Viewer for the depth completion, in one Rerun window:

    top:     RGB | sensor depth | model depth | merged depth
    bottom:  3D point cloud of the sensor | of the model | of the merged depth

All views are synchronised; the 3D views can be rotated and zoomed with the mouse. Runs exactly
the code of the ROS node (depth_models.py + depth_fusion.fill), only with a display.

    python navigation/mapping/depth_completion/viewer.py                       # live, ROS 2
    python navigation/mapping/depth_completion/viewer.py --model depthpro2-s --stream
    python navigation/mapping/depth_completion/viewer.py --frames f.npz        # offline, no ROS
    python navigation/mapping/depth_completion/viewer.py --set edge_jump=0           # edge filter off
    python navigation/mapping/depth_completion/viewer.py --eval        # + error below 3 m over time

Live mode subscribes the raw bridge topics (--color, --depth, --info). Offline mode plays a
frames.npz (bench/bag_to_frames.py makes one from a rosbag). --save f.rrd writes a recording
instead of opening the window (open it later with `rerun f.rrd`).

Model depth = model prediction times the global scale. Merged depth = measurement where the
sensor measured, filled holes elsewhere; pixels discarded by the edge filter are magenta. In
the merged cloud, measured points have their RGB colour and filled points are orange; both can
be toggled separately in the Rerun entity tree (cloud/merged/measured, cloud/merged/filled).

--eval adds the error below 3 m as curves over time (AbsRel [%], MAE [cm]). For that, random
32 px blocks of measured pixels are hidden, filled and compared with the measurement; this
costs a second fill per frame.
"""
import argparse
import math
import os
import sys
import threading
import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_fusion import GradientFusion, fill  # noqa: E402
from depth_models import MODELS, load_model  # noqa: E402

FUSION_KEYS = ("down", "lam", "mu_near", "mu_far", "far_band", "sigma_edge", "z_ref",
               "outlier", "iters", "tol")
FILLED_COLOR = (255, 140, 0)
NEAR_M = 3.0


# ---------------------------------------------------------------- frame sources

def offline_frames(path):
    """Frames of a frames.npz: (rgb, depth [m], fx, fy, cx, cy). Principal point = image centre."""
    data = np.load(path)
    for i in range(len(data["depth"])):
        rgb, depth = data["rgb"][i], data["depth"][i].astype(np.float32)
        H, W = depth.shape
        fx = 0.5 * W / math.tan(math.radians(float(data["fov_x"][i])) / 2)
        yield rgb, depth, fx, fx, W / 2, H / 2


def live_frames(a):
    """Newest RGB/depth pair from ROS 2 with the intrinsics of the latest camera_info."""
    import rclpy
    from message_filters import ApproximateTimeSynchronizer, Subscriber
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    def to_rgb(msg):
        ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}[msg.encoding]
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * ch]
        img = img.reshape(msg.height, msg.width, ch)[..., :3]
        return np.ascontiguousarray(img[..., ::-1] if msg.encoding.startswith("bgr") else img)

    def to_depth(msg):
        if msg.encoding == "32FC1":
            d = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.step // 4)[:, : msg.width]
            return np.nan_to_num(d.copy(), nan=0.0, posinf=0.0)
        d = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[:, : msg.width]
        return d.astype(np.float32) * 0.001

    rclpy.init()
    node = rclpy.create_node("depth_completion_viewer")
    lock, state = threading.Lock(), {"pair": None, "info": None}

    def on_pair(c, d):
        with lock:
            state["pair"] = (c, d)

    subs = [Subscriber(node, Image, a.color, qos_profile=qos_profile_sensor_data),
            Subscriber(node, Image, a.depth, qos_profile=qos_profile_sensor_data)]
    sync = ApproximateTimeSynchronizer(subs, queue_size=5, slop=0.02)
    sync.registerCallback(on_pair)
    node.create_subscription(CameraInfo, a.info, lambda m: state.__setitem__("info", m),
                             qos_profile_sensor_data)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try:
        while rclpy.ok():
            with lock:
                pair, state["pair"] = state["pair"], None
            info = state["info"]
            if pair is None or info is None:
                time.sleep(0.005)
                continue
            rgb, depth = to_rgb(pair[0]), to_depth(pair[1])
            H, W = rgb.shape[:2]
            if depth.shape != (H, W):
                import cv2
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            s = W / info.width                      # camera_info may be for another resolution
            yield rgb, depth, info.k[0] * s, info.k[4] * s, info.k[2] * s, info.k[5] * s
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


# ---------------------------------------------------------------- display

def holdout_error(depth, d_t, pred, fusion_kw, fill_kw, log_s, rng):
    """AbsRel [%] and MAE [cm] below NEAR_M on hidden 32 px blocks of measured pixels, or None."""
    H, W = depth.shape
    pick = rng.random((-(-H // 32), -(-W // 32))) < 0.15
    blocks = np.kron(pick, np.ones((32, 32), bool))[:H, :W] & (depth > 0)
    hidden = torch.where(torch.from_numpy(blocks).to(d_t.device), torch.zeros_like(d_t), d_t)
    f, _ = fill(hidden, pred, GradientFusion(**fusion_kw), log_s, **fill_kw)
    f = f.cpu().numpy()
    m = blocks & (f > 0) & (depth < NEAR_M)
    if m.sum() < 50:
        return None
    err = np.abs(f[m] - depth[m])
    return 100 * float((err / depth[m]).mean()), 100 * float(err.mean())


def blueprint(evaluate):
    image = lambda path, name: rrb.Spatial2DView(origin=path, name=name)
    # same start view for all clouds: slightly behind and above the camera, looking forward
    eye = rrb.EyeControls3D(position=(0.0, -1.0, -1.5), look_target=(0.0, 0.0, 3.0), eye_up=(0.0, -1.0, 0.0))
    cloud = lambda path, name: rrb.Spatial3DView(origin="cloud", contents=path, name=name, eye_controls=eye)
    rows = [rrb.Horizontal(image("image/rgb", "RGB"), image("image/sensor", "Sensor"),
                           image("image/model", "Model"), image("image/merged", "Merged")),
            rrb.Horizontal(cloud("cloud/sensor/**", "Sensor 3D"), cloud("cloud/model/**", "Model 3D"),
                           cloud("cloud/merged/**", "Merged 3D"))]
    shares = [1, 2]
    if evaluate:
        rows.append(rrb.Horizontal(rrb.TimeSeriesView(origin="error/absrel", name="AbsRel < 3 m [%]"),
                                   rrb.TimeSeriesView(origin="error/mae", name="MAE < 3 m [cm]")))
        shares = [1, 2, 0.7]
    return rrb.Blueprint(rrb.Vertical(*rows, row_shares=shares), collapse_panels=True)


def backproject(depth, rgb, fx, fy, cx, cy, stride, max_depth):
    """Points (N, 3) in the camera frame (x right, y down, z forward) and their colours."""
    d = depth[::stride, ::stride]
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    m = (d > 0) & (d < max_depth)
    z = d[m]
    pts = np.stack([(u[m] - cx) / fx * z, (v[m] - cy) / fy * z, z], axis=1)
    return pts, rgb[::stride, ::stride][m]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="moge2-b", choices=MODELS)
    ap.add_argument("--num-tokens", type=int, default=600, help="MoGe only")
    ap.add_argument("--stream", action="store_true", help="Depth Pro 2 only: live stream mode")
    ap.add_argument("--set", nargs="*", default=[], help="parameters, e.g. edge_jump=0.02 edge_grow=4 down=4")
    ap.add_argument("--frames", help="offline: play a frames.npz instead of ROS")
    ap.add_argument("--save", help="write a .rrd recording instead of opening the viewer")
    ap.add_argument("--eval", action="store_true", help="error below 3 m on hidden blocks (costs a second fill)")
    ap.add_argument("--stride", type=int, default=3, help="every n-th pixel in the point clouds")
    ap.add_argument("--max-depth", type=float, default=8.0, help="[m] range of colours and clouds")
    ap.add_argument("--color", default="/camera/color/image_raw")
    ap.add_argument("--depth", default="/camera/depth/image_raw")
    ap.add_argument("--info", default="/camera/color/camera_info")
    a = ap.parse_args()

    params = {k: float(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    fusion_kw = {k: (int(v) if k in ("down", "iters") else v) for k, v in params.items() if k in FUSION_KEYS}
    fill_kw = {k: (int(v) if k == "edge_grow" else v) for k, v in params.items() if k not in FUSION_KEYS}

    print(f"Loading {a.model} ...")
    model = load_model(a.model, a.num_tokens, stream=a.stream)
    rr.init("depth_completion_viewer", spawn=a.save is None)
    if a.save:
        rr.save(a.save)
    rr.send_blueprint(blueprint(a.eval))
    rr.log("cloud", rr.ViewCoordinates.RDF, static=True)
    rr.log("image/merged/discarded", rr.AnnotationContext([rr.AnnotationInfo(1, "discarded", (255, 0, 255))]),
           static=True)

    fusion, log_s, rng, err = GradientFusion(**fusion_kw), None, np.random.default_rng(0), None
    frames = offline_frames(a.frames) if a.frames else live_frames(a)
    n, last = 0, time.perf_counter()
    for i, (rgb, depth, fx, fy, cx, cy) in enumerate(frames):
        fov_x = math.degrees(2 * math.atan(rgb.shape[1] / (2 * fx)))
        pred = model(rgb, fov_x)
        d_t = torch.from_numpy(depth).to(model.dev)
        filled, log_s = fill(d_t, pred, fusion, log_s, **fill_kw)
        filled = filled.cpu().numpy()
        est = pred.cpu().numpy() * math.exp(log_s)
        est = np.where(np.isfinite(est), est, 0).astype(np.float32)   # model x global scale
        measured = depth > 0
        discarded = ~measured & (filled <= 0) & (est > 0)

        rr.set_time("frame", sequence=i)
        depth_img = lambda d: rr.DepthImage(d, meter=1.0, depth_range=(0.0, a.max_depth))
        rr.log("image/rgb", rr.Image(rgb))
        rr.log("image/sensor", depth_img(depth))
        rr.log("image/model", depth_img(est))
        rr.log("image/merged/depth", depth_img(filled))
        rr.log("image/merged/discarded", rr.SegmentationImage(discarded.astype(np.uint8)))

        bp = lambda d: backproject(d, rgb, fx, fy, cx, cy, a.stride, a.max_depth)
        for name, d in (("sensor", depth), ("model", est), ("merged/measured", np.where(measured, filled, 0))):
            pts, col = bp(d)
            rr.log(f"cloud/{name}", rr.Points3D(pts, colors=col, radii=-1.5))
        pts, _ = bp(np.where(measured, 0, filled))
        rr.log("cloud/merged/filled", rr.Points3D(pts, colors=FILLED_COLOR, radii=-1.5))

        if a.eval:
            e = holdout_error(depth, d_t, pred, fusion_kw, fill_kw, log_s, rng)
            if e:
                err = e
                rr.log("error/absrel", rr.Scalars(e[0]))
                rr.log("error/mae", rr.Scalars(e[1]))

        n += 1
        now = time.perf_counter()
        if now - last >= 2.0:
            print(f"{n / (now - last):5.1f} Hz | global scale {math.exp(log_s):.3f} | "
                  f"holes {100 * (~measured).mean():.0f} % | discarded {100 * discarded.mean():.1f} %"
                  + (f" | error <3 m {err[0]:.2f} % / {err[1]:.1f} cm" if err else ""), flush=True)
            n, last = 0, now


if __name__ == "__main__":
    main()
