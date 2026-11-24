"""
Step 5: nvblox TSDF mapping.

Integrates the filtered depth of pointcloud_publisher.py (same data as
/camera/filtered/points, but as image: nvblox_torch integrates depth images
projectively). The camera pose is looked up in TF at the frame's original stamp.
Publishes the occupied voxels around the camera as /nvblox/occupied_points_rgb.
"""
import links_and_nodes as ln
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data

import tf2_ros
from tf2_ros import TransformException

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header

from cv_bridge import CvBridge

from robotutils.geom.se3 import pose2tf

from nvblox_torch.mapper import Mapper, QueryType
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.sensor import Sensor

def make_local_offsets(half_xy: float, z_min: float, z_max: float, vs: float) -> torch.Tensor:
    """
    Offsets der Voxelzentren relativ zu einem Grid-Ursprung, der auf einem
    Vielfachen von vs liegt. Zentren bei (i + 0.5) * vs -> exakt nvblox-Voxelzentren.
    """
    n_xy = int(round(half_xy / vs))
    ixy = torch.arange(-n_xy, n_xy, dtype=torch.float32)
    iz0, iz1 = int(np.floor(z_min / vs)), int(np.ceil(z_max / vs))
    iz = torch.arange(iz0, iz1, dtype=torch.float32)

    X, Y, Z = torch.meshgrid(ixy, ixy, iz, indexing="ij")
    offs = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)
    return (offs + 0.5) * vs          # z ist hier bereits absolut (global_frame)


def create_pointcloud2_xyzrgb(points_xyz, colors_rgb, frame_id, stamp) -> PointCloud2:
    header = Header()
    header.frame_id = frame_id
    header.stamp = stamp

    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    colors_rgb = np.asarray(colors_rgb, dtype=np.uint8)

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got shape {points_xyz.shape}")
    if colors_rgb.shape != (points_xyz.shape[0], 3):
        raise ValueError(f"Expected Nx3 colors, got shape {colors_rgb.shape}")

    n = points_xyz.shape[0]

    # pack RGB into 24 bits of a float32 (PCL/RViz convention)
    r = colors_rgb[:, 0].astype(np.uint32)
    g = colors_rgb[:, 1].astype(np.uint32)
    b = colors_rgb[:, 2].astype(np.uint32)
    rgb_packed = ((r << 16) | (g << 8) | b).view(np.float32)

    data = np.empty(n, dtype=np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")]
    ))
    data["x"] = points_xyz[:, 0]
    data["y"] = points_xyz[:, 1]
    data["z"] = points_xyz[:, 2]
    data["rgb"] = rgb_packed

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    return PointCloud2(
        header=header,
        height=1,
        width=n,
        is_bigendian=False,
        is_dense=True,
        point_step=16,
        row_step=16 * n,
        data=data.tobytes(),
        fields=fields,
    )


class TsdfOccupiedColoredPointCloud(Node):
    def __init__(self):
        super().__init__("nvblox_tsdf_occupied_colored_pc")

        # -------- parameters --------
        self.declare_parameter("depth_topic", "camera/filtered/depth/image_raw")   # pointcloud_publisher.py
        self.declare_parameter("camera_info_topic", "camera/completed/camera_info")
        self.declare_parameter("rgb_topic", "camera/completed/color/image_raw")   # muss auf Depth registriert sein!

        self.declare_parameter("camera_frame", "camera")
        self.declare_parameter("global_frame", "map")

        self.declare_parameter("voxel_size", 0.02)
        self.declare_parameter("publish_hz", 10.0)

        # lokales Query-Fenster um die Kamera (Map selbst ist unbegrenzt)
        self.declare_parameter("query_half_extent_xy", 2.0)   # -> 4 x 4 m
        self.declare_parameter("query_z_range", [-0.5, 2.0])  # absolut im global_frame

        # occupied: (w > w_min) & (dist < occ_band_vox * voxel_size)
        #   unknown (w == 0) zählt als frei
        #   occ_band_vox = 0.5 -> auch der Voxel knapp VOR der Oberfläche zählt (keine Lücken bei dünnen Objekten)
        #   occ_band_vox = 0.0 -> nur Voxel hinter der Oberfläche
        self.declare_parameter("w_min", 0.5)
        self.declare_parameter("occ_band_vox", 0.5)

        self.declare_parameter("max_points", 300000)
        self.declare_parameter("tf_timeout_s", 0.02)
        self.declare_parameter("pointcloud_topic", "/nvblox/occupied_points_rgb")

        gp = lambda n: self.get_parameter(n).value
        self.depth_topic = gp("depth_topic")
        self.info_topic = gp("camera_info_topic")
        self.rgb_topic = gp("rgb_topic")
        self.camera_frame = gp("camera_frame")
        self.global_frame = gp("global_frame")
        self.voxel_size = float(gp("voxel_size"))
        self.publish_hz = float(gp("publish_hz"))
        self.half_xy = float(gp("query_half_extent_xy"))
        z_min, z_max = (float(v) for v in gp("query_z_range"))
        self.w_min = float(gp("w_min"))
        self.occ_thresh_m = float(gp("occ_band_vox")) * self.voxel_size
        self.max_points = int(gp("max_points"))
        self.tf_timeout_s = float(gp("tf_timeout_s"))
        self.pc_topic = gp("pointcloud_topic")

        # -------- TF --------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # -------- ROS IO --------
        self.bridge = CvBridge()
        self.sub_info = self.create_subscription(CameraInfo, self.info_topic, self.on_info, 10)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.on_depth, qos_profile_sensor_data)
        self.sub_rgb = None
        if self.rgb_topic:
            self.sub_rgb = self.create_subscription(Image, self.rgb_topic, self.on_rgb, qos_profile_sensor_data)
        self.pub_pc = self.create_publisher(PointCloud2, self.pc_topic, 1)

        # -------- nvblox --------
        self.sensor: Sensor | None = None
        self.mapper = Mapper(
            voxel_sizes_m=[self.voxel_size],
            integrator_types=ProjectiveIntegratorType.TSDF,
        )

        #------ reset map service -----
        self.ln_clnt = ln.client("NvbloxResetService", [])
        reset_svc = self.ln_clnt.get_service_provider("nvblox.reset_map", "navigation/reset_map_svc")
        reset_svc.set_handler(self._svc_reset_map)
        reset_svc.do_register()

        # lokale Offsets einmal auf der GPU vorberechnen
        self.local_offsets = make_local_offsets(self.half_xy, z_min, z_max, self.voxel_size).to("cuda")
        self.get_logger().info(f"Query window: {self.local_offsets.shape[0] / 1e6:.1f} Mio. points")

        # Zustand
        self.last_rgb_msg: Image | None = None
        self.last_T_wc: torch.Tensor | None = None
        self.last_stamp = None
        self.fx = self.fy = self.cx = self.cy = None
        self._depth_gpu: torch.Tensor | None = None
        self._depth_cpu_pinned: torch.Tensor | None = None

        self.timer = self.create_timer(1.0 / max(self.publish_hz, 1e-6), self.on_step)

    # ------------------------------------------------------------------
    def on_info(self, msg: CameraInfo):
        K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.fx, self.fy, self.cx, self.cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        self.sensor = Sensor.from_camera(fu=self.fx, fv=self.fy, cu=self.cx, cv=self.cy,
                                         width=int(msg.width), height=int(msg.height))
        self.get_logger().info("CameraInfo OK")
        self.destroy_subscription(self.sub_info)

    def on_rgb(self, msg: Image):
        self.last_rgb_msg = msg

    def lookup_T_global_cam(self, stamp):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.global_frame, self.camera_frame, stamp,
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout_s))
        except TransformException:
            return None
        t = tf_msg.transform.translation
        r = tf_msg.transform.rotation
        pose = np.array([t.x, t.y, t.z, r.x, r.y, r.z, r.w], dtype=np.float64)
        return torch.from_numpy(pose2tf(pose).astype(np.float32))

    def _ensure_depth_buffers(self, h: int, w: int):
        if self._depth_gpu is None:
            self._depth_cpu_pinned = torch.empty((h, w), dtype=torch.float32, pin_memory=True)
            self._depth_gpu = torch.empty((h, w), dtype=torch.float32, device="cuda")

    # ---- Integration: jedes Tiefenbild genau einmal ----
    def on_depth(self, msg: Image):
        if self.sensor is None or msg.header.stamp == self.last_stamp:
            return
        T_wc = self.lookup_T_global_cam(msg.header.stamp)
        if T_wc is None:
            return

        depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        self._ensure_depth_buffers(*depth_mm.shape[:2])
        self._depth_cpu_pinned.copy_(torch.from_numpy(depth_mm.astype(np.float32) * 0.001))
        self._depth_gpu.copy_(self._depth_cpu_pinned, non_blocking=True)

        self.mapper.add_depth_frame(self._depth_gpu, T_wc, self.sensor, mapper_id=0)
        self.last_stamp = msg.header.stamp
        self.last_T_wc = T_wc

    # ---- Abfrage + Publish ----
    def _colorize(self, pts: np.ndarray, rgb_img: np.ndarray, T_wc: torch.Tensor) -> np.ndarray:
        """Nur Punkte im aktuellen Sichtfeld bekommen Farbe, der Rest grau."""
        colors = np.full((pts.shape[0], 3), 180, dtype=np.uint8)
        H, W = rgb_img.shape[:2]
        T_cw = torch.linalg.inv(T_wc)
        p = torch.from_numpy(pts)
        pc = p @ T_cw[:3, :3].T + T_cw[:3, 3]
        Z = pc[:, 2]
        valid = Z > 0.1
        u = (self.fx * pc[:, 0] / Z + self.cx).long()
        v = (self.fy * pc[:, 1] / Z + self.cy).long()
        valid &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        vm = valid.numpy()
        if vm.any():
            colors[vm] = rgb_img[v[valid].numpy(), u[valid].numpy()]
        return colors

    def _svc_reset_map(self, svc, req, resp):
        try:
            self.mapper.clear(mapper_id=0)
            self.get_logger().info("nvblox map cleared")
            resp.error_message = ""
        except Exception as e:
            resp.error_message = str(e)
        finally:
            svc.respond()

    def on_step(self):
        self.ln_clnt.wait_and_handle_service_group_requests(None, 0)

        if self.last_T_wc is None:
            return
        T_wc = self.last_T_wc

        # Grid-Ursprung auf Voxelraster einrasten (nur xy, z ist absolut)
        vs = self.voxel_size
        cx = round(float(T_wc[0, 3]) / vs) * vs
        cy = round(float(T_wc[1, 3]) / vs) * vs
        shift = torch.tensor([cx, cy, 0.0], device="cuda")
        query_pts = self.local_offsets + shift

        out = self.mapper.query_layer(QueryType.TSDF, query_pts, mapper_id=0)
        dist, wgt = out[:, 0], out[:, 1]
        occupied = (wgt > self.w_min) & (dist < self.occ_thresh_m)   # unknown (wgt=0) -> frei
        idx = torch.nonzero(occupied, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return

        if idx.numel() > self.max_points:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[: self.max_points]]

        pts = query_pts.index_select(0, idx).cpu().numpy()

        colors = np.full((pts.shape[0], 3), 180, dtype=np.uint8)
        if self.last_rgb_msg is not None and self.fx is not None:
            try:
                rgb_img = self.bridge.imgmsg_to_cv2(self.last_rgb_msg, desired_encoding="rgb8")
                colors = self._colorize(pts, rgb_img, T_wc)
            except Exception as e:
                self.get_logger().warn(f"Colorize failed: {e}", throttle_duration_sec=5.0)

        self.pub_pc.publish(create_pointcloud2_xyzrgb(pts, colors, self.global_frame,
                                                      self.get_clock().now().to_msg()))


def main():
    rclpy.init()
    node = TsdfOccupiedColoredPointCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
