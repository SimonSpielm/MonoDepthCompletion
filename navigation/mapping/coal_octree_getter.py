import os, pickle, time
import threading
from pathlib import Path
from typing import Optional

import coal
import links_and_nodes as ln
import numpy as np
import odb_ln_interface.ln_interface as ln_odb
import rclpy
import trimesh
from builtin_interfaces.msg import Time as RosTime
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from std_srvs.srv import Empty

from robotutils.geom.utils import invert_transformation
from python_modules.environment_awareness.robot_interface import Robot_Interface


# Input topics
MAP_BACKEND = "nvblox"  # "nvblox" or "octomap"
MAP_BACKEND_TOPICS = {
    "nvblox": "/nvblox/occupied_points_rgb",
    "octomap": "/octomap_point_cloud_centers",
}
RAW_POINTCLOUD_TOPIC = "/camera/depth/points"
FILTERED_PC_TOPIC = "/coal_octree_getter/filtered_points"
OCTOMAP_RESET_SERVICE = "/octomap_server/reset"
NVBLOX_RESET_SERVICE = "nvblox.reset_map"
MAP_FRAME = "map"

OCTREE_RESOLUTION = 0.01
OBJECT_FILTER_MARGIN = 0.05


class CoalOctreeGetter:
    """Hold the latest map in Ws; optionally remove an object's occupied volume.

    get_octree(obj=None)                -> (coal.OcTree in Ws, timestamp)
    get_occupied_boxes_in_Wt(obj=None) -> (AABBs (N, 2, 3) in Wt, changed)
    get_prepared_octomap(obj=None)      -> (frocc.PreparedOctomap in Ws, changed)

    The object filter uses a mesh's local AABB, transformed by the object's pose
    and padded by OBJECT_FILTER_MARGIN. Unresolved objects leave the map intact.
    Prepared maps are built only when ``prepare_frocc`` is enabled; the map stays
    in Ws so it survives robot-base motion and can be reused by the FROCC backend.
    """

    def __init__(
        self,
        use_pointcloud_fast: bool = False,
        publish_filtered_pointcloud: bool = True,
        prepare_frocc: bool = False,
    ):
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=None)

        self._node = Node("coal_octree_getter")
        self._lock = threading.Lock()
        self._mesh_lock = threading.Lock()
        self.active = True
        self.prepare_frocc = prepare_frocc

        self._resolution = OCTREE_RESOLUTION
        self._map_version = 0
        self._timestamp: Optional[float] = None
        self._cloud_stamp: Optional[RosTime] = None
        self._raw_points = np.empty((0, 3), dtype=np.float64)
        self._raw_octree: Optional[coal.OcTree] = None

        # Object name -> (map version, object pose, filtered points, filtered octree)
        # The octree entry is populated only if get_octree() needs it.
        self._filtered_cache = {}
        # Object name -> (filtered points reference, base pose, AABBs in Wt)
        self._boxes_cache = {}
        # Object name -> (filtered points reference, map version, PreparedOctomap in Ws)
        self._prepared_cache = {}
        # Object type -> (local mesh AABB min, local mesh AABB max)
        self._mesh_bounds_cache = {}

        self.ln_clnt = ln.client("CoalOctomapBridge", [])
        self.odb = ln_odb.ln_interface(clnt=self.ln_clnt)
        self.robot_interface = Robot_Interface()

        self._pending_cloud = None
        self._pending_cloud_lock = threading.Lock()
        self._filtered_pc_pub = None
        if publish_filtered_pointcloud:
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._filtered_pc_pub = self._node.create_publisher(
                PointCloud2, FILTERED_PC_TOPIC, qos
            )
            self._node.create_timer(0.05, self._publish_pending_cloud)

        self._nvblox_reset_svc = None
        topic = RAW_POINTCLOUD_TOPIC if use_pointcloud_fast else MAP_BACKEND_TOPICS[MAP_BACKEND]
        self._subscription = self._node.create_subscription(
            PointCloud2, topic, self._pc_callback, 1
        )
        self._reset_client = self._node.create_client(Empty, OCTOMAP_RESET_SERVICE)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._executor.shutdown()
        self._thread.join(timeout=1.0)
        self._node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

    # --- Latest point cloud -------------------------------------------------

    def _pc_callback(self, msg: PointCloud2) -> None:
        if not self.active:
            return

        # Both the object filter and the Wt transform assume Ws == "map".
        if msg.header.frame_id.lstrip("/") != MAP_FRAME:
            self._node.get_logger().warn(
                f"Pointcloud is in '{msg.header.frame_id}', not '{MAP_FRAME}'. "
                "Transform it to map before publishing."
            )
            return

        points = point_cloud2.read_points_numpy(
            msg, field_names=["x", "y", "z"], skip_nans=True
        )
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self._update_map(points, stamp=msg.header.stamp)

    def _update_map(self, points, stamp=None, resolution=None) -> None:
        """Replace the entire map and invalidate all map-dependent caches."""
        resolution = self._resolution if resolution is None else float(resolution)
        octree = coal.makeOctree(points, resolution) if len(points) else None
        prepared = self._build_prepared_map(points, resolution) if self.prepare_frocc else None

        with self._lock:
            self._raw_points = points
            self._raw_octree = octree
            self._resolution = resolution
            self._timestamp = time.time()
            self._cloud_stamp = stamp
            self._map_version += 1
            self._filtered_cache.clear()
            self._boxes_cache.clear()
            self._prepared_cache.clear()
            if self.prepare_frocc:
                self._prepared_cache[""] = (points, self._map_version, prepared)
        return octree

    @staticmethod
    def _build_prepared_map(points, resolution):
        """Build the reusable FROCC map in Ws, or None for an empty map."""
        if not len(points):
            return None
        from frocc import PreparedOctomap

        half = 0.5 * resolution
        boxes = np.stack((points - half, points + half), axis=1)
        return PreparedOctomap.from_boxes(boxes)

    def _snapshot(self):
        """Read one consistent map generation without copying its point array."""
        with self._lock:
            if self._timestamp is None:
                return None
            return (
                self._raw_points,
                self._raw_octree,
                self._timestamp,
                self._cloud_stamp,
                self._map_version,
                self._resolution,
            )

    # --- Public map access -------------------------------------------------

    def get_octree(self, obj: Optional[str] = None):
        """Return (COAL octree in Ws, timestamp), optionally without `obj`.

        Every call also queues its associated Ws points for RViz debugging.
        Returns (None, None) until a map arrives; an empty map has no octree.
        """
        snapshot = self._snapshot()
        if snapshot is None:
            return None, None

        raw_points, raw_octree, timestamp, stamp, version, resolution = snapshot
        name = (obj or "").strip()
        points, octree = self._filtered_data(name, raw_points, raw_octree, version)

        if octree is None and len(points):
            octree = coal.makeOctree(points, resolution)
            # Preserve the built octree if this filter result is still cached.
            with self._lock:
                cached = self._filtered_cache.get(name)
                if cached is not None:
                    cached_version, cached_pose, cached_points, _ = cached
                    if cached_version == version and cached_points is points:
                        self._filtered_cache[name] = (
                            cached_version, cached_pose, cached_points, octree
                        )

        self._queue_debug_cloud(points, stamp)
        return octree, timestamp

    def get_occupied_boxes_in_Wt(self, obj: Optional[str] = None):
        """Return (boxes, changed); boxes have shape (N, 2, 3) in Wt.

        The boxes are AABBs enclosing rotated resolution-sized voxels.
        `changed` is relative to this instance's last call for the same obj.
        """
        snapshot = self._snapshot()
        if snapshot is None:
            return None, False

        raw_points, raw_octree, _, _, version, resolution = snapshot
        name = (obj or "").strip()
        points, _ = self._filtered_data(name, raw_points, raw_octree, version)
        T_Wt_Ws = invert_transformation(self.robot_interface.get_Ws_T_Wt())

        with self._lock:
            cached = self._boxes_cache.get(name)
            if cached is not None:
                cached_points, cached_pose, cached_boxes = cached
                if cached_points is points and np.allclose(
                    cached_pose, T_Wt_Ws, atol=1e-6, rtol=0.0
                ):
                    return cached_boxes, False

        R = T_Wt_Ws[:3, :3]
        t = T_Wt_Ws[:3, 3]
        centers = points @ R.T + t
        half_extents = (resolution / 2.0) * np.abs(R).sum(axis=1)
        boxes = np.stack((centers - half_extents, centers + half_extents), axis=1)
        boxes.setflags(write=False)  # Do not allow callers to change cached boxes.

        with self._lock:
            if version == self._map_version:
                self._boxes_cache[name] = (points, T_Wt_Ws.copy(), boxes)
        return boxes, True

    def get_prepared_octomap(self, obj: Optional[str] = None):
        """Return (frocc.PreparedOctomap in Ws, changed) when enabled.

        The prepared map stays in the map frame (Ws); the FROCC backend places it
        relative to the robot with ``set_octomap_pose``. It therefore stays valid
        across robot-base motion and only invalidates on map/filter changes.
        ``changed`` is relative to the previous call for the same obj.
        """
        if not self.prepare_frocc:
            return None, False

        snapshot = self._snapshot()
        if snapshot is None:
            return None, False

        raw_points, raw_octree, _, _, version, resolution = snapshot
        name = (obj or "").strip()
        points, _ = self._filtered_data(name, raw_points, raw_octree, version)

        with self._lock:
            cached = self._prepared_cache.get(name)
            if cached is not None:
                cached_points, cached_version, prepared = cached
                if cached_points is points and cached_version == version:
                    return prepared, False

        from frocc import PreparedOctomap

        if len(points):
            half = 0.5 * resolution
            boxes_in_Ws = np.stack((points - half, points + half), axis=1)
            prepared = PreparedOctomap.from_boxes(boxes_in_Ws)
        else:
            prepared = None

        with self._lock:
            if version == self._map_version:
                self._prepared_cache[name] = (points, version, prepared)
        return prepared, True

    # --- Object filtering --------------------------------------------------

    def _filtered_data(self, name, raw_points, raw_octree, version):
        """Return (points, octree_or_None) for this map and object's current pose."""
        if not name:
            return raw_points, raw_octree

        pose = self.robot_interface.get_Ws_T_Obj(name)
        if pose is None:
            return raw_points, raw_octree
        pose = np.asarray(pose, dtype=np.float64)

        with self._lock:
            cached = self._filtered_cache.get(name)
            if cached is not None:
                cached_version, cached_pose, cached_points, cached_octree = cached
                if cached_version == version and np.array_equal(cached_pose, pose):
                    return cached_points, cached_octree

        bounds = self._get_mesh_bounds(name.split("$", 1)[0])
        if bounds is None:
            return raw_points, raw_octree

        local_min, local_max = bounds
        points_in_obj = (raw_points - pose[:3, 3]) @ pose[:3, :3]
        inside = np.all(
            (points_in_obj >= local_min - OBJECT_FILTER_MARGIN)
            & (points_in_obj <= local_max + OBJECT_FILTER_MARGIN),
            axis=1,
        )
        points = raw_points[~inside] if inside.any() else raw_points
        octree = raw_octree if points is raw_points else None

        with self._lock:
            if version == self._map_version:
                self._filtered_cache[name] = (version, pose.copy(), points, octree)
        return points, octree

    def _get_mesh_bounds(self, obj_type):
        with self._mesh_lock:
            if obj_type in self._mesh_bounds_cache:
                return self._mesh_bounds_cache[obj_type]

            files = self.odb.list_property(obj_type, "collision")
            main = os.environ.get("MAIN")
            if not files or not main:
                self._node.get_logger().warn(
                    f"Missing collision geometry for '{obj_type}' (check ODB / MAIN)."
                )
                return None

            mesh_path = Path(main) / "odb" / obj_type / "geometry" / files[0]
            try:
                bounds = np.asarray(
                    trimesh.load(str(mesh_path), force="mesh").bounds,
                    dtype=np.float64,
                )
            except (OSError, ValueError, TypeError) as exc:
                self._node.get_logger().warn(f"Cannot load '{mesh_path}': {exc}")
                return None

            local_bounds = (bounds[0], bounds[1])
            self._mesh_bounds_cache[obj_type] = local_bounds
            return local_bounds

    def clear_object_mesh_cache(self) -> None:
        """Reload mesh bounds on the next request, e.g. after editing an ODB mesh."""
        with self._mesh_lock:
            self._mesh_bounds_cache.clear()
        with self._lock:
            self._filtered_cache.clear()
            self._boxes_cache.clear()
            self._prepared_cache.clear()

    # --- ROS service, persistence, RViz -----------------------------------

    def reset_octomap(self, timeout_sec: float = 2.0) -> bool:
        """Reset the active map backend; does not clear this instance's map."""
        if MAP_BACKEND == "nvblox":
            return self._reset_nvblox()
        return self._reset_ros_octomap(timeout_sec)

    def _reset_nvblox(self) -> bool:
        try:
            if self._nvblox_reset_svc is None:
                self._nvblox_reset_svc = self.ln_clnt.get_service(
                    NVBLOX_RESET_SERVICE, "navigation/reset_map_svc"
                )
            self._nvblox_reset_svc.req.trigger = 1
            self._nvblox_reset_svc.call()
            if self._nvblox_reset_svc.resp.error_message:
                raise RuntimeError(self._nvblox_reset_svc.resp.error_message)
            return True
        except Exception as e:
            self._node.get_logger().warn(f"nvblox reset failed: {e}")
            return False

    def _reset_ros_octomap(self, timeout_sec: float = 2.0) -> bool:
        """Reset external octomap_server."""
        if not self._reset_client.wait_for_service(timeout_sec=timeout_sec):
            self._node.get_logger().warn("Octomap reset service unavailable.")
            return False

        future = self._reset_client.call_async(Empty.Request())
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                self._node.get_logger().warn("Octomap reset timed out.")
                return False
            time.sleep(0.01)

        if not future.done() or future.exception() is not None:
            self._node.get_logger().error("Octomap reset failed.")
            return False
        return True

    def save_octree(self, directory: str) -> Path:
        """Save raw Ws points and resolution as directory/octree.pickle."""
        snapshot = self._snapshot()
        if snapshot is None:
            raise RuntimeError("No map has been received yet.")
        points, _, _, _, _, resolution = snapshot
        path = Path(directory).expanduser() / "octree.pickle"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump({"points_in_Ws": points, "resolution": resolution}, file)
        return path

    def load_octree(self, path: str):
        """Load a map created by save_octree(). Use only trusted pickle files."""
        file_path = Path(path).expanduser()
        if file_path.is_dir():
            file_path /= "octree.pickle"
        with file_path.open("rb") as file:
            data = pickle.load(file)

        points = np.asarray(data["points_in_Ws"], dtype=np.float64).reshape(-1, 3)
        return self._update_map(points, resolution=float(data["resolution"]))

    def _queue_debug_cloud(self, points, stamp):
        if self._filtered_pc_pub is None:
            return
        with self._pending_cloud_lock:
            self._pending_cloud = (points, stamp)  # Keep only the newest request.

    def _publish_pending_cloud(self):
        with self._pending_cloud_lock:
            pending = self._pending_cloud
            self._pending_cloud = None
        if pending is None:
            return

        points, stamp = pending
        header = Header()
        header.frame_id = MAP_FRAME
        header.stamp = stamp or self._node.get_clock().now().to_msg()
        message = point_cloud2.create_cloud_xyz32(
            header, points.astype(np.float32, copy=False)
        )
        self._filtered_pc_pub.publish(message)
