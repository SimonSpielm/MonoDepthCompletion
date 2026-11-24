"""Interface to Generic7R and SMMPL for EDAN and MAYA.

Generic7R/SMMPL operate on the last seven arm joints. For EDAN the
additional axis is kept fixed and added back to the returned configurations.
"""

import os
import time

import numpy as np
import generic7rrobot as gr

from robotutils.ros.markers import RvizMarkers
from python_modules.environment_awareness.arm_collision_model import Arm_Collision_Model
from navigation.wheelchair_interface.wheelchair_utils import stopwatch


class SMMInterface:
    """Connect the reduced Pinocchio robot model to Generic7R and SMMPL."""

    def __init__(self, pin_arm=None, tol=2.75e-5, markers=None):
        self.pin_arm = pin_arm if pin_arm is not None else Arm_Collision_Model()
        self.cfg = self.pin_arm.cfg
        self.robot_interface = self.pin_arm.robot_interface
        self.markers = markers if markers is not None else RvizMarkers(namespace="SMM")

        os.environ["GENERIC7RROBOT_URDF"] = str(self.cfg.urdf_generic7r_path)
        self.smm_robot = gr.Generic7RRobot.create([-np.pi] * 7, [np.pi] * 7)

        self.tol = tol
        self.smmpl_init = False

    # Joint configurations

    def nullspace_optimization_init(self) -> None:
        if self.smmpl_init:
            return
        from smmpl_python.smmpl_server import SMMPLServer

        self.smmpl = SMMPLServer("C++ service provider")
        self.smmpl.set_joint_limits(
            self.pin_arm.model.lowerPositionLimit[-7:].tolist(),
            self.pin_arm.model.upperPositionLimit[-7:].tolist(),
        )
        self.smmpl_init = True

    def get_current_robot_q(self) -> np.ndarray:
        return self.pin_arm.get_current_robot_q()

    def _normalize_q(self, q=None) -> np.ndarray:
        # The collision model already handles arm-only and full robot q.
        return self.pin_arm.normalize_q(q)

    @staticmethod
    def _split_q_full(q_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split [extra axes..., 7 arm axes]."""
        q_full = np.asarray(q_full, dtype=float).reshape(-1)
        if q_full.size < 7:
            raise ValueError(f"Expected at least 7 joints, got {q_full.size}")
        return q_full[:-7], q_full[-7:]

    @staticmethod
    def _expand_smm_qs(smm_q7: np.ndarray, q_extra: np.ndarray) -> np.ndarray:
        """Prepend EDAN's fixed extra axes to (N, 7) SMM/Planner results."""
        arm = np.asarray(smm_q7, dtype=float)
        extra = np.asarray(q_extra, dtype=float).reshape(-1)

        if arm.size == 0:
            return np.empty((0, extra.size + 7), dtype=float)
        if arm.ndim == 1:
            arm = arm.reshape(1, -1)
        if arm.ndim != 2 or arm.shape[1] != 7:
            raise ValueError(f"SMM result must have shape (N, 7), got {arm.shape}")

        if not extra.size:
            return arm
        return np.hstack((np.tile(extra, (len(arm), 1)), arm))

    # SMM

    def get_closest_point_on_manifold_to_current_elbow(self, smm, take_0=False, q_ref=None):
        """Choose the SMM point closest to the live or specified elbow pose."""
        if take_0:
            return 0

        smm = np.asarray(smm, dtype=float)
        if smm.ndim != 2 or not len(smm):
            raise ValueError("Cannot select a point from an empty SMM")

        # For future/planned poses use FK of q_ref, otherwise the measured elbow.
        if q_ref is None:
            start_elbow = self.robot_interface.get_elbow_positon()
        else:
            start_elbow = self.pin_arm.get_elbow_position_for_q(self._normalize_q(q_ref))

        finite = np.isfinite(smm).all(axis=1)
        if not finite.any():
            raise ValueError("All SMM configurations contain non-finite values")
        indices = np.flatnonzero(finite)
        elbows = self.get_elbow_positions(smm[finite])
        distances = np.sum((elbows - start_elbow) ** 2, axis=1)
        return int(indices[np.argmin(distances)])

    @staticmethod
    def classify_manifold_index(q):
        q2, q4, q6 = np.asarray(q)[-7:][[1, 3, 5]]
        return int(1 + (q4 < 0) * 4 + (q6 < 0) * 2 + (q2 < 0))

    def get_smm_qs(self, q=None):
        """Compute SMM configurations; the additional EDAN axis stays fixed."""
        q_full = self._normalize_q(q)
        q_extra, q_arm7 = self._split_q_full(q_full)

        smm_q7 = self.smm_robot.compute_smm(q_arm7, self.tol)
        return self._expand_smm_qs(smm_q7, q_extra)

    def get_preimage(self, q=None):
        q_full = self._normalize_q(q)
        q_extra, q_arm7 = self._split_q_full(q_full)

        fk = self.smm_robot.get_fk_solution(q_arm7)
        self.smm_robot.compute_preimage(fk)
        branches = [
            np.asarray(branch, dtype=float).reshape(-1, 7)
            for branch in self.smm_robot.get_preimage()
        ]
        smm_q7 = np.concatenate(branches, axis=0) if branches else np.empty((0, 7))
        return self._expand_smm_qs(smm_q7, q_extra)

    def get_elbow_positions(self, qs):
        qs = np.asarray(qs, dtype=float)
        if not qs.size:
            return np.empty((0, 3))
        return np.asarray([self.pin_arm.get_elbow_position_for_q(q) for q in qs])

    def filter_smm(self, smm, q_ref=None):
        """Split the SMM ring into accessible, feasible and infeasible masks."""
        smm = np.asarray(smm, dtype=float)
        if smm.ndim != 2 or smm.shape[1] != self.pin_arm.model.nq:
            raise ValueError(f"Expected (N, {self.pin_arm.model.nq}) SMM, got {smm.shape}")

        n = len(smm)
        if not n:
            empty = np.zeros(0, dtype=bool)
            return empty.copy(), empty.copy(), empty.copy()

        feas = (
            np.isfinite(smm).all(axis=1)
            & (smm >= self.pin_arm.model.lowerPositionLimit).all(axis=1)
            & (smm <= self.pin_arm.model.upperPositionLimit).all(axis=1)
        )
        acc = np.zeros(n, dtype=bool)
        if feas.any():
            start_idx = self.get_closest_point_on_manifold_to_current_elbow(smm, q_ref=q_ref)
            if feas[start_idx]:
                acc[start_idx] = True
                for direction in (-1, 1):
                    i = (start_idx + direction) % n
                    while i != start_idx and feas[i]:
                        acc[i] = True
                        i = (i + direction) % n

        return acc, feas & ~acc, ~feas

    def plot_elbow_positions(self, collision_free, accesable, feasible, infeasible):
        """Visualize the four elbow-position groups in RViz/Meshcat."""
        T = (np.eye(4) if type(self.markers).__name__ == "MeshcatMarkers"
             else self.robot_interface.get_Ws_T_Wt())

        groups = (
            (collision_free, [0.0, 1.0, 0.0], "collision_free"),
            (accesable, [1.0, 1.0, 0.0], "accessible_smm"),
            (feasible, [1.0, 0.5, 0.0], "feasible_smm"),
            (infeasible, [1.0, 0.0, 0.0], "infeasible_smm"),
        )
        for points, color, name in groups:
            points = np.asarray(points, dtype=float).reshape(-1, 3)
            if len(points):
                points_h = np.column_stack((points, np.ones(len(points))))
                centers = (T @ points_h.T).T[:, :3]
            else:
                centers = [[10000, 0, 0]]  # clear previous marker
            self.markers.set_marker(
                name, marker_type="Spheres", centers=centers, color=color, radii=0.01
            )

    def check_collisionfree(self, accessible_mask, smm_result, q_ref=None):
        """Walk in both directions until a collision or inaccessible point."""
        smm_result = np.asarray(smm_result, dtype=float)
        n = len(smm_result)
        accessible_mask = np.asarray(accessible_mask, dtype=bool)
        if accessible_mask.shape != (n,):
            raise ValueError("Accessibility mask does not match SMM")

        collision_free = np.zeros(n, dtype=bool)
        if not n or not accessible_mask.any():
            return collision_free, accessible_mask.copy()

        self.pin_arm.update_env_octree()
        start_idx = self.get_closest_point_on_manifold_to_current_elbow(smm_result, q_ref=q_ref)
        if not accessible_mask[start_idx]:
            return collision_free, accessible_mask.copy()

        def is_cf(i):
            return self.pin_arm.check_collisionfree_q(smm_result[i], update_octomap=False)

        # In the original code the start was always assumed collision-free.
        if not is_cf(start_idx):
            return collision_free, accessible_mask.copy()
        collision_free[start_idx] = True

        for direction in (-1, 1):
            i = (start_idx + direction) % n
            while i != start_idx and accessible_mask[i] and is_cf(i):
                collision_free[i] = True
                i = (i + direction) % n

        return collision_free, accessible_mask & ~collision_free

    def get_collision_free_mid_attractor(self, collision_free_points, q_ref=None):
        """Return the elbow point in the middle of the collision-free angular arc."""
        points = np.asarray(collision_free_points, dtype=float).reshape(-1, 3)
        n = len(points)
        if not n:
            raise ValueError("No collision-free elbow points")
        if n == 1:
            return points[0].copy()

        base = np.zeros(3)
        if q_ref is None:
            ee_pos = self.pin_arm.get_current_ee_pos()
        else:
            ee_pos = self.pin_arm.get_ee_pose_for_q(self._normalize_q(q_ref))[:3, 3]

        axis = np.asarray(ee_pos, dtype=float) - base
        length = np.linalg.norm(axis)
        if length < 1e-12:
            return points[n // 2].copy()
        axis /= length

        u = None
        for point in points:
            candidate = point - base
            candidate -= np.dot(candidate, axis) * axis
            length = np.linalg.norm(candidate)
            if length >= 1e-12:
                u = candidate / length
                break
        if u is None:
            return points[n // 2].copy()
        v = np.cross(axis, u)

        offsets = points - base
        projected = offsets - np.outer(offsets @ axis, axis)
        angles = np.arctan2(projected @ v, projected @ u)
        ordered = np.sort(angles)
        gaps = np.diff(np.r_[ordered, ordered[0] + 2 * np.pi])
        k = int(np.argmax(gaps))

        arc_start = ordered[(k + 1) % n]
        arc_end = ordered[k]
        middle = (arc_start + 0.5 * ((arc_end - arc_start) % (2 * np.pi))) % (2 * np.pi)
        distances = np.abs((angles - middle + np.pi) % (2 * np.pi) - np.pi)
        return points[int(np.argmin(distances))].copy()

    def plot_attractor(self, p):
        T = (np.eye(4) if type(self.markers).__name__ == "MeshcatMarkers"
             else self.robot_interface.get_Ws_T_Wt())
        p_tf = T @ np.r_[np.asarray(p, dtype=float).reshape(3), 1.0]
        self.markers.set_marker(
            "elbow_attractor", marker_type="Spheres", centers=[p_tf[:3]],
            color=[0.6, 0.0, 0.8], radii=0.02,
        )

    # SMMPL planning

    def _planner_collision_flags(self, configs, start_arm7, q_extra):
        """Check only finite, joint-limit-feasible samples of the starting branch."""
        configs = np.asarray(configs, dtype=float).reshape(-1, 7)
        in_collision = np.ones(len(configs), dtype=np.int8)
        if not len(configs):
            return in_collision

        full_configs = self._expand_smm_qs(configs, q_extra)
        valid = (
            np.isfinite(full_configs).all(axis=1)
            & (np.sign(configs[:, 3]) == np.sign(start_arm7[3]))
            & (full_configs >= self.pin_arm.model.lowerPositionLimit).all(axis=1)
            & (full_configs <= self.pin_arm.model.upperPositionLimit).all(axis=1)
        )
        indices = np.flatnonzero(valid)
        if len(indices):
            self.pin_arm.update_env_octree()
            in_collision[indices] = self.pin_arm.is_in_collision_batch(full_configs[indices])
        return in_collision

    def smmpl_connectivity_check(
        self, viapoints: list, start_config: np.ndarray
    ) -> list[tuple[int, np.ndarray, int, np.ndarray]]:
        """Check SMMPL connections; viapoints are given in Wt."""
        self.nullspace_optimization_init()
        q_full = self._normalize_q(start_config)
        q_extra, q_arm7 = self._split_q_full(q_full)

        J2_T_W = self.pin_arm.get_J2_T_W(q_full)
        viapoints_j2 = [J2_T_W @ np.asarray(tf) for tf in viapoints]

        self.smmpl.start_plan_local(q_arm7, viapoints_j2)
        in_collision = self._planner_collision_flags(
            self.smmpl.get_segments(), q_arm7, q_extra
        )
        self.smmpl.send_collisions(in_collision.tolist())

        connections = self.smmpl.compute_segment_connections(
            q_arm7, viapoints_j2, start_planning=False, v2=True
        )
        return [
            (
                edge[0], self._expand_smm_qs(edge[1], q_extra)[0],
                edge[2], self._expand_smm_qs(edge[3], q_extra)[0],
            )
            for edge in connections
        ]

    @stopwatch
    def smmpl_trajectory(self, viapoints: list, start_config=None) -> list[np.ndarray]:
        """Plan in Wt; the first viapoint describes the current pose."""
        self.nullspace_optimization_init()
        q_full = self._normalize_q(start_config)
        q_extra, q_arm7 = self._split_q_full(q_full)

        J2_T_W = self.pin_arm.get_J2_T_W(q_full)
        viapoints_j2 = [J2_T_W @ np.asarray(tf) for tf in viapoints]

        self.smmpl.start_plan_local(q_arm7, viapoints_j2[1:])
        t0 = time.perf_counter()
        in_collision = self._planner_collision_flags(
            self.smmpl.get_segments(), q_arm7, q_extra
        )
        self.smmpl.send_collisions(in_collision.tolist())
        print(f"{int((in_collision == 0).sum())} collision-free planner samples "
              f"in {time.perf_counter() - t0:.4f}s")

        q_path = self.smmpl.return_plan_local_smmpl(v2=True)
        return [self._expand_smm_qs(q_arm7, q_extra)[0] for q_arm7 in q_path]

    # Attractors and midpoint selection

    def get_collision_free_arc_indices(self, collision_free) -> np.ndarray:
        """Ordered indices of the contiguous collision-free SMM arc.

        The SMM is cyclic, so the free mask may wrap around the array end.
        Start at the first free index after a blocked gap and walk forward.
        """
        mask = np.asarray(collision_free, dtype=bool)
        n = len(mask)
        if not n or not mask.any():
            return np.empty(0, dtype=int)
        if mask.all():
            return np.arange(n, dtype=int)

        starts = np.flatnonzero(mask & ~np.roll(mask, 1))
        start = int(starts[0])
        indices = []
        i = start
        while mask[i]:
            indices.append(i)
            i = (i + 1) % n
            if i == start:
                break
        return np.asarray(indices, dtype=int)

    def get_highest_v_iso_attractor(
        self,
        free_qs,
        free_elbows,
        polytope: str = "strong",
        inner_fraction: float = 0.7,
    ):
        """Collision-free elbow point with the highest v_iso in the inner fraction."""
        n = len(free_qs)
        if n == 0:
            return None

        fraction = float(np.clip(inner_fraction, 0.0, 1.0))
        keep = n if fraction >= 1.0 else max(1, int(round(n * fraction)))
        start = (n - keep) // 2
        free_qs = free_qs[start:start + keep]
        free_elbows = free_elbows[start:start + keep]

        scores = np.array([
            self.pin_arm.get_v_iso(q_i, polytope=polytope) for q_i in free_qs
        ])
        best = int(np.argmax(scores))
        return free_elbows[best].copy() if scores[best] > 0.0 else None

    def get_attractor(
        self,
        q=None,
        plot: bool = True,
        use_v_iso: bool = True
    ):
        """Find a collision-free elbow attractor for the current EE pose.

        ``use_v_iso=True`` selects the collision-free SMM point with the highest
        isotropic velocity; ``False`` keeps the original midpoint of the arc.
        """
        reference_q = self._normalize_q(q)

        smm_result = self.get_smm_qs(reference_q)
        accessible, feasible, infeasible = self.filter_smm(smm_result, q_ref=reference_q)
        collision_free, accessible = self.check_collisionfree(
            accessible, smm_result, q_ref=reference_q
        )
        if not collision_free.any():
            raise ValueError("No collision-free SMM segment found")

        arc_indices = self.get_collision_free_arc_indices(collision_free)
        free_qs = smm_result[arc_indices]
        free_elbows = self.get_elbow_positions(free_qs)
        attractor = None
        if use_v_iso:
            attractor = self.get_highest_v_iso_attractor(free_qs, free_elbows)
        if attractor is None:
            attractor = self.get_collision_free_mid_attractor(free_elbows, q_ref=reference_q)

        if plot:
            self.plot_elbow_positions(
                free_elbows,
                self.get_elbow_positions(smm_result[accessible]),
                self.get_elbow_positions(smm_result[feasible]),
                self.get_elbow_positions(smm_result[infeasible]),
            )
            self.plot_attractor(attractor)

        return attractor

    def get_collision_free_mid_config(self, q=None) -> tuple[int, np.ndarray]:
        """Return the midpoint index and full reduced joint configuration."""
        reference_q = self._normalize_q(q)
        smm_result = self.get_smm_qs(reference_q)
        if not len(smm_result):
            raise ValueError("No SMM result found")

        accessible, _, _ = self.filter_smm(smm_result, q_ref=reference_q)
        collision_free, _ = self.check_collisionfree(
            accessible, smm_result, q_ref=reference_q
        )
        if not collision_free.any():
            raise ValueError("No collision-free segment found")

        n = len(smm_result)
        start_idx = self.get_closest_point_on_manifold_to_current_elbow(
            smm_result, q_ref=reference_q
        )
        if not collision_free[start_idx]:
            raise ValueError("Current/start point is not collision-free")

        if collision_free.all():
            return start_idx, smm_result[start_idx].copy()

        left_indices, right_indices = [], []
        i = (start_idx - 1) % n
        while i != start_idx and collision_free[i]:
            left_indices.append(i)
            i = (i - 1) % n

        i = (start_idx + 1) % n
        while i != start_idx and collision_free[i]:
            right_indices.append(i)
            i = (i + 1) % n

        segment_indices = left_indices[::-1] + [start_idx] + right_indices
        mid_idx = int(segment_indices[len(segment_indices) // 2])
        return mid_idx, smm_result[mid_idx].copy()
