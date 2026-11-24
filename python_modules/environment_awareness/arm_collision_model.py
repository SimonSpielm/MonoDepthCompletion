#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from threading import RLock
from typing import Optional, Tuple

import numpy as np
import pinocchio as pin
print("Pinocchio version:", pin.__version__, pin.__file__)

from robotutils.geom.utils import invert_transformation

from python_modules.environment_awareness.robot_interface import Robot_Interface
from python_modules.environment_awareness.robot_collision_cfg import (
    RobotCollisionConfig,
    get_robot_collision_config,
)


def _compute_joints_to_lock_from_vmask(full_model: pin.Model, vmask: np.ndarray) -> list[int]:
    """Return joint IDs whose velocity coordinates are all inactive."""
    if vmask.size != full_model.nv:
        raise ValueError(
            f"active_vmask_full has len={vmask.size}, "
            f"but full_model.nv={full_model.nv}"
        )

    active = np.asarray(vmask, dtype=bool)
    locked = []
    for joint_id in range(1, full_model.njoints):
        joint = full_model.joints[joint_id]
        if joint.nv and not active[joint.idx_v : joint.idx_v + joint.nv].any():
            locked.append(joint_id)
    return locked


class Arm_Collision_Model:
    """Collision checking, kinematics and IK for EDAN (8 DOF) / MAYA (7 DOF).

    Public methods accept a full reduced configuration or seven arm joints.
    For EDAN a seven-joint input retains the current additional axis.
    Generic7R and SMMPL use the last seven coordinates exclusively.
    """

    def __init__(self, use_octomap: bool = True, use_frocc: bool = True) -> None:
        self.cfg: RobotCollisionConfig = get_robot_collision_config()
        self.robot_interface = Robot_Interface()
        self._build_robot_model()

        self._num_threads = os.cpu_count() or 1
        self._geometry_pool = None
        self.use_octomap = use_octomap
        self.use_frocc = use_frocc
        self._frocc = None
        self.octree_getter = None
        self.env_geom_id: Optional[int] = None
        self.lock = RLock()
        self.last_update = 0

        if use_octomap:
            self._initialize_octomap()

        self.q = pin.neutral(self.model)
        self.dq_buffer = np.zeros(self.model.nv)
        print(
            f"Init done (REDUCED) – robot={self.cfg.robot_name}, "
            f"urdf={self.cfg.urdf_path}, nv={self.model.nv}, "
            f"geoms={len(self.geom_model.geometryObjects)}, "
            f"pairs={len(self.geom_model.collisionPairs)}, "
            f"parallel collision checking with pinocchio possible? {hasattr(pin, "computeCollisionsInParallel")}"
            f"collision_backend={'frocc (1 thread, full_batch)' if self._frocc is not None else 'pinocchio'}"
        )

    def _build_robot_model(self) -> None:
        """Load the URDF and lock all joints excluded by the velocity mask."""
        urdf = self.cfg.urdf_path
        full_model = pin.buildModelFromUrdf(urdf)
        self.full_data = full_model.createData()

        full_geom_model = pin.GeometryModel()
        pin.buildGeomFromUrdf(
            full_model, urdf, pin.GeometryType.COLLISION, full_geom_model
        )
        locked_joints = _compute_joints_to_lock_from_vmask(
            full_model, self.cfg.active_vmask_full
        )
        self.model, self.geom_model = pin.buildReducedModel(
            full_model, full_geom_model, locked_joints, pin.neutral(full_model)
        )
        # The reduced model must have scalar 1-DOF joints and at least 7 arm axes.
        if self.model.nq != self.model.nv or self.model.nq < 7:
            raise ValueError(
                f"{self.cfg.robot_name}: expected nq=nv>=7 after reduction, "
                f"got nq={self.model.nq}, nv={self.model.nv}"
            )
        if self.model.nv != int(np.count_nonzero(self.cfg.active_vmask_full)):
            raise ValueError("Reduced model does not match active velocity mask")
        for joint_id in range(1, self.model.njoints):
            joint = self.model.joints[joint_id]
            if joint.nv and (joint.nq != 1 or joint.nv != 1):
                raise ValueError(
                    f"{self.cfg.robot_name}: joint {joint_id} has "
                    f"nq={joint.nq}, nv={joint.nv}; fast IK requires "
                    "scalar 1-DOF joints"
                )
        self.data = self.model.createData()
        self.geom_data = self.geom_model.createData()

        self.ee_frame_id = self.model.getFrameId(self.cfg.ee_frame)
        if self.ee_frame_id == len(self.model.frames):
            raise ValueError(f"EE frame '{self.cfg.ee_frame}' not found")

        self.elbow_jid = self.model.getJointId(self.cfg.elbow_joint)
        if self.elbow_jid <= 0:
            raise ValueError(f"Elbow joint '{self.cfg.elbow_joint}' not found")

    def _initialize_octomap(self) -> None:
        """Attach the first received environment octree to the geometry model."""
        from navigation.mapping.coal_octree_getter import CoalOctreeGetter

        self.octree_getter = CoalOctreeGetter(
            use_pointcloud_fast=False,
            prepare_frocc=self.use_frocc,
        )
        first_octree, first_timestamp = self.wait_for_first_octree()
        with self.lock:
            self.env_geom_id = self._add_env_geometry(first_octree)
            self.last_update = first_timestamp
            self._setup_robot_vs_env_pairs()
            self.geom_data = self.geom_model.createData()
            self._setup_security_margins()

        if self.use_frocc:
            try:
                from python_modules.environment_awareness.frocc_collision import (
                    FroccCollisionBackend,
                )
                self._frocc = FroccCollisionBackend(self)
            except Exception as exc:  # defensive: keep the pinocchio backend
                print(f"[Arm_Collision_Model] frocc init failed, using pinocchio: {exc}")
                self._frocc = None

    # Octomap and environment geometry

    def wait_for_first_octree(self, timeout: float = 60, interval: float = 0.2):
        print(f"Wait for first octomap for {timeout} sec...")
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            octree, msg_time = self.octree_getter.get_octree()
            if octree is not None:
                print("Received first octomap...")
                return octree, msg_time
            time.sleep(interval)
        raise TimeoutError(f"Could not get octomap within {timeout} seconds.")

    def _add_env_geometry(self, octree) -> int:
        env_obj = pin.GeometryObject(
            self.cfg.env_geom_name,
            0,  # world
            pin.SE3.Identity(),
            octree,
        )
        return self.geom_model.addGeometryObject(env_obj)

    def _setup_robot_vs_env_pairs(self) -> None:
        """Use geometry names so pair selection survives model reindexing."""
        if self.env_geom_id is None:
            raise RuntimeError("Environment geometry has not been initialized")
        name_to_gid = {go.name: i for i, go in enumerate(self.geom_model.geometryObjects)}

        for name in self.cfg.robot_collision_geom_names:
            geometry_id = name_to_gid.get(name)
            if geometry_id is None or geometry_id == self.env_geom_id:
                continue
            self.geom_model.addCollisionPair(
                pin.CollisionPair(geometry_id, self.env_geom_id)
            )

    def _setup_security_margins(self, value: Optional[float] = None) -> None:
        margin = self.cfg.margin if value is None else float(value)
        margins = np.zeros((self.geom_model.ngeoms, self.geom_model.ngeoms))
        for pair in self.geom_model.collisionPairs:
            margins[pair.first, pair.second] = margin

        self.geom_data.setSecurityMargins(
            self.geom_model, margins, upper=True, sync_distance_upper_bound=False
        )
        self._sync_geometry_pool()

    def _sync_geometry_pool(self) -> None:
        """Synchronize geometry, placements, and margins for parallel checks."""
        if not hasattr(pin, "GeometryPool"):
            return

        self._geometry_pool = pin.GeometryPool(self.model, self.geom_model, self._num_threads)

        # Includes placements, active pairs, and collision requests/margins.
        self._geometry_pool.update(self.geom_data)

    def update_env_octree(self, obj: Optional[str] = None) -> bool:
        """Refresh environment geometry and placement; return whether octree changed."""
        if not self.use_octomap:
            return False

        new_octree, timestamp = self.octree_getter.get_octree(obj=obj)
        if new_octree is None:
            return False

        with self.lock:
            changed = timestamp != self.last_update
            if changed:
                self.last_update = timestamp

            Wt_T_Ws = invert_transformation(self.robot_interface.get_Ws_T_Wt())
            if self._frocc is not None:
                # Same update as the Pinocchio branch: install the bridge's
                # prepared Ws map and place it with Wt_T_Ws.
                self._frocc.update_from_bridge(obj=obj, octomap_pose=Wt_T_Ws)
            else:
                assert self.env_geom_id is not None
                env_geometry = self.geom_model.geometryObjects[self.env_geom_id]
                # Swap on a map update or a different object filter (filtered
                # octrees are rebuilt per call, so identity changes).
                if changed or env_geometry.geometry is not new_octree:
                    env_geometry.geometry = new_octree
                env_geometry.placement = pin.SE3(Wt_T_Ws)
                pin.updateGeometryPlacements(
                    self.model, self.data, self.geom_model, self.geom_data
                )
                self._sync_geometry_pool()
        return changed

    def normalize_q(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """Return nq reduced joint angles; 7-arm input keeps EDAN's extra axis."""
        if q is None:
            return self.get_current_robot_q()

        nq = self.model.nq
        n_extra = nq - 7
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size == 7 and n_extra:
            extra = self.get_current_robot_q()[:n_extra]
            result = np.concatenate((extra, q))
        elif q.size >= nq:
            result = q[-nq:].copy()
        else:
            raise ValueError(f"Expected 7 arm joints or >= {nq} values, got {q.size}")

        if not np.isfinite(result).all():
            raise ValueError("Joint configuration contains non-finite values")
        return result

    def normalize_q_batch(self, qs: np.ndarray) -> np.ndarray:
        """Normalize an (N, joints) batch with one extra-axis snapshot."""
        qs = np.asarray(qs, dtype=float)
        if qs.ndim != 2:
            raise ValueError(f"Expected a (N, joints) array, got {qs.shape}")

        nq = self.model.nq
        n_extra = nq - 7
        if qs.shape[1] == 7 and n_extra:
            extra = self.get_current_robot_q()[:n_extra]
            result = np.hstack((np.tile(extra, (len(qs), 1)), qs))
        elif qs.shape[1] >= nq:
            result = qs[:, -nq:].copy()
        else:
            raise ValueError(f"Expected (N, 7) or (N, >= {nq}), got {qs.shape}")

        if not np.isfinite(result).all():
            raise ValueError("Joint configuration batch contains non-finite values")
        return result

    def get_current_robot_q(self) -> np.ndarray:
        nq = self.model.nq
        q = np.asarray(self.robot_interface.get_q(), dtype=float).reshape(-1)
        if q.size < nq:
            raise ValueError(f"Robot interface returned {q.size} joints; need {nq}")
        result = q[-nq:].copy()
        if not np.isfinite(result).all():
            raise ValueError("Current joint configuration contains non-finite values")
        return result

    # Collision checking

    def is_in_collision(self, q: np.ndarray) -> bool:
        reduced_q = self.normalize_q(q)
        with self.lock:
            if self._frocc is not None:
                return self._frocc.is_in_collision(reduced_q)
            return bool(pin.computeCollisions(
                self.model, self.data, self.geom_model, self.geom_data,
                reduced_q, True,  # stopAtFirstCollision
            ))

    def is_in_collision_batch(self, qs: np.ndarray) -> np.ndarray:
        """Check N configurations; True means collision.

        Inputs may have shape (N, 7), (N, model.nq), or longer interface rows.
        All seven-joint EDAN rows share one snapshot of the extra axis.
        Frocc (1 worker thread) runs a batch full check when active.
        """
        qs = self.normalize_q_batch(qs)
        if not len(qs):
            return np.empty(0, dtype=bool)

        with self.lock:
            if self._frocc is not None:
                return self._frocc.is_in_collision_batch(qs)

        # The pool holds its own geometry data; avoid updating it mid-query.
        with self.lock:
            if self._geometry_pool is None or not hasattr(pin, "computeCollisionsInParallel"):
                return np.array(
                    [bool(pin.computeCollisions(
                        self.model, self.data, self.geom_model, self.geom_data, q, True
                    )) for q in qs],
                    dtype=bool,
                )

            columns = np.asfortranarray(qs.T)  # Pinocchio: (nq, N), F-order
            result = pin.computeCollisionsInParallel(
                self._num_threads,
                self._geometry_pool,
                columns,
                stop_at_first_collision_in_configuration=True,
                stop_at_first_collision_in_batch=False,
            )
        return np.asarray(result, dtype=bool).reshape(-1)

    # Robot state

    def update_robot_config(self, q=None):
        """
        Reads q from Robot_Interface and writes reduced q.
        Robot_Interface may expose leading joint axes; only trailing model.nq values are used.
        """
        q = self.normalize_q(q)
        self.q = q.copy()
        pin.framesForwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    # Jacobian and inverse kinematics

    def get_J(self, q: np.ndarray, reference="ee") -> np.ndarray:
        if reference not in {"ee", "wt"}:
            raise ValueError("reference must be 'ee' or 'wt'")
        reference_frame = (
            pin.ReferenceFrame.LOCAL
            if reference == "ee"
            else pin.LOCAL_WORLD_ALIGNED
        )
        return pin.computeFrameJacobian(
            self.model,
            self.data,
            self.normalize_q(q),
            self.ee_frame_id,
            reference_frame,
        )  # Shape: (6, nv)

    def get_v_iso(self, q: Optional[np.ndarray] = None, polytope: str = "strong") -> float:
        """Isotropic EE velocity of the seven arm joints at q (m/s).

        ``polytope="strong"`` holds the orientation fixed; ``"weak"`` frees it.
        """
        from python_modules.environment_awareness.manipulability.manipulability import (
            strong_velocity_polytope,
            weak_velocity_polytope,
        )

        q = self.normalize_q(q)
        J6 = self.get_J(q)[:, -7:]
        dq_max = np.asarray(self.model.velocityLimit[-7:], dtype=float)
        if polytope == "weak":
            poly = weak_velocity_polytope(J6, -dq_max, dq_max)
        elif polytope == "strong":
            poly = strong_velocity_polytope(J6, -dq_max, dq_max)
        else:
            raise ValueError(f"Unknown polytope {polytope!r}; use 'weak' or 'strong'")
        if poly is None or poly.is_flat:
            return 0.0
        return float(poly.metrics().v_iso)


    def local_ik_newton(
        self,
        target: pin.SE3,
        q0: Optional[np.ndarray] = None,
        max_iters: int = 50,
        pos_tol: float = 0.001,
        rot_tol: float = np.deg2rad(0.1),
        damp: float = 1e-8,
        alpha: float = 1.0,
        fix_extra_axes: bool = False,
        return_none: bool = True,
    ) -> Optional[np.ndarray]:

        if isinstance(target, np.ndarray):
            target = pin.SE3(target)

        q = self.normalize_q(q0)
        self.q = q.copy()

        model, data, fid = self.model, self.data, self.ee_frame_id
        offset = self.model.nv - 7 if fix_extra_axes else 0
        dq = self.dq_buffer
        dq[:offset] = 0.0

        def converged(error):
            return (
                np.linalg.norm(error[:3]) < pos_tol
                and np.linalg.norm(error[3:]) < rot_tol
            )

        success = False

        for _ in range(max_iters):
            pin.computeJointJacobians(model, data, q)
            pin.updateFramePlacement(model, data, fid)

            error = pin.log6(data.oMf[fid].actInv(target)).vector
            if converged(error):
                success = True
                break

            J = pin.getFrameJacobian(
                model, data, fid, pin.ReferenceFrame.LOCAL
            )[:, offset:]

            JJ = J @ J.T
            JJ.flat[::7] += damp

            dq[offset:] = alpha * (J.T @ np.linalg.solve(JJ, error))
            q += dq

            if np.linalg.norm(dq[offset:]) < 1e-8:
                break

        if not success:
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacement(model, data, fid)
            error = pin.log6(data.oMf[fid].actInv(target)).vector
            success = converged(error)

        if not success:
            if return_none:
                return None
            print("[Warning] IK not properly solved")

        # Wrap all joint angles to [-pi, pi)
        q[:] = (q + np.pi) % (2 * np.pi) - np.pi
        return q


    def check_ee_pose(self, target_pose: pin.SE3) -> Tuple[bool, Optional[np.ndarray]]:
        # An unconverged IK candidate must not be reported as a valid pose.
        q_candidate = self.local_ik_newton(target_pose, return_none=True)
        if q_candidate is None:
            return False, None
        in_collision = self.is_in_collision(q_candidate)
        return not in_collision, (q_candidate if not in_collision else None)

    # Public collision checks

    def check_collisionfree_q(self, q, update_octomap=True) -> bool:
        q = self.normalize_q(q)
        if update_octomap:
            self.update_env_octree()
        return not self.is_in_collision(q)

    def check_collisionfree_target(self, target: np.ndarray, update_octomap=True):
        """Check if an end-effector target pose in Wt is collision-free."""
        if np.asarray(target).shape != (4, 4):
            raise ValueError("target must be a 4x4 homogeneous transformation")
        self.update_robot_config()
        if update_octomap:
            self.update_env_octree()

        target_pin = pin.SE3(target)
        valid, q_new = self.check_ee_pose(target_pin)
        return valid, q_new

    # Geometry utilities

    def get_elbow_position_for_q(self, q=None):
        q = self.normalize_q(q)
        pin.forwardKinematics(self.model, self.data, q)
        T = self.data.oMi[self.elbow_jid]
        return T.translation.copy()

    def get_ee_pose_for_q(self, q=None):
        q = self.normalize_q(q)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        ee = self.data.oMf[self.ee_frame_id]
        return ee.homogeneous.copy()

    def print_ee_pose(self):
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        ee = self.data.oMf[self.ee_frame_id]
        print(ee.translation)
        return ee.translation.copy()

    def get_current_ee_pos(self):
        """Return the live measured joint state's FK EE position in Wt."""
        return self.get_ee_pose_for_q(self.get_current_robot_q())[:3, 3]

    def get_J2_T_W(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """Return W->J2 for EDAN, identity for MAYA.

        With ``q`` supplied, the EDAN extra axis comes from the planning state
        rather than the live robot. This avoids a planning-frame mismatch.
        """
        if self.cfg.robot_name == "maya":
            return np.eye(4)

        q_at_joint2 = pin.neutral(self.model)
        n_extra = self.model.nq - 7
        q_source = self.normalize_q(q) if q is not None else self.get_current_robot_q()
        q_at_joint2[:n_extra] = q_source[:n_extra]
        if self.model.njoints <= 2:
            raise ValueError("EDAN reduced model does not contain joint ID 2")
        previous_q = self.q.copy()
        try:
            self.update_robot_config(q_at_joint2)
            W_T_J2 = self.data.oMi[2].homogeneous.copy()
        finally:
            self.update_robot_config(previous_q)
        return invert_transformation(W_T_J2)

    def get_collision_aabbs_for_q(
        self,
        q: Optional[np.ndarray] = None,
        arm_scale: float = 2.0,
        hand_scale: float = 7.0,
    ):
        if q is None:
            q = self.robot_interface.get_q()
        q = self.normalize_q(q)

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateGeometryPlacements(self.model, self.data, self.geom_model, self.geom_data)

        out = {}
        for gid, go in enumerate(self.geom_model.geometryObjects):
            if go.name == self.cfg.env_geom_name:
                continue

            go.geometry.computeLocalAABB()
            aabb = go.geometry.aabb_local

            # Smaller geometries (usually hands) receive a larger padding.
            scale = hand_scale if aabb.volume() < 1e-4 else arm_scale
            half_extent = 0.5 * (aabb.max_ - aabb.min_) * scale
            center = aabb.center()
            offsets = np.array(
                [
                    [-1, -1, -1], [1, -1, -1],
                    [-1, 1, -1], [1, 1, -1],
                    [-1, -1, 1], [1, -1, 1],
                    [-1, 1, 1], [1, 1, 1],
                ]
            )
            local_corners = center + offsets * half_extent
            placement = self.geom_data.oMg[gid]
            world_corners = np.array(
                [placement.act(corner) for corner in local_corners]
            )

            out[go.name] = world_corners

        return out
