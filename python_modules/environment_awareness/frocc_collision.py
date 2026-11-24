"""Frocc 0.16 collision backend for :class:`Arm_Collision_Model`.

The octree bridge prepares the ``frocc.PreparedOctomap`` in ``Ws`` when
``prepare_frocc`` is enabled. This backend mirrors the Pinocchio update path:
the prepared map is installed with ``set_octomap`` and the map is placed with
``set_octomap_pose(Robot_T_Octomap = Wt_T_Ws)``. There is intentionally no
FROCC 0.15 box-in-Wt fallback.

Robot-specific settings (locked joints, collision geometry names, margin) come
from :mod:`python_modules.environment_awareness.robot_collision_cfg`.
"""
from __future__ import annotations

import numpy as np

from robotutils.geom.utils import invert_transformation


class FroccCollisionBackend:
    """Thin wrapper around ``frocc.CollisionChecker`` bound to a collision model."""

    def __init__(self, model, threads=1, slots=1):
        from frocc import CollisionChecker

        if not hasattr(CollisionChecker, "set_octomap_pose"):
            raise RuntimeError(
                "FROCC >= 0.16 is required (set_octomap_pose is missing). "
                "Deploy frocc/main@spie-si/snapshot with FROCC 0.16 or newer."
            )
        if not getattr(model.octree_getter, "prepare_frocc", False):
            raise RuntimeError(
                "FroccCollisionBackend requires CoalOctreeGetter(prepare_frocc=True)."
            )

        cfg = model.cfg
        self._model = model

        prepared, _ = model.octree_getter.get_prepared_octomap()
        pose = self._octomap_pose()
        self.checker = CollisionChecker.from_urdf(
            cfg.urdf_path,
            prepared,
            threads=threads,
            slots=slots,
            collision_pairs=[(g, "octomap") for g in cfg.robot_collision_geom_names],
            locked_joints=dict(cfg.frocc_locked_joints),
            octomap_pose=pose,
            margin=float(cfg.margin),
        )
        self._validate()
        print(
            f"[FroccCollisionBackend] ready: nq={self.checker.nq}, "
            f"joints={self.checker.joint_names}, threads={self.checker.threads}, "
            f"voxels={int(self.checker.statistics.get('source_voxels', 0))}, "
            f"map_frame=Ws (octomap_pose), backend={self.checker.backend}"
        )

    def _validate(self):
        """Fail loudly unless the frocc reduced model is q-compatible with pinocchio."""
        model = self._model.model
        free = tuple(model.names[j] for j in range(1, model.njoints))
        if self.checker.nq != model.nq or tuple(self.checker.joint_names) != free:
            self.checker.close()
            raise ValueError(f"frocc model mismatch: {self.checker.joint_names} vs {free}")
        missing = self._model.cfg.robot_collision_geom_names - set(self.checker.geometry_names)
        if missing:
            self.checker.close()
            raise ValueError(f"frocc missing collision geometries: {sorted(missing)}")

    def _octomap_pose(self):
        """Return Robot_T_Octomap (Wt_T_Ws) for the current robot base."""
        return invert_transformation(self._model.robot_interface.get_Ws_T_Wt())

    def update_from_bridge(self, obj=None, octomap_pose=None) -> None:
        """Install the prepared map and its Wt placement, like the Pinocchio branch."""
        prepared, _ = self._model.octree_getter.get_prepared_octomap(obj=obj)
        self.checker.set_octomap(prepared)
        if octomap_pose is None:
            octomap_pose = self._octomap_pose()
        self.checker.set_octomap_pose(octomap_pose)

    def is_in_collision(self, q) -> bool:
        return not bool(self.checker.full(q))

    def is_in_collision_batch(self, qs) -> np.ndarray:
        free = self.checker.full_batch(np.ascontiguousarray(qs, dtype=np.float64))
        return ~np.asarray(free, dtype=bool).reshape(-1)
