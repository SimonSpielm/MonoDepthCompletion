#!/usr/bin/env python3
"""Velocity capability and pose projection for serial arms (numpy / scipy).

Capability (used by the governor to compute d_max):
- ``JointLimits``: joint position/velocity limits -> configuration-dependent
  joint velocity box (braking distance near position limits).
- ``strong_velocity_polytope`` (orientation fixed) and
  ``weak_velocity_polytope`` (orientation free): ``VelocityPolytope`` with
  ``max_velocity_along`` and ``metrics`` (``PolytopeMetrics``: kind
  "strong"/"weak", v_iso, v_max, volume, isotropy).
- ``damped_pinv``.

Projection (import wherever a target pose is limited by d_max / r_max):
- ``cap_translation_to_sphere``: position onto a sphere of radius d_max.
- ``cap_rotation_slerp``: orientation onto a cone of angle r_max (SLERP).
- ``project_pose``: both at once for 4x4 homogeneous poses.
- ``TargetDeviationLimits``: validated (d_max, r_max) pair with ``apply``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp


# ---------------------------------------------------------------------------
# Joint limits -> joint velocity box
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointLimits:
    """Position, velocity and deceleration limits of a serial arm (nq == nv).

    ``velocity_bounds(q)`` gives the joint velocity box: the nominal range
    ``[-dq_max, dq_max]``, tightened near a position limit so the joint can
    still brake to zero before reaching it (constant deceleration
    ``qddot_max``, optional safety ``margin``)::

        dq_hi = min( dq_max,  sqrt(2 * qddot_max * max(q_max - margin - q, 0)))
        dq_lo = max(-dq_max, -sqrt(2 * qddot_max * max(q - q_min - margin, 0)))
    """

    q_min: np.ndarray
    q_max: np.ndarray
    dq_max: np.ndarray
    qddot_max: np.ndarray
    margin: float = 0.0

    @classmethod
    def from_model(cls, model, qddot_max, margin: float = 0.0) -> "JointLimits":
        if model.nq != model.nv:
            raise RuntimeError(
                f"JointLimits erwartet nq == nv (nq={model.nq}, nv={model.nv}); "
                "Modelle mit Kontinuierlich-/Free-Flyer-Gelenken werden nicht unterstützt."
            )
        n = model.nv

        def vec(name: str) -> np.ndarray:
            value = getattr(model, name, None)
            if value is None:
                raise RuntimeError(f"arm.model.{name} fehlt. Bitte manuell setzen.")
            return np.asarray(value, dtype=float).reshape(n)

        dq_max = vec("velocityLimit")
        if np.any(dq_max <= 0.0):
            raise ValueError(f"velocityLimit muss > 0 sein (URDF prüfen): {dq_max}")
        qddot = np.broadcast_to(np.asarray(qddot_max, dtype=float), (n,)).copy()
        if np.any(qddot <= 0.0):
            raise ValueError("qddot_max muss > 0 sein")
        return cls(
            q_min=vec("lowerPositionLimit"),
            q_max=vec("upperPositionLimit"),
            dq_max=dq_max,
            qddot_max=qddot,
            margin=float(margin),
        )

    def _braking_speed(self, room: np.ndarray) -> np.ndarray:
        """Highest speed that can still stop within ``room`` of the limit."""
        return np.sqrt(2.0 * self.qddot_max * np.maximum(room, 0.0))

    def velocity_bounds(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Configuration-dependent joint velocity box ``(dq_lo, dq_hi)``."""
        q = np.asarray(q, dtype=float).reshape(self.dq_max.shape)
        v_up = self._braking_speed(self.q_max - self.margin - q)
        v_down = self._braking_speed(q - self.q_min - self.margin)
        return np.maximum(-self.dq_max, -v_down), np.minimum(self.dq_max, v_up)

    def clip(self, q: np.ndarray) -> np.ndarray:
        """Clip q to the position limits (only joints with BOTH limits finite)."""
        q = np.asarray(q, dtype=float).copy()
        mask = np.isfinite(self.q_min) & np.isfinite(self.q_max)
        q[mask] = np.clip(q[mask], self.q_min[mask], self.q_max[mask])
        return q


# ---------------------------------------------------------------------------
# Velocity capability
# ---------------------------------------------------------------------------


def damped_pinv(J: np.ndarray, damping: float = 1e-4) -> np.ndarray:
    """Damped pseudoinverse J.T @ inv(J @ J.T + damping * I)."""
    J = np.asarray(J, dtype=float)
    return J.T @ np.linalg.inv(J @ J.T + float(damping) * np.eye(J.shape[0]))


PolytopeKind = Literal["strong", "weak"]


@dataclass(frozen=True)
class PolytopeMetrics:
    """Direction-independent capability of a translational velocity polytope.

    kind    "strong" (orientation held fixed) or "weak" (orientation free)
    v_iso   (m/s)     speed reachable in EVERY direction (largest origin-centred
                      ball inside the polytope) -> worst case, best classifier
    v_max   (m/s)     speed in the best direction (farthest vertex)
    volume  ((m/s)^3) polytope volume -> "overall" capability, hides thin directions
    """

    kind: PolytopeKind
    v_iso: float
    v_max: float
    volume: float

    @property
    def v_vol(self) -> float:
        """Cube root of the volume, in m/s (comparable to v_iso / v_max)."""
        return float(np.cbrt(self.volume))

    @property
    def isotropy(self) -> float:
        """v_iso / v_max in [0, 1]; 0 = flat in some direction, 1 = sphere."""
        return self.v_iso / self.v_max if self.v_max > 0.0 else 0.0


@dataclass(frozen=True)
class VelocityPolytope:
    """Translational velocity polytope ``{v : H v <= d}`` (unit normals).

    ``points`` (3 x N) span the polytope as their convex hull (vertices, for the
    weak polytope all box corners). If the polytope is flat (singular),
    ``H``/``d`` are None and ray queries, v_iso and volume are 0.
    """

    kind: PolytopeKind
    H: Optional[np.ndarray]
    d: Optional[np.ndarray]
    volume: float
    points: np.ndarray

    @property
    def is_flat(self) -> bool:
        return self.H is None

    def max_velocity_along(self, direction: np.ndarray, eps: float = 1e-12) -> float:
        """Max EE speed along ``direction`` (m/s); strong: with zero angular velocity."""
        if self.is_flat:
            return 0.0
        u = np.asarray(direction, dtype=float).reshape(3)
        norm = float(np.linalg.norm(u))
        if norm < eps:
            return 0.0
        proj = self.H @ (u / norm)
        facing = proj > eps
        if not np.any(facing):
            return 0.0
        return max(0.0, float(np.min(self.d[facing] / proj[facing])))

    def metrics(self) -> PolytopeMetrics:
        v_max = float(np.max(np.linalg.norm(self.points, axis=0))) if self.points.size else 0.0
        v_iso = 0.0 if self.is_flat else max(0.0, float(np.min(self.d)))
        return PolytopeMetrics(kind=self.kind, v_iso=v_iso, v_max=v_max, volume=self.volume)


@lru_cache(maxsize=None)
def _index_combinations(n: int, k: int) -> Tuple[np.ndarray, ...]:
    """Column index arrays of all k-subsets of range(n)."""
    combos = np.array(list(combinations(range(n), k)), dtype=int).reshape(-1, k)
    return tuple(combos[:, i] for i in range(k))


@lru_cache(maxsize=None)
def _box_corner_signs(n: int) -> np.ndarray:
    """All 2^n sign vectors of an n-box, shape (2^n, n)."""
    return np.array(list(product((-1.0, 1.0), repeat=n)), dtype=float)


@lru_cache(maxsize=None)
def _slice_vertex_indices(n: int, m: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index tables for vertices of {lo <= dq <= hi, Jw dq = 0} (m constraints).

    Every vertex has n - m joints on a bound (``fixed``, pattern ``signs``)
    and m joints solved from the m x m system (``free``).
    """
    free = np.array(list(combinations(range(n), m)), dtype=int)
    fixed = np.array([[k for k in range(n) if k not in f] for f in free], dtype=int)
    signs = np.array(list(product((False, True), repeat=n - m)), dtype=bool)
    return free, fixed, signs


def weak_velocity_polytope(
    J: np.ndarray,
    dq_lo: np.ndarray,
    dq_hi: np.ndarray,
    eps: float = 1e-12,
) -> Optional[VelocityPolytope]:
    """Weak translational polytope ``{Jv dq : dq_lo <= dq <= dq_hi}`` (orientation free).

    Pure numpy, no hull: the image of a box is a zonotope ``c + G t, |t| <= 1``;
    in 3D every facet normal is the cross product of two generators
    (hyper-plane shifting), the volume is ``8 * sum |det(g_a, g_b, g_c)|``.
    ``J`` is the 6 x n (only the linear rows are used) or 3 x n Jacobian.
    Returns None if n < 2.
    """
    J = np.asarray(J, dtype=float)
    Jv = J.reshape(-1, J.shape[-1])[:3]
    lo = np.asarray(dq_lo, dtype=float).reshape(-1)
    hi = np.asarray(dq_hi, dtype=float).reshape(-1)
    n = Jv.shape[1]
    if n < 2:
        return None
    center = Jv @ (0.5 * (hi + lo))
    G = Jv * (0.5 * (hi - lo))  # 3 x n half-length generators
    points = (center + _box_corner_signs(n) @ G.T).T  # all corners, superset of vertices

    volume = 0.0
    if n >= 3:
        a, b, c = _index_combinations(n, 3)
        dets = np.einsum("ij,ij->i", G[:, a].T, np.cross(G[:, b].T, G[:, c].T))
        volume = 8.0 * float(np.abs(dets).sum())

    i, j = _index_combinations(n, 2)
    normals = np.cross(G[:, i].T, G[:, j].T)
    norms = np.linalg.norm(normals, axis=1)
    keep = norms > eps * max(1.0, float(norms.max()))
    if volume <= eps or not np.any(keep):
        return VelocityPolytope(kind="weak", H=None, d=None, volume=0.0, points=points)
    normals = normals[keep] / norms[keep, None]
    reach = np.abs(normals @ G).sum(axis=1)
    offset = normals @ center
    return VelocityPolytope(
        kind="weak",
        H=np.vstack((normals, -normals)),
        d=np.concatenate((offset + reach, reach - offset)),
        volume=volume,
        points=points,
    )


def strong_velocity_polytope(
    J6: np.ndarray,
    dq_lo: np.ndarray,
    dq_hi: np.ndarray,
    eps: float = 1e-10,
) -> Optional[VelocityPolytope]:
    """Strong translational polytope ``{Jv dq : Jw dq = 0, dq_lo <= dq <= dq_hi}``.

    Orientation held fixed: unlike the weak polytope, this sees that a
    stretched arm cannot move outward by tilting the wrist.
    Vertices of the joint-space slice: for every choice of 3 free joints with
    invertible Jw[:, free], set the other n-3 joints to all bound
    combinations, solve for the free joints and keep feasible solutions; map
    them with Jv and take one convex hull. For n = 7: 35 x 16 candidates.
    ``J6`` rows are [linear; angular]. Returns None if n < 4 or Jw has
    rank < 3 (angular singularity).
    """
    J6 = np.asarray(J6, dtype=float).reshape(6, -1)
    lo = np.asarray(dq_lo, dtype=float).reshape(-1)
    hi = np.asarray(dq_hi, dtype=float).reshape(-1)
    Jv, Jw = J6[:3], J6[3:]
    n = J6.shape[1]
    if n < 4:
        return None

    free, fixed, signs = _slice_vertex_indices(n)
    A = Jw[:, free].transpose(1, 0, 2)  # (C, 3, 3)
    scale = max(1.0, float(np.abs(Jw).max())) ** 3
    good = np.abs(np.linalg.det(A)) > eps * scale
    if not np.any(good):
        return None
    free, fixed, A = free[good], fixed[good], A[good]

    vals = np.where(signs[None], hi[fixed][:, None, :], lo[fixed][:, None, :])  # (C, S, n-3)
    rhs = -np.einsum("crk,csk->csr", Jw[:, fixed].transpose(1, 0, 2), vals)    # (C, S, 3)
    x = np.linalg.solve(A[:, None], rhs[..., None])[..., 0]                     # (C, S, 3)
    tol = 1e-9 * max(1.0, float(np.abs(hi - lo).max()))
    ok = np.all((x >= lo[free][:, None] - tol) & (x <= hi[free][:, None] + tol), axis=2)
    v = (np.einsum("crk,csk->csr", Jv[:, free].transpose(1, 0, 2), x)
         + np.einsum("crk,csk->csr", Jv[:, fixed].transpose(1, 0, 2), vals))
    V = v[ok]                                                                   # (N, 3)
    if V.shape[0] == 0:
        V = np.zeros((1, 3))  # only dq = 0 feasible

    try:
        hull = ConvexHull(V)
    except (RuntimeError, ValueError):  # QhullError: flat / degenerate polytope
        return VelocityPolytope(kind="strong", H=None, d=None, volume=0.0, points=V.T)
    return VelocityPolytope(
        kind="strong",
        H=hull.equations[:, :3],
        d=-hull.equations[:, 3],
        volume=float(hull.volume),
        points=V[hull.vertices].T,
    )


# ---------------------------------------------------------------------------
# Pose projection (d_max / r_max)
# ---------------------------------------------------------------------------


def rotation_angle_between(R0: np.ndarray, R1: np.ndarray) -> float:
    """Shortest angular distance between two rotations, in radians."""
    R0 = np.asarray(R0, dtype=float).reshape(3, 3)
    R1 = np.asarray(R1, dtype=float).reshape(3, 3)
    return float(Rotation.from_matrix(R0.T @ R1).magnitude())


def cap_translation_to_sphere(
    p_cur: np.ndarray,
    p_des: np.ndarray,
    radius: float,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float, float]:
    """Project the desired position onto a sphere of ``radius`` around p_cur.

    Returns (p_cmd, distance p_cur->p_des, ratio in [0, 1]).
    """
    p_cur = np.asarray(p_cur, dtype=float).reshape(3)
    p_des = np.asarray(p_des, dtype=float).reshape(3)
    radius = max(0.0, float(radius))

    delta = p_des - p_cur
    distance = float(np.linalg.norm(delta))
    if distance < eps or distance <= radius + eps:
        return p_des.copy(), distance, 1.0

    ratio = radius / distance
    return p_cur + ratio * delta, distance, float(ratio)


def cap_rotation_slerp(
    R_cur: np.ndarray,
    R_des: np.ndarray,
    max_angle: float,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float, float]:
    """Limit the orientation deviation to ``max_angle`` along the shortest SLERP path.

    Returns (R_cmd, angle R_cur->R_des, SLERP ratio in [0, 1]).
    """
    R_cur = np.asarray(R_cur, dtype=float).reshape(3, 3)
    R_des = np.asarray(R_des, dtype=float).reshape(3, 3)
    angle = rotation_angle_between(R_cur, R_des)
    max_angle = max(0.0, float(max_angle))
    if angle < eps or angle <= max_angle + eps:
        return R_des.copy(), angle, 1.0

    ratio = max_angle / angle
    rotations = Rotation.from_matrix(np.stack((R_cur, R_des), axis=0))
    result = Slerp([0.0, 1.0], rotations)([ratio]).as_matrix()[0]
    return result, angle, float(ratio)


def project_pose(
    T_cur: np.ndarray,
    T_des: np.ndarray,
    d_max: float,
    r_max: Optional[float] = None,
) -> np.ndarray:
    """Limit a desired 4x4 pose to d_max (m) and r_max (rad) around the current pose.

    Translation and rotation are limited independently; r_max=None leaves the
    orientation unchanged.
    """
    T_cur = np.asarray(T_cur, dtype=float).reshape(4, 4)
    T_cmd = np.asarray(T_des, dtype=float).reshape(4, 4).copy()
    T_cmd[:3, 3] = cap_translation_to_sphere(T_cur[:3, 3], T_cmd[:3, 3], d_max)[0]
    if r_max is not None:
        T_cmd[:3, :3] = cap_rotation_slerp(T_cur[:3, :3], T_cmd[:3, :3], r_max)[0]
    return T_cmd


@dataclass(frozen=True)
class TargetDeviationLimits:
    """Validated target-deviation limits ``(d_max, r_max)``.

    ``r_max=None`` leaves the orientation unchanged. ``apply`` projects a
    desired pose onto the limits around a current pose (see ``project_pose``).
    """

    d_max: float
    r_max: Optional[float] = None

    def __post_init__(self) -> None:
        d_max = float(self.d_max)
        if not np.isfinite(d_max) or d_max < 0.0:
            raise ValueError(f"d_max must be finite and >= 0, got {self.d_max}")
        if self.r_max is not None:
            r_max = float(self.r_max)
            if not np.isfinite(r_max) or r_max < 0.0:
                raise ValueError(f"r_max must be finite and >= 0, got {self.r_max}")

    def apply(self, T_cur: np.ndarray, T_des: np.ndarray) -> np.ndarray:
        """Limit ``T_des`` to the configured deviation from ``T_cur``."""
        return project_pose(T_cur, T_des, self.d_max, self.r_max)