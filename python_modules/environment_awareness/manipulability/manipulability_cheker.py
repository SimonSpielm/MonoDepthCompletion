#!/usr/bin/env python3
"""Adaptive target-deviation limits d_max / r_max.

Per control step (original edan governor, with two changes marked *):
  joint limits -> * STRONG translational velocity polytope (zero angular
  velocity, i.e. what the arm can do while holding the orientation)
  -> ray capacity v_now toward the target
  -> 3-point one-step look-ahead along the target direction
  -> * escape bonus: if the capability ahead is higher than now, allow it:
       v_ref = v_now + trend_bonus * max(0, mean(v_probes) - v_now)
     (additive, so a bad configuration can be left even at v_now ~ 0;
      moving into a worse configuration is not braked)
  -> d_max = alpha_poly * v_ref * control_dt -> clip [d_min, d_max]
  -> EMA (faster when d shrinks)
r_max is currently fixed (cfg.r_max).

The limits are *target deviation limits*, NOT certified bounds on joint or
Cartesian velocities of the downstream controller. To apply them to a target
pose, use ``manipulability.project_pose(T_cur, T_des, d_max, r_max)``.

Runs as an LN node: desired EE pose from ``shared.constraints``
(``W_T_EE_des``, W0 frame); d_max and r_max published continuously on
``manipulability.dmax``.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Tuple, Union

import links_and_nodes as ln
import numpy as np

from python_modules.environment_awareness.arm_collision_model import Arm_Collision_Model
from python_modules.environment_awareness.robot_interface import Robot_Interface
from python_modules.environment_awareness.manipulability.manipulability import (
    JointLimits,
    damped_pinv,
    strong_velocity_polytope,
)
from robotutils.geom.se3 import simulink2tf
from robotutils.ros.markers import RvizMarkers


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdaptiveMaxDistConfig:
    # Per-robot deviations from the defaults below. Only list what differs.
    ROBOT_OVERRIDES: ClassVar[Dict[str, Dict[str, Any]]] = {
        # Original edan tuning: no extra caution, symmetric smoothing.
        # trend_bonus 2.0 = small-signal equivalent of the old b+ = 2.0
        # (old: 1 + 2*tanh(rel), new: 1 + 2*rel).
        "edan": {"alpha_poly": 1.0, "d_smooth_down": 0.1, "trend_bonus": 2.0},
        "maya": {},
    }

    # Translation deviation (m).
    d_min: float = 0.02
    d_max: float = 0.12
    control_dt: float = 1.0 / 7.0   # period d refers to; deliberately larger than the 20 Hz node period (bigger sphere)
    alpha_poly: float = 0.8         # < 1: more cautious than the raw capability
    d_smooth: float = 0.1           # EMA factor per control_dt when d grows
    d_smooth_down: float = 0.3      # EMA factor per control_dt when d shrinks (follow faster)

    # Rotation deviation (rad), currently fixed.
    r_max: float = float(np.deg2rad(20.0))

    # Joint-limit braking (rad/s^2, scalar or per joint) and margin (rad).
    qddot_max: Union[float, Tuple[float, ...]] = 4.0
    q_margin: float = 0.0

    # Look-ahead probes along the target direction (m) and escape bonus.
    probe_dist_1: float = 0.05
    probe_dist_2: float = 0.10
    probe_dist_3: float = 0.15
    pinv_damping: float = 1e-4
    trend_bonus: float = 3.5        # 1 = allow the full capability ahead, > 1 more generous

    eps: float = 1e-9

    def __post_init__(self) -> None:
        if not 0.0 <= self.d_min <= self.d_max:
            raise ValueError(f"need 0 <= d_min <= d_max, got {self.d_min}, {self.d_max}")
        if self.control_dt <= 0.0 or self.alpha_poly < 0.0:
            raise ValueError("need control_dt > 0 and alpha_poly >= 0")
        if not (0.0 <= self.d_smooth <= 1.0 and 0.0 <= self.d_smooth_down <= 1.0):
            raise ValueError("d_smooth and d_smooth_down must be in [0, 1]")
        if self.r_max < 0.0:
            raise ValueError("r_max must be >= 0")
        if np.any(np.asarray(self.qddot_max) <= 0.0) or self.q_margin < 0.0:
            raise ValueError("need qddot_max > 0 and q_margin >= 0")
        if not 0.0 < self.probe_dist_1 < self.probe_dist_2 < self.probe_dist_3:
            raise ValueError(f"probe distances must be increasing, got {self.probe_distances}")
        if self.trend_bonus < 0.0:
            raise ValueError("trend_bonus must be >= 0")

    @property
    def probe_distances(self) -> Tuple[float, float, float]:
        return self.probe_dist_1, self.probe_dist_2, self.probe_dist_3

    def smoothing_alpha(self, shrinking: bool, dt: Optional[float] = None) -> float:
        """EMA factor for one step of length dt (default control_dt)."""
        base = self.d_smooth_down if shrinking else self.d_smooth
        if dt is None or abs(dt - self.control_dt) < 1e-12:
            return float(base)
        return float(1.0 - (1.0 - base) ** (float(dt) / self.control_dt))

    @classmethod
    def from_robot(cls, robot_name: Optional[str] = None, **overrides: Any) -> "AdaptiveMaxDistConfig":
        """Config for ROBOT_NAME (edan|maya); keyword overrides take precedence."""
        robot = (robot_name or os.environ.get("ROBOT_NAME", "")).strip().lower()
        try:
            robot_overrides = cls.ROBOT_OVERRIDES[robot]
        except KeyError:
            raise ValueError(
                f"Unknown robot {robot!r}; expected one of {sorted(cls.ROBOT_OVERRIDES)}"
            ) from None
        return cls(**{**robot_overrides, **overrides})


@dataclass(frozen=True)
class GovernorStepInfo:
    """Diagnostics of the last update with a translation request."""

    v_now: float
    status: str
    v_probes: Tuple[float, ...]
    v_ref: float
    d_poly: float
    d_raw: float
    d_clamped: float
    d_eff: float
    dist_full: float


# ---------------------------------------------------------------------------
# Governor
# ---------------------------------------------------------------------------


class AdaptiveMaxDistGovernor:
    """Computes d_max / r_max from the directional velocity capability."""

    def __init__(
        self,
        arm: Optional[Arm_Collision_Model] = None,
        cfg: Optional[AdaptiveMaxDistConfig] = None,
        markers=None,
        robot_interface: Optional[Robot_Interface] = None,
        robot_name: Optional[str] = None,
    ):
        self.robot_name = (robot_name or os.environ.get("ROBOT_NAME", "")).strip().lower()
        self.cfg = AdaptiveMaxDistConfig.from_robot(self.robot_name) if cfg is None else cfg
        self.arm = Arm_Collision_Model(use_octomap=False) if arm is None else arm
        self.markers = RvizMarkers(namespace="AdaptiveDeviationGovernor") if markers is None else markers
        self.robot_interface = Robot_Interface() if robot_interface is None else robot_interface
        self.limits = JointLimits.from_model(
            self.arm.model, qddot_max=self.cfg.qddot_max, margin=self.cfg.q_margin
        )
        print(f"[AdaptiveMaxDistGovernor] robot={self.robot_name!r} cfg={self.cfg}")

        self._d_prev = float(self.cfg.d_max)  # original behaviour: start at d_max
        self.last_step_info: Optional[GovernorStepInfo] = None

    @property
    def d_max_current(self) -> float:
        """Current translational limit (m); cfg.d_max before the first update."""
        return self._d_prev

    @property
    def r_max_current(self) -> float:
        """Current rotational limit (rad); fixed for now."""
        return float(self.cfg.r_max)

    # --- Kinematics ------------------------------------------------------------

    def _j6(self, q: np.ndarray) -> np.ndarray:
        """Jacobian, rows [linear; angular], in the 'wt' frame."""
        return np.asarray(self.arm.get_J(q, reference="wt"), dtype=float)

    def _ee_pos(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(self.arm.get_ee_pose_for_q(q), dtype=float)[:3, 3]

    def _capacity(self, q: np.ndarray, J6: np.ndarray, d_hat: np.ndarray) -> Tuple[float, str]:
        """Max EE speed along d_hat at q, holding the orientation (strong polytope)."""
        dq_lo, dq_hi = self.limits.velocity_bounds(q)
        polytope = strong_velocity_polytope(J6, dq_lo, dq_hi)
        if polytope is None or polytope.is_flat:
            return 0.0, "fallback_zero"
        v = polytope.max_velocity_along(d_hat)
        if not np.isfinite(v) or v < 0.0:
            return 0.0, "fallback_zero"
        return float(v), "ok"

    # --- Look-ahead ----------------------------------------------------------

    def _lookahead(
        self, q: np.ndarray, J6: np.ndarray, d_hat: np.ndarray, dist_full: float, v_now: float
    ) -> Tuple[Tuple[float, ...], float]:
        """Probe capacities along d_hat and the resulting reference speed v_ref."""
        c = self.cfg
        J_pinv = damped_pinv(J6, damping=c.pinv_damping)
        twist_dir = np.concatenate((d_hat, np.zeros(3)))  # translate, hold orientation
        p0 = self._ee_pos(q)

        v_probes = []
        for dist in c.probe_distances:
            t = min(float(dist), dist_full)
            q_probe = self.limits.clip(q + J_pinv @ (t * twist_dir))
            v = self._capacity(q_probe, self._j6(q_probe), d_hat)[0]
            if t > c.eps:
                # Count the probe only as far as the linear IK actually got there
                # (near singularities / joint limits it falls short).
                reached = float(np.dot(self._ee_pos(q_probe) - p0, d_hat)) / t
                v *= float(np.clip(reached, 0.0, 1.0))
            v_probes.append(v)

        v_ahead = float(np.mean(v_probes))
        v_ref = v_now + c.trend_bonus * max(0.0, v_ahead - v_now)
        return tuple(v_probes), v_ref

    # --- Distance policy ---------------------------------------------------

    def _update_distance(
        self, v_now: float, v_ref: float, dt: Optional[float]
    ) -> Tuple[float, float, float, float]:
        """Speed -> desired deviation -> clamp -> exponential smoothing."""
        c = self.cfg
        d_poly = c.alpha_poly * v_now * c.control_dt
        d_raw = c.alpha_poly * v_ref * c.control_dt
        d_clamped = float(np.clip(d_raw, c.d_min, c.d_max))
        a = c.smoothing_alpha(shrinking=d_clamped < self._d_prev, dt=dt)
        self._d_prev = (1.0 - a) * self._d_prev + a * d_clamped
        return d_poly, d_raw, d_clamped, self._d_prev

    def plot_projection_sphere(self, distance: float) -> None:
        """Visualize the d_max sphere around the measured EE in RViz."""
        p = self.robot_interface.get_ee_pos()
        T = self.robot_interface.get_Ws_T_Wt()
        transformed = (T @ np.hstack((p, np.array([1.0]))))[:3]
        self.markers.set_marker(
            "adaptive_deviation_sphere",
            marker_type="Spheres",
            centers=[transformed],
            color=[0.6, 0.0, 0.8, 0.3],
            radii=float(distance),
            publish=True,
        )

    # --- Single public update ------------------------------------------------

    def update(
        self, desired_T44: np.ndarray, dt: Optional[float] = None, plot: bool = True
    ) -> Tuple[float, float]:
        """Update the limits from the current robot state and the desired pose.

        dt: actual period between calls (s); None = cfg.control_dt. Only the
        EMA factor depends on it. Returns (d_max, r_max).
        """
        p_des = np.asarray(desired_T44, dtype=float).reshape(4, 4)[:3, 3]
        self.arm.update_robot_config()
        q = self.arm.q.copy()
        delta = p_des - self._ee_pos(q)
        distance = float(np.linalg.norm(delta))
        self.last_step_info = None

        if distance >= self.cfg.eps:  # no translation requested: keep the filter state
            direction = delta / distance
            J6 = self._j6(q)
            v_now, status = self._capacity(q, J6, direction)
            v_probes, v_ref = self._lookahead(q, J6, direction, distance, v_now)
            d_poly, d_raw, d_clamped, d_eff = self._update_distance(v_now, v_ref, dt)
            self.last_step_info = GovernorStepInfo(
                v_now=v_now, status=status, v_probes=v_probes, v_ref=v_ref,
                d_poly=d_poly, d_raw=d_raw, d_clamped=d_clamped, d_eff=d_eff,
                dist_full=distance,
            )

        if plot:
            self.plot_projection_sphere(self._d_prev)
        return self.d_max_current, self.r_max_current


# ---------------------------------------------------------------------------
# LN node: continuous d_max / r_max publisher
# ---------------------------------------------------------------------------


class ManipulabilityChecker:
    """Desired EE pose from ``shared.constraints``; d_max and r_max published on
    ``manipulability.dmax``.

    d_max is the sphere radius itself (control_dt as configured). The node
    rate only changes how often it is refreshed; the EMA is converted to the
    node period inside the governor.
    """

    def __init__(self, governor: Optional[AdaptiveMaxDistGovernor] = None, debug: bool = True):
        self.ln_clnt = ln.client("ManipulabilityChecker", [])
        if governor is None:
            governor = AdaptiveMaxDistGovernor(robot_interface=Robot_Interface(self.ln_clnt))
        self.governor = governor
        self.constraints_sub = self.ln_clnt.subscribe(
            "shared.constraints", "python_modules/shared_control_modules/constraints_to_tcu"
        )
        self.dmax_pub = self.ln_clnt.publish(
            "manipulability.dmax", "python_modules/environment_awareness/manipulability/dmax"
        )
        self.debug = debug
        self._last_desired_in_Wt: Optional[np.ndarray] = None

    def run(self, rate_hz: float = 20.0) -> None:
        """Publish d_max / r_max every period; update on the latest target."""
        period = 1.0 / rate_hz
        print(
            f"[ManipulabilityChecker] robot={self.governor.robot_name} "
            f"d_max/r_max on 'manipulability.dmax' @ {rate_hz:.0f} Hz"
        )
        while True:
            loop_start = time.monotonic()

            if self.constraints_sub.read(blocking_or_timeout=0.001) is not None:
                W0_T_EE_des = simulink2tf(np.asarray(self.constraints_sub.packet.W_T_EE_des, dtype=float))
                Wt_T_W0 = np.linalg.inv(self.governor.robot_interface.get_W0_T_Wt())
                self._last_desired_in_Wt = Wt_T_W0 @ W0_T_EE_des
            if self._last_desired_in_Wt is not None:
                self.governor.update(self._last_desired_in_Wt, dt=period)

            d_max = self.governor.d_max_current
            r_max = self.governor.r_max_current
            self.dmax_pub.packet.d_max = d_max
            self.dmax_pub.packet.r_max = r_max
            self.dmax_pub.write()
            if self.debug:
                info = self.governor.last_step_info
                detail = "no-motion" if info is None else (
                    f"dist={info.dist_full:.4f}m v_now={info.v_now:.3f}m/s "
                    f"v_probes={np.round(info.v_probes, 3).tolist()} v_ref={info.v_ref:.3f}m/s "
                    f"d_poly={info.d_poly:.4f} d_raw={info.d_raw:.4f} d_clamped={info.d_clamped:.4f} "
                    f"{info.status}"
                )
                print(
                    f"[ManipulabilityChecker] d_max={d_max:.4f}m r_max={np.rad2deg(r_max):.1f}deg "
                    f"target={'yes' if self._last_desired_in_Wt is not None else 'none'} {detail}"
                )

            elapsed = time.monotonic() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)


if __name__ == "__main__":
    ManipulabilityChecker().run()