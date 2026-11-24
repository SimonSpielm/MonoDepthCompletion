import time

import numpy as np
from collections import deque

import links_and_nodes as ln

from python_modules.environment_awareness.arm_collision_model import Arm_Collision_Model
from python_modules.environment_awareness.smm_interface import SMMInterface
from python_modules.environment_awareness.robot_interface import Robot_Interface
from python_modules.environment_awareness.arm_collision_avoidance.service.cm_svc import ServiceProvider
from robotutils.ros.markers import RvizMarkers


class LastValidBuffer(deque):
    def __init__(self, maxlen=50, dist_thresh=0.005, angle_thresh_deg=1.0):
        super().__init__(maxlen=maxlen)
        self.dist_thresh = dist_thresh
        self.angle_thresh = np.deg2rad(angle_thresh_deg)

        self.last_valid_T = None
        self.last_valid_elbow = None

    def append(self, T, elbow):
        """
        T:     4x4 Transformationsmatrix
        elbow: (3,) oder (3,1) Ellbogenposition im Weltframe
        """
        self.last_valid_T = T
        self.last_valid_elbow = elbow

        if len(self):
            last_T, _ = self[-1]
            if not self._is_significant_change(last_T, T):
                return

        print("[Last Valid Buffer] Appended")
        return super().append((T, elbow))

    def _is_significant_change(self, A, B):  # TODO test this
        # Translation
        tA, tB = A[:3, 3], B[:3, 3]
        dist = float(np.linalg.norm(tB - tA))

        # Rotation
        R_rel = A[:3, :3].T @ B[:3, :3]
        cos_angle = (np.trace(R_rel) - 1.0) / 2.0
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        return (dist >= self.dist_thresh) or (angle >= self.angle_thresh)


class ArmCollisionManager:
    def __init__(self):
        self.ln_clnt = ln.client("ArmCollisionManager", [])
        self.markers = RvizMarkers(namespace="Arm_Manager")

        self.pin_robot = Arm_Collision_Model(use_octomap=True)
        self.smm_robot = SMMInterface(self.pin_robot, markers=self.markers)
        self.robot_interface = Robot_Interface(self.ln_clnt)
        self.last_valid_buffer = LastValidBuffer()
        self.svc = ServiceProvider(self)

        self.attractor_pub = self.ln_clnt.publish(
            "arm_collision_avoidance.elbow_attractor",
            "python_modules/environment_awareness/arm_collision_avoidance/elbow_attractor",
        )
        self.safety_config = self.ln_clnt.subscribe(
            "user_config_handler.safety_config", "python_modules/periphery/safety_config"
        )

    def check_collisionfree_target(self, target: np.ndarray, update_octomap=True, update_buffer=True):
        """
        checks if a given endeffector pose would result in collision

        target MUST be given in W0 !

        returns:
            True  -> collisionfree
            False -> collision
        """
        Ws_T_W0 = self.robot_interface.get_Ws_T_W0()
        Wt_T_W0 = np.linalg.inv(self.robot_interface.get_W0_T_Wt())
        Ws_T_target = Ws_T_W0 @ target
        Wt_T_target = Wt_T_W0 @ target

        valid, q_new = self.pin_robot.check_collisionfree_target(Wt_T_target, update_octomap)

        if valid and q_new is not None:
            if update_buffer:
                current_elbow = self.robot_interface.get_elbow_positon()
                self.last_valid_buffer.append(Ws_T_target, current_elbow)
            return True
        return False

    def get_last_valid(self):
        print("[Recovery] Get Last Valid needed")
        if self.last_valid_buffer.last_valid_T is not None:
            Wt_T_lastvalid = (
                np.linalg.inv(self.robot_interface.get_Ws_T_Wt()) @ self.last_valid_buffer.last_valid_T
            )  # stored in ws, check in Wt
            if self.check_collisionfree_target(Wt_T_lastvalid, update_buffer=False):
                return self.robot_interface.get_W0_T_Ws() @ self.last_valid_buffer.last_valid_T  # fii in w0

        print("[Recovery] Latest valid not valid anymore")

        for T, _ in reversed(self.last_valid_buffer):  # T in Ws
            Wt_T_lastvalid = np.linalg.inv(self.robot_interface.get_Ws_T_Wt()) @ T  # check in Wt
            print("[Recovery] Check last valid one before")
            if self.check_collisionfree_target(Wt_T_lastvalid, update_buffer=False):
                return self.robot_interface.get_W0_T_Ws() @ T

        found, Wt_T_free = self.recovery_strategy()
        if found:
            W0_T_free = self.robot_interface.get_W0_T_Wt() @ Wt_T_free
            return W0_T_free

        print("There is no last valid")
        return None

    def get_attractor_or_none(self):
        try:
            return self.smm_robot.get_attractor()
        except Exception as e:
            print(f"attractor calculation failed:\n {e} ")
            return None

    def recovery_strategy(self):
        """
        sample in Wt
        """
        print("[Recovery] Spehere Sampling strategy")
        T_center = self.robot_interface.get_Wt_T_msr()

        samples_per_radius = 20
        r0, r_step, r_max = 0.01, 0.01, 0.05
        max_total = 300

        def sample_T_in_ball(Tc, radius):
            d = np.random.normal(size=3)
            d /= np.linalg.norm(d) + 1e-12
            r = radius * (np.random.rand() ** (1.0 / 3.0))

            T = Tc.copy()
            T[:3, 3] = Tc[:3, 3] + r * d
            return T

        last_radius = None

        for i in range(max_total):
            radius = min(r0 + (i // samples_per_radius) * r_step, r_max)

            if radius != last_radius:
                print(f"[Recovery] Sampling radius = {radius:.3f} m")
                last_radius = radius

            T_try = sample_T_in_ball(T_center, radius)

            if self.check_collisionfree_target(T_try, update_buffer=False):
                print(f"[Recovery] Solution found at radius {radius:.3f} m after {i+1} samples")
                return True, T_try

        print("[Recovery] No solution found")
        return False, T_try

    def run(self, rate_hz: float = 25.0) -> None:
        """Handle services and publish the elbow attractor continuously."""
        period = 1.0 / rate_hz
        print(
            f"[ArmCollisionManager] elbow attractor on "
            f"'arm_collision_avoidance.elbow_attractor' @ {rate_hz:.0f} Hz"
        )
        while True:
            loop_start = time.monotonic()
            self.ln_clnt.wait_and_handle_service_group_requests(None, 0.001)

            self.safety_config.read(-1)
            active = bool(self.safety_config.packet.elbow_obstacle_avoidance)

            self.attractor_pub.packet.active = int(active)
            if active:
                attractor = self.get_attractor_or_none()
                if attractor is not None:
                    self.attractor_pub.packet.position = np.asarray(attractor, dtype=float).reshape(3)
                    self.attractor_pub.packet.valid = 1
                else:
                    self.attractor_pub.packet.valid = 0
            else:
                self.attractor_pub.packet.valid = 0
            self.attractor_pub.write()

            elapsed = time.monotonic() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)


if __name__ == "__main__":
    ArmCollisionManager().run()