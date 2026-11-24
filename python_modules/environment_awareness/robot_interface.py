import numpy as np
import time

import links_and_nodes as ln
from robotutils.geom.se3 import simulink2tf
from robotutils.geom.utils import invert_transformation
from robotutils.ros.markers import RvizMarkers
import world_representation.ln_interface


class Robot_Interface:
    def __init__(self, ln_clnt=None, name="RobotInterface"):
        if ln_clnt is None:
            self.ln_clnt = ln.client(name, [])
        else:
            self.ln_clnt = ln_clnt

        self.wsr = world_representation.ln_interface.ln_interface(
            clnt=self.ln_clnt,
            name="edan_world",
        )

        self.inport_telemetry = self.ln_clnt.subscribe(
            "smart_assist_telemetry",
            "simulink/robot_model/smart_assist_telemetry",
        )
        self.inport_robotkin = self.ln_clnt.subscribe("robot_kin", "simulink/high_level/robot_kin")
        self.inport_odometry = self.ln_clnt.subscribe("Ws_to_Wt_no_reset", "ln/frame34")

        self.motion_svc = self.ln_clnt.get_service(
            "The_state_machine.svc_execute_joint_motion", "python_modules/shared_control_modules/execute_joint_motion"
        )

    def get_q(self) -> np.ndarray:  # TODO check maya
        self.inport_telemetry.read(-1)
        q = self.inport_telemetry.packet.q_msr
        return q

    def get_Wt_T_msr(self) -> np.ndarray:
        self.inport_telemetry.read(-1)
        Wt_T_msr = simulink2tf(self.inport_telemetry.packet.Wt_T_msr)
        return Wt_T_msr

    def get_ee_pos(self) -> np.ndarray:
        Wt_T_msr = self.get_Wt_T_msr()
        return Wt_T_msr[:3, 3]

    def get_Ws_T_Wt(self) -> np.ndarray:
        self.inport_odometry.read(-1)
        Ws_T_Wt = simulink2tf(self.inport_odometry.packet.frame)
        if np.any(Ws_T_Wt[:3, :3]):
            return Ws_T_Wt
        return np.eye(4)

    def get_elbow_positon(self) -> np.ndarray:
        self.inport_robotkin.read(-1)
        T = simulink2tf(self.inport_robotkin.packet.link5)  # TODO check maya
        return T[:3, 3]

    def get_1_T_W(self) -> np.ndarray:
        self.inport_robotkin.read()
        return invert_transformation(simulink2tf(self.inport_robotkin.packet.link2))

    def get_interpolator_frame(self) -> np.ndarray:
        self.inport_telemetry.read(-1)
        return simulink2tf(self.inport_telemetry.packet.Wt_T_d)

    def get_Ws_T_W0(self) -> np.ndarray:
        self.inport_telemetry.read(-1)
        W0_T_Wt = simulink2tf(self.inport_telemetry.packet.W0_T_Wt)
        Ws_T_Wt = simulink2tf(self.inport_telemetry.packet.Ws_T_Wt)
        return Ws_T_Wt @ np.linalg.inv(W0_T_Wt)

    def get_W0_T_Ws(self) -> np.ndarray:
        return np.linalg.inv(self.get_Ws_T_W0())

    def get_W0_T_Wt(self) -> np.ndarray:
        self.inport_telemetry.read(-1)
        return simulink2tf(self.inport_telemetry.packet.W0_T_Wt)

    def get_Ws_T_EE(self) -> np.ndarray:
        Wt_T_EE = self.get_Wt_T_msr()
        Ws_T_Wt = self.get_Ws_T_Wt()
        return Ws_T_Wt @ Wt_T_EE

    def execute_path(self, q_path=None):
        """
        Execute the planned joint path via the robot interface service.

        Args:
            q_path: Joint path to execute. If None, uses last planned path.

        Returns:
            bool: True if execution succeeded, False otherwise
        """

        if q_path is None or len(q_path) == 0:
            print("[Planner] No valid path to execute.")
            return False

        self.motion_svc.req.path = q_path
        self.motion_svc.call()

        if self.motion_svc.resp.error_message_len:
            print(f"[Planner] Execution error: {self.motion_svc.resp.error_message}")
            return False

        print("[Planner] Path executed successfully")
        return True

    def visualize_path(self, path=None, markers=None):
        """
        Visualize the planned path in RViz.

        Args:
            q_path: Joint path to visualize. If None, uses last planned path.
        """
        if markers is None:
            markers = RvizMarkers()
        if path is None:
            print("[Planner] No path to visualize.")
            return

        for i, T in enumerate(path):
            markers.set_marker(
                name=f"t{i}",
                marker_type="CoordinateSystem",
                pose=T,
                scale=0.005,
            )
            time.sleep(0.02)

        print(f"[Planner] Visualized {len(path)} poses")

    def get_Ws_T_Obj(self, object_name: str):
        entry = self.wsr.object_store.get(object_name)
        if entry is None:
            print(f"[RobotInterface] [Warning] Object {object_name} not in WSR")
            raise Exception(f"[RobotInterface] [Warning] Object {object_name} not in WSR")
            return None

        frame = entry.get("relative_frame", None)  # Todo check if in Ws
        if frame is None:
            return None
        return np.asarray(frame, dtype=np.float64).reshape(4, 4)

    def get_Wt_T_Obj(self, object_name: str):
        Ws_T_Obj = self.get_Ws_T_Obj(object_name)
        if Ws_T_Obj is None:
            return None
        return invert_transformation(self.get_Ws_T_Wt()) @ Ws_T_Obj