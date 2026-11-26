#!/usr/bin/env python

import numpy as np
import time

import links_and_nodes as ln
import odb_ln_interface.ln_interface as ln_odb
import world_representation.ln_interface
from motion_interface.ln_client import Moti, World

from robotutils.base.print_tools import print_red
from robotutils.geom.se3 import compact2tf, simulink2tf, tf2euler
from robotutils.ros.markers import RvizMarkers

from python_modules.anchoring.ln_interface import AnchoringLnInterface
from python_modules.controller_interfaces.frame_interpolator.ln_interface import FII_LnInterface
from python_modules.state_machine.ln_interface import TSM_LnInterface

from navigation.trajectory_planner.trajectory_commands import Command, ExecutionMode, Trajectory
from navigation.wheelchair_interface.wheelchair_utils import fill_arrays

class WheelchairInterface:
    def __init__(self, ln_client):
        self.ln_clnt = ln_client
        self.moti = Moti(ln_clnt=self.ln_clnt)
        self.wsr = world_representation.ln_interface.ln_interface(clnt=self.ln_clnt, name="maya_world")
        self.world = World(self.wsr, self.moti)
        self.odb = ln_odb.ln_interface(clnt=self.ln_clnt)
        self.fii_svc = FII_LnInterface(ln_clnt=self.ln_clnt)
        self.tsm_svc = TSM_LnInterface(ln_clnt=self.ln_clnt)
        self.anchoring_svc = AnchoringLnInterface(ln_clnt=self.ln_clnt)
        self.position_controller_command = self.ln_clnt.publish(
            "position_controller_commands",
            "navigation/position_controller_commands",
        )
        self.ln_inport_position_controller_infos = self.ln_clnt.subscribe(
            "position_controller_infos",
            "navigation/position_controller_infos",
        )
        self.ln_inport_modes = self.ln_clnt.subscribe("tsm.active_task", "python_modules/shared_control_modules/active_task")
        self.ln_inport_wbc_mode = self.ln_clnt.subscribe("tsm.control_parameters", "python_modules/shared_control_modules/control_parameters")
        self.ln_inport_Ws_T_Wt = self.ln_clnt.subscribe("Ws_to_Wt_no_reset", "ln/frame34")
        self.ln_inport_target_object = self.ln_clnt.subscribe(
            "anchoring.task_target_info", "python_modules/anchoring/task_target_info"
        )
        self.ln_inport_telemetry = self.ln_clnt.subscribe(
            "smart_assist_telemetry", "simulink/robot_model/smart_assist_telemetry"
        )
        self.ln_inport_sct_state = self.ln_clnt.subscribe(
            "shared.constraints", "python_modules/shared_control_modules/constraints_to_tcu"
        )

        self.C_T_W = np.array(
            [
                [-1.0000, 0, 0, -0.0400],
                [0, -1.0000, 0, -0.3160],
                [0, 0, 1.0000, 0],
                [0, 0, 0, 1.0000],
            ]
        )
        self.W_T_C = np.linalg.inv(self.C_T_W)

    def state_reached(self, command: Command) -> None:
        print(f"State reached: {command.state_name}")
        req_kwargs = {"original_frame_name"}
        kwargs = {}
        for kwarg in req_kwargs:
            kwargs[kwarg] = getattr(command, kwarg)
        if kwargs:
            self.fii_svc.trigger(trigger="wheelchair_trajectory_planner", trigger_kwargs=kwargs)

    def plot_door_frame(self, Ws_T_Obj: np.array) -> None:
        if "door_frame" in self.world.wsr.list_objects():
            self.world.remove_object("door_frame")
        self.world.add_object("door_frame", "door_frame", Ws_T_Obj)

    def plot_trajectory(self, trajectory: Trajectory) -> None:
        rviz = RvizMarkers()
        x = trajectory.x_path
        y = trajectory.y_path
        points = np.array([np.array([x, y, 0]) for x, y in zip(x, y)])
        rviz.set_marker(
            name="traj",
            marker_type="Line",
            points=points,
            color=[1, 0, 0],
            scale=0.02,
        )

    def get_sct_state(self):
        self.ln_inport_sct_state.read(-1)
        sct_state = self.ln_inport_sct_state.packet.state_name.rstrip()
        return sct_state

    def get_obj_T_KF(self, obj, parsed_frame):
        """Return the 4x4 transformation matrix or None if the frame is undefined."""
        obj_type = obj.split("$")[0]
        kf = parsed_frame.split(".")[-1]

        try:
            values = self.odb.list_property(obj_type, kf)
            return np.array(values[0].split(), dtype=float).reshape(4, 4)

        except (IndexError, KeyError):
            print_red(f"Frame '{parsed_frame}' not defined for '{obj}'")
            return None

    def get_Ws_T_Obj_from_wsr(self, obj):
        topic = self.ln_clnt.subscribe(f"maya_world.{obj}.frame", "ln/frame34")
        try:
            topic.read(0.2)
            return compact2tf(topic.packet.frame)
        except Exception:
            print_red(f"Object pose not published: edan_world.{obj}")

    def get_Ws_T_Obj_from_anchoring(self):
        """
        get Ws_T_Object from target_object topic (usually publsihed by anchoring)

        :return: (4x4) Ws_T_Object
        """
        self.ln_inport_target_object.read(-1)
        compact = self.ln_inport_target_object.packet.filtered_tf
        if np.allclose(compact, np.zeros((12,))):
            print_red("Anchoring target is empty, aborting")
            return
        Ws_T_Obj = compact2tf(compact)
        if self.ln_inport_target_object.packet.active == 1:
            return Ws_T_Obj
        else:
            print_red("Task not active anymore")
            return

    def get_Ws_T_C(self, controller_offset=0):
        """
        get odometry published by platform model

        :return Ws_T_C: odometry (4x4)
        """
        self.ln_inport_Ws_T_Wt.read()
        W_T_C_offset = self.W_T_C.copy()
        W_T_C_offset[0, 3] -= controller_offset
        Ws_T_Wt = simulink2tf(self.ln_inport_Ws_T_Wt.packet.frame)
        return Ws_T_Wt @ self.W_T_C

    def wait_until_status(self, status):
        """
        wait until desired status of position controler is reached in platform model
                - -1: tcm!=3 or/and wcm!=2 -> no execution possible
                -  0: ready for new command
                -  1: finished task
                -  2: executing

        :param status: (int) desired status to be reached
        :type status: int
        """

        while not self.check_status(status):
            if not self.task_still_active():
                print("Task not active anymore")
                return False
            self.ln_clnt.wait_and_handle_service_group_requests(None, 0.1)
        return True

    def s_reached(self, s):
        self.ln_inport_position_controller_infos.read()
        return self.ln_inport_position_controller_infos.packet.progress > s

    def task_still_active(self):
        self.ln_inport_position_controller_infos.read()
        return self.ln_inport_position_controller_infos.packet.status != -1

    def check_status(self, status):
        """
        check if desired status of  position controler is reached

                - -1: tcm!=3 or/and wcm!=2 -> no execution possible
                -  0: ready for new command
                -  1: finished task
                -  2: executing

        :param status: desired status to be reached
        :type status: int

        :return: (boolean) is current platform model head/rot-controller control status or not
        """
        self.ln_inport_position_controller_infos.read()
        return self.ln_inport_position_controller_infos.packet.status == status

    def set_wheelchair_task(self, command):
        if command.set_task:
            self.tsm_svc.set_task(
                actor="wheelchair",
                target_device="wheelchair",
                name=command.mode.name,
                target_control_mode="position",
                odometry_active=0
        )
        
    def set_manual_wheelchair_driving(self):
        self.tsm_svc.set_task(
            actor="wheelchair",
            target_device="wheelchair",
            name="wheelchair_driving",
            target_control_mode="manual",
            odometry_active=0
        )
    
    def set_to_default_task(self):
            self.tsm_svc.set_task(
                actor="right_arm",
                target_device = "arm",
                name="DirectControl",
                target_control_mode="frame",
            )

    def set_wheelchair_modes(self, command):
        """
        initial activation of modi to execute trajectory
        """
        self.reset_topic()
        self.set_wheelchair_task(command)

    def check_precision(self, Ws_T_despose, thres_pos=0.25, thres_yaw=3):
        """
        check if target pose is reached wihtin thresholds

        :param Ws_T_despose: target pose in Ws
        :type Ws_T_despose: np.array (4x4)

        :return boolean
        """
        C_T_Ws = np.linalg.inv(self.get_Ws_T_C())
        C_T_despose = C_T_Ws @ Ws_T_despose
        euler = tf2euler(C_T_despose)
        d_pos = np.linalg.norm(euler[:2])
        d_yaw = np.rad2deg(euler[5])

        if d_pos <= thres_pos and d_yaw <= thres_yaw:
            return True
        else:
            print("actual orientation not within target zone")
            return False

    def update_wbc_gain(self, wbc_command_gain: float):
        self.position_controller_command.packet.wbc_command_gain = wbc_command_gain
        self.position_controller_command.write()
        print(f"Updated wbc_command_gain: {wbc_command_gain}")

    
    def load_objects(self):
        self.anchoring_svc.update_world_model()

    def send_trajectory(
        self,
        trajectory: Trajectory,
        mode: ExecutionMode = ExecutionMode.path,
        wbc_command_gain: float = 1,
        automatic: bool = 0,
        message_length: int = 1000,
        plot: bool = True,
    ):
        """
        send trajectory to  position controlle

        :param trajectory: Trajectory to be sent to the controller
        :param samples: amount of viapoints contained by (fixed length) arrays
        :param mode: execution mode
        """
        assert trajectory.valid(), "Sending to controller failed: trajectory not valid"

        print("Sending path...")
        if plot:
            self.plot_trajectory(trajectory)

        samples = trajectory.length()
        path_x, path_y, distances, time_scaling = fill_arrays(
            trajectory.x_path, trajectory.y_path, trajectory.distances, trajectory.timestamps, message_length=message_length
        )

        self.position_controller_command.packet.x_path = path_x
        self.position_controller_command.packet.y_path = path_y
        self.position_controller_command.packet.distances = distances
        self.position_controller_command.packet.time_scaling = time_scaling
        self.position_controller_command.packet.samples = samples
        self.position_controller_command.packet.target_orientation = trajectory.yaw_end
        self.position_controller_command.packet.trigger = 1
        self.position_controller_command.packet.mode = mode.value
        self.position_controller_command.packet.wait = 0
        self.position_controller_command.packet.wbc_command_gain = wbc_command_gain
        self.position_controller_command.packet.automatic = bool(automatic)
        
        self.wait_until_status(0)
        self.position_controller_command.write()
        print("Path sent")

    def send_heading_command(self, heading, mode=ExecutionMode.heading):
        """
        send heading commands

        :param heading: desired heading distance
        :type heading: float
        :param mode: execution mode
        :type mode: ExecutionMode
        """
        self.position_controller_command.packet.heading = heading
        self.position_controller_command.packet.rotation = 0
        self.position_controller_command.packet.trigger = 2
        self.position_controller_command.packet.mode = mode.value
        self.position_controller_command.packet.wait = 0
        self.position_controller_command.write()

    def send_wait(self, value=0):
        self.position_controller_command.packet.wait = value
        self.position_controller_command.write()

    def send_rotation_command(self, rotation, mode=ExecutionMode.rotation):
        """
        send rotation commands

        :param rotation: desired rotation in degrees
        :type rotation: float
        :param mode: execution mode
        :type mode: ExecutionMode
        """
        self.position_controller_command.packet.heading = 0
        self.position_controller_command.packet.rotation = rotation
        self.position_controller_command.packet.trigger = 1
        self.position_controller_command.packet.mode = mode.value
        self.position_controller_command.packet.wait = 0
        self.position_controller_command.write()

    def reset_topic(self, max_samples=1000):
        """
        reset position controller commands

        :param max_samples: length of arrays contained by message
        :type max_samples: int
        """
        self.position_controller_command.packet.x_path = np.zeros(max_samples)
        self.position_controller_command.packet.y_path = np.zeros(max_samples)
        self.position_controller_command.packet.distances = np.zeros(max_samples)
        self.position_controller_command.packet.time_scaling = np.zeros(max_samples)
        self.position_controller_command.packet.samples = 0
        self.position_controller_command.packet.target_orientation = 0
        self.position_controller_command.packet.trigger = 0
        self.position_controller_command.packet.heading = 0
        self.position_controller_command.packet.rotation = 0
        self.position_controller_command.packet.mode = 0
        self.position_controller_command.packet.wait = 0
        self.position_controller_command.packet.automatic = 0
        self.position_controller_command.write()
        
        while self.check_status(1):
            print("Wait for reset of controller")
            

    def wait_until_modi_activated(self, timeout: float = 5.0) -> bool:
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            self.ln_inport_modes.read()
            self.ln_inport_wbc_mode.read()

            modes = self.ln_inport_modes.packet
            wbc = self.ln_inport_wbc_mode.packet

            if (modes.wheelchair_control_mode == 2 and
                    modes.target_control_mode == 3) or wbc.wbc_mode == 7:
                return True
            self.ln_clnt.wait_and_handle_service_group_requests(None, 0.1)

        print_red("Timeout! Skill and Planning out of sync!")
        return False