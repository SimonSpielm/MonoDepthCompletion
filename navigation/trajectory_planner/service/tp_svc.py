import json
import math
import traceback
from dataclasses import fields
import numpy as np

from robotutils.geom.se3 import construct_transformation_matrix

from navigation.trajectory_planner.trajectory_commands import Command, ExecutionMode



class ServiceProvider:
    def __init__(self, trajectory_manager, name="TrajectoryPlanner"):
        self.manager = trajectory_manager
        svc1 = self.manager.ln_clnt.get_service_provider(
            f"{name}.traj_to_pose",
            "navigation/traj_to_pose_svc",
        )
        svc1.set_handler(self.svc_traj_to_pose)
        svc1.do_register()

        svc2 = self.manager.ln_clnt.get_service_provider(
            f"{name}.rotate",
            "navigation/rotation_service",
        )
        svc2.set_handler(self.svc_rotation)
        svc2.do_register()

        svc3 = self.manager.ln_clnt.get_service_provider(
            f"{name}.drive",
            "navigation/heading_service",
        )
        svc3.set_handler(self.svc_heading)
        svc3.do_register()

        svc4 = self.manager.ln_clnt.get_service_provider(
            f"{name}.sct_commands",
            "navigation/sct_commands",
        )
        svc4.set_handler(self.svc_sct_commands)
        svc4.do_register()

    def _respond(self, svc, req, resp):
        svc.respond()
        resp.result = None
        resp.error_message = ""
        resp.error_type = ""

    def svc_rotation(self, svc, req, resp) -> None:
        try:
            rot = float(req.rotation)
            assert -180 <= rot <= 180, "commanded rotation out of [-180,180]"
            if self.manager.commandsFIFO.is_empty() and self.manager.current_command is None and self.manager.interface.task_still_active():
                command = Command(state_name="rotation", mode=ExecutionMode.rotation, rotation=rot, reset_task_to_default=True, set_task=True)
                self.manager.commandsFIFO.put(command)
            else:
                print("[Service] TP is busy - Abort command")
        except Exception:
            ex = traceback.format_exc()
            resp.error_message = ex
        finally:
            print("[Service] Rotation: " + str(req.rotation))
            self._respond(svc, req, resp)

    def svc_heading(self, svc, req, resp) -> None:
        try:
            head = float(req.heading)
            if self.manager.commandsFIFO.is_empty() and self.manager.current_command is None and self.manager.interface.task_still_active():
                command = Command(state_name="heading", mode=ExecutionMode.heading, heading=head, reset_task_to_default=True, set_task=True)
                self.manager.commandsFIFO.put(command)
            else:
                print("[Service] TP is busy - Abort command")
        except Exception:
            ex = traceback.format_exc()
            resp.error_message = ex
        finally:
            print("[Service] Heading: " + str(req.heading))
            self._respond(svc, req, resp)

    def svc_traj_to_pose(self, svc, req, resp) -> None:
        try:
            print("[Service] Service Call recieved ")
            if self.manager.commandsFIFO.is_empty() and self.manager.current_command is None and self.manager.interface.task_still_active():
                mode = ExecutionMode.obstacle_avoidance if req.obstacle_avoidance else ExecutionMode.path
                if mode is not ExecutionMode.nothing:
                    if req.type == "relative":
                        target_pose = self.calc_target_pose_of_xytheta(req.x_des, req.y_des, req.theta_des)
                    else:
                        target_pose = self.calc_absolute_pose(req.x_des, req.y_des, req.theta_des)
                    command = Command(state_name="service", mode=mode, parsed_frame=target_pose, reset_task_to_default=True, set_task=True) #state_name = service -> trigger to not wait for specific state
                    self.manager.commandsFIFO.put(command)
                    print(f"[Service] called trajectory planner, Command: {command}")
            else:
                print("[Service] TP is busy - Abort command")
        except Exception:
            ex = traceback.format_exc()
            resp.error_message = ex
        finally:
            self._respond(svc, req, resp)

    def svc_sct_commands(self, svc, req, resp) -> None:
        try:
            command_list = json.loads(req.commands)
            command_fields = {field.name for field in fields(Command)}
            print(f"Recived service request:\n{command_list}")
            if self.manager.commandsFIFO.is_empty() and self.manager.current_command is None and self.manager.interface.task_still_active():

                for state_name, attributes in command_list.items():
                    command_attrs = {'state_name': state_name}
                    extra_attrs = {}
                    mode = ExecutionMode.path

                    for attr_name, value in attributes.items():
                        if attr_name in command_fields:
                        # special processing for certain attributes
                            if attr_name == "parsed_frame" and isinstance(value, list):
                                value = np.array(value, dtype=float)
                                if value.shape == (3,):
                                    # theta in rad (SCT format) -> deg for the platform model
                                    value[2] = math.degrees(float(value[2]))
                            elif attr_name == "wbc_command_gain" and value == 0:
                                if self.manager.commandsFIFO.not_empty():
                                    value = self.manager.commandsFIFO[-1].wbc_command_gain
                                else:
                                    value = 1
                            elif attr_name == "mode":
                                mode = ExecutionMode.get_execution_mode(value)
                                if mode is ExecutionMode.nothing:
                                    raise ValueError(f"unknown execution mode '{value}' in state '{state_name}'")
                                continue
                            command_attrs[attr_name] = value
                        else:
                            if attr_name == "obstacle_avoidance":
                                if value:
                                    mode = ExecutionMode.obstacle_avoidance
                            else:
                                extra_attrs[attr_name] = value

                    # special case for Update Only WBC Gain. This is a hack: Remove this with the TODO in sct/external.py
                    if extra_attrs.get("update_only_wbc_gain"):
                        command_attrs["original_frame_name"] = "update_wbc"
                        command_attrs["parsed_frame"] = "update_wbc"

                    # [x, y, theta] poses (m, deg) coming from go_to_static / go_to_relative:
                    parsed_frame = command_attrs.get("parsed_frame")
                    if isinstance(parsed_frame, np.ndarray) and parsed_frame.shape == (3,):
                        if mode is ExecutionMode.heading:
                            # heading: drive straight by the x distance (m)
                            command_attrs["heading"] = float(parsed_frame[0])
                        elif mode is ExecutionMode.rotation:
                            # rotation: in place by the theta angle (deg)
                            command_attrs["rotation"] = float(parsed_frame[2])
                        elif not extra_attrs.get("relative", False):
                            # go_to_static: build the absolute pose matrix here
                            command_attrs["parsed_frame"] = construct_transformation_matrix(
                                x=float(parsed_frame[0]), y=float(parsed_frame[1]), z_rot_deg=float(parsed_frame[2])
                            )
                        # go_to_relative + path mode: keep the 3-vector, it is composed with the
                        # start pose at execution time (trajectory_manager.get_target_pose)
                    elif mode in (ExecutionMode.heading, ExecutionMode.rotation) and not command_attrs.get(mode.name):
                        # no [x, y, theta] pose and no explicit heading/rotation value
                        raise ValueError(
                            f"state '{state_name}': mode '{mode.name}' requires a [x, y, theta] pose for 'go_to_relative'"
                        )

                    if mode in (ExecutionMode.heading, ExecutionMode.rotation):
                        if "automatic" in command_attrs and not command_attrs["automatic"]:
                            raise ValueError(
                                f"state '{state_name}': 'automatic' can only be disabled in path mode, "
                                f"mode '{mode.name}' is always automatic"
                            )
                        command_attrs["automatic"] = True

                    # Erstellen des Command-Objekts mit gesammelten Attributen
                    command = Command(mode=mode, **command_attrs)
                    self.manager.commandsFIFO.put(command)

            else:
                print("[Service] TP is busy - Abort command")
        except Exception:
            ex = traceback.format_exc()
            resp.error_message = ex
        finally:
            self._respond(svc, req, resp)

    def calc_absolute_pose(self, x: float, y: float, theta: float) -> np.ndarray:
        Ws_T_despose = construct_transformation_matrix(x=x, y=y, z_rot_deg=theta)
        return Ws_T_despose

    def calc_target_pose_of_xytheta(self, x: float, y: float, theta: float) -> np.ndarray:
        C_T_despose = construct_transformation_matrix(x=x, y=y, z_rot_deg=theta)
        Ws_T_C = self.manager.interface.get_Ws_T_C()
        Ws_T_despose = Ws_T_C @ C_T_despose
        return Ws_T_despose