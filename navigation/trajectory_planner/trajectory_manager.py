import time
import numpy as np
import rclpy

import links_and_nodes as ln
from robotutils.base.print_tools import print_red
from robotutils.geom.utils import build_tf
from robotutils.geom.se3 import construct_transformation_matrix, tf2euler

from navigation.trajectory_planner.service.tp_svc import ServiceProvider
from navigation.trajectory_planner.trajectory_commands import Command, CommandFIFO, ExecutionMode, Trajectory
from navigation.trajectory_planner.trajectory_planner import TrajectoryPlanner
from navigation.wheelchair_interface.wheelchair_interface import WheelchairInterface
from navigation.wheelchair_interface.wheelchair_utils import valid_parsed_frame


class TrajectoryManager:
    def __init__(self):
        self.commandsFIFO = CommandFIFO()
        self.current_command = None  # command currently being executed (busy state for service guards)
        self.MAX_SAMPLES = 1000
        self.go_on = True

        self.ln_clnt = ln.client("Trajectory_Manager", [])

        self.trajectory_planner = TrajectoryPlanner(self.commandsFIFO)
        self.interface = WheelchairInterface(self.ln_clnt)
        self.svc = ServiceProvider(self)

        self.plot_trajectory = True

        self.ln_inport_target_object = self.ln_clnt.subscribe(
            "anchoring.task_target_info", "python_modules/anchoring/task_target_info"
        )

    def wait_for_sct_state(self, command: Command) -> bool:
        """
        wait until the active SCT state matches the command's state.

        raw service calls (set_task=True) are not part of an SCT task and are passed through immediately.
        while waiting, the wheelchair is kept in hold (wait=1).

        :param command: the command to wait for
        :return: True if the state is active (or no wait needed), False if the task was aborted
        """
        if command.set_task:
            return self.go_on  # raw service calls still respect an aborted task, as the former `while self.go_on` guard did
        while self.go_on:
            sct_state = self.interface.get_sct_state()
            if sct_state == command.state_name:
                return True
            if sct_state == "default" or sct_state == "":
                print("sct state was default or empty -> something could be wrong with fii")
                self.go_on = False
                return False
            if self.interface.task_still_active():
                self.interface.send_wait(1)
                print(f"Wait until next state: {command.state_name}")
                time.sleep(0.1)
            else:
                print("Task interuption - probably neccecary modes not activated")
                self.go_on = False
                return False
            self.ln_clnt.wait_and_handle_service_group_requests(None, 0.1)
        return False

    def rotation(self):
        command = self.commandsFIFO.pop()
        self.current_command = command
        if command.mode == ExecutionMode.rotation:
            rot = command.rotation
            assert -180 <= rot <= 180, "commanded rotation out of [-180,180]"
            if not self.wait_for_sct_state(command):
                return False
            self.interface.send_rotation_command(rot, mode=command.mode)
            time.sleep(0.3)
            done = self.interface.wait_until_status(1)
            self.interface.reset_topic(self.MAX_SAMPLES)
            if done and not command.set_task:
                self.interface.state_reached(command)
            print("Finished rotation")
            return done
        return False

    def heading(self):
        command = self.commandsFIFO.pop()
        self.current_command = command
        if command.mode == ExecutionMode.heading:
            head = command.heading
            if not self.wait_for_sct_state(command):
                return False
            self.interface.send_heading_command(head, mode=command.mode)
            time.sleep(0.3)
            done = self.interface.wait_until_status(1)
            self.interface.reset_topic(self.MAX_SAMPLES)
            if done and not command.set_task:
                self.interface.state_reached(command)
            print("Finished heading")
            return done
        return False

    def rotation_step(self, command: Command, Ws_T_despose: np.ndarray, max_lat_error: float = 0.2, max_rot_error: float = 1):
        # Use the target pose the trajectory was planned towards (no recomputation from anchoring)
        self.interface.reset_topic(self.MAX_SAMPLES)
        Ws_T_C = self.interface.get_Ws_T_C()
        despose_T_C = np.linalg.inv(Ws_T_despose) @ Ws_T_C
        _, dy, _, _, _, dyaw = tf2euler(despose_T_C)
        dyaw = -np.rad2deg(dyaw)
        print(f"Rot error:{dyaw},   lat error:{dy}")
        if abs(dy) > max_lat_error:
            print(f"Lateral error to big")
        else:
            if abs(dyaw) > max_rot_error:
                print(f"Rotation step necessary")
                command = Command(mode=ExecutionMode.rotation, rotation=dyaw)
                self.commandsFIFO.insert(0, command)
                self.rotation()

    def get_target_pose(self, command: Command, start_pose: np.ndarray = None):
        frame = command.parsed_frame
        target_id = command.target_id
        if isinstance(frame, np.ndarray):
            if frame.shape == (4, 4):
                frame[2, 3] = 0
                return frame
            if frame.shape != (3,):
                print_red(f"Invalid parsed_frame shape {frame.shape}, expected [x, y, theta] or a 4x4 matrix")
                return None
            # [x, y, theta] pose (m, deg) coming from go_to_relative: a 3-vector parsed_frame always
            # means 'relative to the current pose', so compose with the start pose (4x4 = absolute)
            if start_pose is None:
                start_pose = self.interface.get_Ws_T_C()
            Ws_T_pose = construct_transformation_matrix(x=float(frame[0]), y=float(frame[1]), z_rot_deg=float(frame[2]))
            return start_pose @ Ws_T_pose
        else:
            obj_type = target_id.split("$")[0]               
            Obj_T_despose = self.interface.get_obj_T_KF(target_id, frame)

            if "$" in frame:
                Ws_T_Obj = self.interface.get_Ws_T_Obj_from_wsr(target_id)
            else:
                Ws_T_Obj = self.interface.get_Ws_T_Obj_from_anchoring()

            if np.any(Ws_T_Obj[0:3, 0:3]) and np.any(Obj_T_despose):
                Ws_T_Obj[2, 3] = 0
                if obj_type in {"mug_green"}:   #target pose of these objects should lie on direct path between wc and obj
                        p_obj = Ws_T_Obj[:3, 3]
                        p_wc  = self.interface.get_Ws_T_C()[:3, 3]

                        xhat = p_obj - p_wc
                        Ws_T_Obj = build_tf(X=xhat, position=Ws_T_Obj[:3,3])

                Ws_T_despose = Ws_T_Obj @ Obj_T_despose
                return Ws_T_despose

    def handle_dubins_command(self, trajectory=None):
        if trajectory is None:
            trajectory = Trajectory()

        if self.commandsFIFO.is_empty():
            print("We should not be here")
            self.go_on = False
            return False

        command = self.commandsFIFO.pop()
        self.current_command = command
        if not self.wait_for_sct_state(command): #this keeps busy waiting for the state 
            return False
        
        if not self.interface.wait_until_modi_activated(): #this checks if wcm / tcm is activated
            return False

        if not valid_parsed_frame(command.parsed_frame):
            self.go_on = False
            return False

        start_pose = trajectory.get_final_pose() if trajectory.valid() else self.interface.get_Ws_T_C()  # TODO: check prev final

        Ws_T_despose = self.get_target_pose(command, start_pose)
        if Ws_T_despose is None:
            self.go_on = False
            return False

        if command.mode == ExecutionMode.obstacle_avoidance:
            raise NotImplementedError("Obstacle Avoidance not implemented yet")
        new_trajectory = self.trajectory_planner.get_dubins_trajectory(
            start_pose, Ws_T_despose, trajectory.v_end, command
        )
        if not new_trajectory.valid():
            print("since no path has been found, reset task to default!")
            self.commandsFIFO.last_used.reset_task_to_default = True
            self.go_on = False      # TODO handle this better if no traj is found
            return False

        trajectory.append(new_trajectory)
        self.interface.send_trajectory(
            trajectory, command.mode, command.wbc_command_gain, command.automatic, self.MAX_SAMPLES, self.plot_trajectory
        )
        time.sleep(0.5)  # give platform model time to update status
        if command.continuous:
            ds = trajectory.distances[-1] - 0.15
            self.go_on = self.wait_until_s_reached(ds)
        else:
            self.go_on = self.wait_until_status_and_update(1)
            if command.orientation_critical and self.go_on:
                self.rotation_step(command, Ws_T_despose)

                                                        
        if command.trigger_load_objects and self.go_on:
            if True:
                time.sleep(2.0)
                self.interface.load_objects()
                time.sleep(2.0)  # wait until objects loaded
            else:
                pass

        if command.state_name != "service" and self.go_on:
            self.interface.state_reached(command)
        elif self.go_on:
            print("Arrived at service request target")

        if self.commandsFIFO.not_empty():
            if command.continuous:
                return self.handle_dubins_command(trajectory)
            return self.handle_next_command()
        return True

    def wait_until_status_and_update(self, status):
        while not self.interface.check_status(status):
            sct_state = self.interface.get_sct_state()
            for i, command in enumerate(self.commandsFIFO):
                if (
                    sct_state == command.state_name and command.parsed_frame == "update_wbc"
                ):
                    self.interface.update_wbc_gain(command.wbc_command_gain)
                    del self.commandsFIFO[i]
                    break
            if not self.interface.task_still_active():
                print("Task interuption - probably neccecary modes not activated")
                return False
        return True

    def wait_until_s_reached(self, s):
        while not self.interface.s_reached(s):
            sct_state = self.interface.get_sct_state()
            for i, command in enumerate(self.commandsFIFO):
                if (
                    sct_state == command.state_name and command.parsed_frame == ""  # this is to update wbc gain -> this is a command with emtpy parsed name
                ):
                    self.interface.update_wbc_gain(command.wbc_command_gain)
                    del self.commandsFIFO[i]
                    break
            if not self.interface.task_still_active():
                print("Task interuption - probably neccecary modes not activated")
                return False
        return True


    def reset_alignment(self, finished:bool=False, reset_task:bool=False):
        self.commandsFIFO.clear()
        self.current_command = None
        self.interface.reset_topic(self.MAX_SAMPLES)
        self.go_on = True
        
        if reset_task or not finished:
            self.interface.set_to_default_task()

        if finished:
            print("[TP] Successfully finished")
        else:
            print("[TP] Not successfully finished")

    def handle_next_command(self):
        finished = False
        if self.commandsFIFO.not_empty():
            if self.go_on:
                command = self.commandsFIFO.peek()
                if command.mode == ExecutionMode.path or command.mode == ExecutionMode.obstacle_avoidance:
                    finished = self.handle_dubins_command()
                elif command.mode == ExecutionMode.heading:
                    finished = self.heading()
                elif command.mode == ExecutionMode.rotation:
                    finished = self.rotation()
            else:
                self.commandsFIFO.clear()
                return False
        return finished


if __name__ == "__main__":
    rclpy.init(args=None)

    manager = TrajectoryManager()
    #manager.interface.set_manual_wheelchair_driving()
    print("Trajectory Planner ready")
    while True:
        manager.ln_clnt.wait_and_handle_service_group_requests(None, 0.3)
        if manager.commandsFIFO.not_empty():
            manager.interface.set_wheelchair_modes(manager.commandsFIFO.peek())  #TODO: distinguish svc / sct call 
            time.sleep(0.3) # need time here 
            while manager.commandsFIFO.not_empty():
                finished = manager.handle_next_command()
            manager.reset_alignment(finished, manager.commandsFIFO.last_used.reset_task_to_default)
        print("Waiting for requests")
