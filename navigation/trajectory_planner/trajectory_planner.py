import numpy as np

from robotutils.base.print_tools import print_red

from navigation.trajectory_planner.path_planners.dubins_path.constrained_dubin_path import plan
from navigation.trajectory_planner.timescaling import Graph
from navigation.trajectory_planner.trajectory_commands import Command, Trajectory
from navigation.wheelchair_interface.wheelchair_utils import stopwatch


class TrajectoryPlanner:

    def __init__(self, commandsFIFO):
        self.commandsFIFO = commandsFIFO

    def _compute_trajectory_with_timescaling(
        self, path_x, path_y, path_yaw,
        v_start, v_end, v_max, a_max, a_min,
        v_max_start, v_max_end, distance_slow_start, distance_slow_end
    ) -> Trajectory:
        if not np.any(path_x):
            print_red("No path found")
            return Trajectory()

        graph = Graph(path_x, path_y, v_max, a_max, a_min)
        t = graph.get_time_scaling(v_start, v_end, v_max_start, v_max_end, distance_slow_start, distance_slow_end)
        return Trajectory(graph.x, graph.y, graph.s, t, path_yaw[-1], v_end)

    @stopwatch
    def get_dubins_trajectory(self, start_pose, target_pose, v_start, command: Command) -> Trajectory:
        """
        Computes a time-scaled dubins trajectory from start_pose to target_pose.

        The planning and timescaling parameters are read from the command:
        Command is the single source of truth for the standard values (overridable per sct command).
        v_start is the velocity of the previous trajectory (not a command parameter).
        """

        path_x, path_y, path_yaw = plan(
            Ws_T_despose=target_pose,
            Ws_T_C=start_pose,
            min_length_straight_end=command.min_length_straight_end,
            min_length_straight_start=command.min_length_straight_start,
            max_curvature=command.max_curvature,
            min_curvature=command.min_curvature,
        )
        if not np.any(path_x):
            print_red("No path found in first try. Try with rotation in place")
            path_x, path_y, path_yaw = plan(
                Ws_T_despose=target_pose,
                Ws_T_C=start_pose,
                min_length_straight_end=command.min_length_straight_end,
                min_length_straight_start=command.min_length_straight_start,
                max_curvature=100,
                min_curvature=90
            )

        return self._compute_trajectory_with_timescaling(
            path_x, path_y, path_yaw,
            v_start, command.v_end, command.v_max, command.a_max, command.a_min,
            command.v_max_start, command.v_max_end,
            command.distance_slow_start, command.distance_slow_end
        )