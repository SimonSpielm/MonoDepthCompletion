from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from robotutils.geom.se3 import euler2tf


@dataclass
class Trajectory:
    x_path: np.ndarray = field(default_factory=lambda: np.array([]))
    y_path: np.ndarray = field(default_factory=lambda: np.array([]))
    distances: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    yaw_end: float | None = None
    v_end: float = 0.0

    def append(self, trajectory: Trajectory) -> None:
        index = 0
        if len(self.distances):
            index = 1   # dont know why not smaller, this works TODO
            trajectory.distances += (self.distances[-1] - trajectory.distances[0] + 0.001)
            trajectory.timestamps += (self.timestamps[-1] - trajectory.timestamps[0] + 0.001)

        self.x_path = np.append(self.x_path, trajectory.x_path[index:])
        self.y_path = np.append(self.y_path, trajectory.y_path[index:])
        self.distances = np.append(self.distances, trajectory.distances[index:])
        self.timestamps = np.append(self.timestamps, trajectory.timestamps[index:])
        if trajectory.v_end is not None:
            self.v_end = trajectory.v_end
        if trajectory.yaw_end is not None:
            self.yaw_end = trajectory.yaw_end

    def length(self) -> int:
        return len(self.x_path)

    def get_final_pose(self) -> np.ndarray:
        x = self.x_path[-1]
        y = self.y_path[-1]
        return euler2tf([x,y,0,0,0,self.yaw_end])

    def valid(self) -> bool:
        return bool(len(self.x_path) == len(self.y_path) == len(self.distances) == len(self.timestamps) > 0 and self.yaw_end is not None and np.any(np.diff(self.timestamps) > 0) and np.any(np.diff(self.distances) > 0))

    def plot(self) -> None:
        """
        plots the velocity, position, and acceleration profiles along the path

        Parameters:
            v : np.ndarray
                velocity profile along the path as 1D np array
            t : np.ndarray
                timestamps for each s as 1D np array
        """
        import matplotlib.pyplot as plt
        plt.close()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        v = np.diff(self.distances) / np.diff(self.timestamps)
        v = np.insert(v, 0, 0)

        ax1.plot(self.distances, v, 'r', label='s-v')
        ax1.set_xlabel('s (m)')
        ax1.set_ylabel('v (m/s)')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.timestamps, self.distances, 'g', label='t-s')
        ax2.set_xlabel('t (s)')
        ax2.set_ylabel('s (m)')
        ax2.legend()
        ax2.grid(True)

        a = np.diff(v) / np.diff(self.timestamps)
        a = np.insert(a, 0, 0)
        ax3.plot(self.timestamps, a, 'r', label='s-a')
        ax3.set_xlabel('s (m)')
        ax3.set_ylabel('a (m/s²)')
        ax3.legend()
        ax3.grid(True)
        plt.savefig("plot_trajectory.png")

class ExecutionMode(Enum):
    """
    class mapping the mode string (coming from service call) to a Execution Mode to mode int (which is required in platform model)

    this mapping is used within the platform model aswell!
    """

    nothing = 0
    rotation = 1
    heading = 2
    path = 3
    obstacle_avoidance = 4

    @staticmethod
    def get_execution_mode(input: str):
        try:
            var = ExecutionMode[input]
        except KeyError:
            print(f"Key error: {input} not a correct Mode. Possible Modes: {ExecutionMode.modes()[1:]}")
            var = ExecutionMode.nothing
        return var

    @staticmethod
    def modes() -> list:
        return [mode.name for mode in ExecutionMode]

@dataclass
class Command:
    state_name: str = None
    original_frame_name: str = None
    continuous: bool = False
    orientation_critical: bool = False
    target_id: str = None
    rotation: float = None
    heading: float = None
    mode: ExecutionMode = None
    parsed_frame: str | np.ndarray = None  # (4,4) = absolute pose, (3,) = [x, y, theta] relative to the current pose
    wbc_command_gain: float = 1.0
    v_end: float = 0
    reset_task_to_default: bool = False
    set_task: bool = False #Sct commands do not set task since tsm handles this, raw svc calls do
    trigger_load_objects: bool = False
    automatic: bool = False
    v_max: float = 0.3
    a_max: float = 0.05
    a_min: float = -0.05
    min_length_straight_end: float = 0.2
    min_length_straight_start: float = 0.0
    max_curvature: float = 2.5
    min_curvature: float = 0.1
    distance_slow_start: float = 0.2
    distance_slow_end: float = 0.1
    v_max_start: float = 0.05
    v_max_end: float = 0.05


class CommandFIFO(list):
    def __init__(self):
        super().__init__()
        self.last_used: Command = Command()

    def clear(self) -> None:
        if len(self):
            self.last_used = self[-1]
        super().clear()

    def put(self, item: Command) -> None:
        self.append(item)

    def pop(self) -> Command:
        if self:
            self.last_used = self[0]
            return super().pop(0)

    def peek(self) -> Command:
        if self:
            return self[0]

    def is_empty(self) -> bool:
        return not len(self) > 0

    def not_empty(self) -> bool:
        return not self.is_empty()