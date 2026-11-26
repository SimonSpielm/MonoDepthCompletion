
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d

from navigation.wheelchair_interface.wheelchair_utils import stopwatch

class Graph:
    """
    A class to create an s-v (distance-velocity) graph for a given path defined by x and y coordinates.
    The goal is to calculate the velocity v for each discretized distance point s along the path while respecting velocity and
        acceleration constraints based on the path curvature.

    - Calculates the distances s along the path for every given viapoint (x,y)
    - Further discretizes the path by cubic spline interpolation to a given resolution/distance between viapoints
    - Velocity and acceleration limits are calculated based on the path curvature at every distance point s
    - Maximizes area under the s-v curve while respecting the velocity and acceleration constraints(t = area under s-v curve)
    - Utilizes a forward-backward algorithm to compute the time scaling

    By initializing a Graph instance the discrete path is interpolated and then further discretized along s
    and the velocity and acceleration limits are determined.

    You can specify the resolution, maximum velocity and acceleration limits.

    Parameters:
        path_x : np.ndarray
            X coordinates of the path viapoins as 1D np array
        path_y : np.ndarray
            Y coordinates of the path viapoins as 1D np array
        v_max : float
            maximum velocity on a straight path [m/s], default 0.5 m/s
        a_max : float
            maximum acceleration on a straight path [m/s²], default 0.25 m/s²
        a_min : float
            minimum (braking) acceleration on a straight path [m/s²], default -0.03 m/s²

    Attributes:
        resolution : float
            resolution for distance discretization [m], default 0.02 m
        s : np.ndarray
            discretized distance points along the path as 1D np array(e.g [0, 0.1, 0.2, ...])
        x : np.ndarray
            interpolated x coordinates at each point s as 1D np array
        y : np.ndarray
            interpolated y coordinates at each point s as 1D np array
        curvature : np.ndarray
            curvature values at each point s as 1D np array
        v_max : np.ndarray
            maximum velocity at each s based on curvature as 1D np array
        a_max : np.ndarray
            maximum acceleration at each s based on curvature as 1D np array
        a_min : np.ndarray
            minimum acceleration at each s based on curvature as 1D np array
    """

    def __init__(
        self,
        path_x:np.ndarray,
        path_y:np.ndarray,
        v_max:float,
        a_max:float,
        a_min:float,
    ):
        self.resolution = 0.02

        distances = self.get_distances(path_x, path_y)
        self.s = np.arange(0, distances[-1], self.resolution)
        if abs(self.s[-1] - self.s[-1]) >= self.resolution/2:
            self.s = np.append(self.s, distances[-1])
        else:
            self.s[-1] = distances[-1]
        self.ds = np.diff(self.s)
        self.x = CubicSpline(distances, path_x)(self.s)
        self.y = CubicSpline(distances, path_y)(self.s)

        self.curvature = self.get_curvature()
        self.a_max = self.get_amax_for_curv(a_max, self.curvature)
        self.a_min = self.get_amin_for_curv(a_min, self.curvature)
        self.v_max = self.get_vmax_for_curv(v_max, self.curvature)

    @stopwatch
    def get_time_scaling(
        self,
        v_start:float,
        v_end:float,
        v_max_start:float,
        v_max_end:float,
        distance_slow_start:float,
        distance_slow_end:float,
    ) -> np.ndarray:

        """
        This method implements the pipeline to compute a velocity profile along the discretized path that adheres to specified
            velocity and acceleration constraints, including start and end velocities.
        A timestamp for every distance point s will be returned.
        Optionally, maximum velocities at the start and end over specified distances can be set to model slow start or slow end

        Parameters:
            v_start : float, optional
                initial velocity at the start of the path [m/s], default 0 m/s
            v_end : float, optional
                final velocity at the end of the path [m/s], default 0 m/s
            v_max_start : float, optional
                maximum velocity over a initial segment of the path [m/s], default 0.3 m/s
            v_max_end : float, optional
                maximum velocity over the final segment of the path [m/s], default 0.3 m/s
            d_slow_start : float, optional
                distance over which to apply `v_max_start` at the beginning of the path [m], default 0 m
            d_slow_end : float, optional
                Distance over which to apply `v_max_end` at the end of the path [m], default 0 m

        Returns:
            t : np.ndarray
                timestamps for each s as 1D np array
        """

        v_start = np.clip(v_start, 0, max(self.v_max))
        v_end = np.clip(v_end, 0, max(self.v_max))
        v_max_end = max(v_end, v_max_end)
        v_max_start = max(v_start, v_max_start)

        for i in range(round(distance_slow_start / self.resolution)):
            self.v_max[i] = v_max_start

        for i in range(round(distance_slow_end / self.resolution)):
            self.v_max[-i] = v_max_end

        # find velocity profile
        v_forward = self.forward(v_start)
        v_backward = self.backward(v_end, v_forward)
        v = np.minimum(v_forward, v_backward)
        v = self.v_moving_average_filter(v)

        t = self.get_timestamps(v)

        return t

    def get_timestamps(self, v:np.ndarray) -> np.ndarray:
        """
        calculates the timestamps for each distance point s on the velocity profile v

        - The function computes the time t required to travel between each pair of distance points along the discretized path.
        - It uses the trapezoidal rule for numerical integration to approximate the time increments dt.
        - Special handling is included to avoid division by zero when the velocities are zero.

        Parameters:
            v : np.ndarray
                velocity at each point s along the path as 1D np array

        Returns:
            t : np.ndarray
                cumulative timestamps for each point s along the discretized path
        """
        t = [0]
        for i in range(len(self.ds)):
            if v[i] > 0 and v[i+1] > 0:
                dt = (1 / v[i] + 1 / v[i + 1]) / 2 * self.ds[i]
            else:
                dt = 2 * self.ds[i] / (v[i] + v[i + 1])
            t.append(t[-1] + dt)
        return t

    def forward(self, v0:float) -> np.ndarray:
        """
        Performs the forward pass of the time scaling algorithm

        - Starting from initial velocity v0 computes the maximum velocity at each s without exceeding maximum acceleration
        - based on the kinematic equation relating velocities, acceleration, and distance

        Parameters:
            v0 : float
                initial velocity at the start of the path [m/s]

        Returns:
            v_forward : np.ndarray
                maximum velocities at each s from the forward pass as 1D np array
        """
        v_forward = np.zeros(len(self.s))
        v_forward[0] = v0

        for i in range(len(self.ds)):
            v_max_possible = np.sqrt(v_forward[i]**2 + 2 * self.a_max[i] * self.ds[i])
            v_forward[i + 1] = min(self.v_max[i + 1], v_max_possible)
        return v_forward

    def backward(self, v_end:float, v_forward:np.ndarray) -> np.ndarray:
        """
        Performs the backward pass of the time scaling algorithm

        - Starting from final velocity v_end computes the maximum velocity at each s without exceeding deceleration limits, 
            considering the forward pass velocities
        - based on the kinematic equation relating velocities, acceleration, and distance

        Parameters:
            v_end : float
                final velocity at the end of the path [m/s]
            v_forward : np.ndarray
                maximum velocities at each s from the forward pass as 1D np array

        Returns:
            v_backward : np.ndarray
                maximum velocities at each s from the backward pass as 1D np array
        """
        v_backward = np.zeros(len(self.s))
        v_backward[-1] = v_end

        for i in range(len(self.ds), 0, -1):
            v_max_possible = np.sqrt(v_backward[i]**2 + 2 * -self.a_min[i - 1] * self.ds[i - 1])
            v_backward[i - 1] = min(v_forward[i - 1], v_max_possible)
        return v_backward

    def v_moving_average_filter(self, v:np.ndarray, samples:int=20) -> np.ndarray:
        """
        applies a moving average filter to the velocity profile

        Parameters:
            v : np.ndarray
                velocity values at each s as 1D np array
            samples : int
                filter kernel size (number of samples over which to average), default 20

        Returns:
            v : np.ndarray
                filtered velocity values as 1D np array
        """
        if len(v) < samples:
            v_filtered = uniform_filter1d(input=v, size=samples, mode="constant", cval=0)
            b = int(samples / 2)
            v[b:-b] = v_filtered[b:-b]
        return v

    def get_distances(self, path_x:np.ndarray, path_y:np.ndarray) -> np.ndarray:
        """
        computes the accumulated distances between the given discrete waypoints (x,y)

        Parameters:
            path_x : np.ndarray
                X coordinates of the discrete path waypoints as 1D np array
            path_y : np.ndarray
                Y coordinates of the discrete path waypoints as 1D np array

        Returns:
            s : np.ndarray
                accumulated distances along the path as 1D np array
        """
        ds = np.hypot(np.diff(path_x), np.diff(path_y))
        s = np.concatenate(([0], np.cumsum(ds)))
        return s

    def get_curvature(self) -> np.ndarray:
        """
        Computes the curvature at each distance point s

        the curvature at each s is calculated using differential geometry formulas for parametric curves given by:
            c = (dx/ds * d²y/ds² - dy/ds * d²x/ds²) / ( (dx/ds)² + (dy/ds)² )^(3/2)

        Returns:
            curvature : np.ndarray
                curvature values at each s as 1D np array.
        """

        dx_ds = np.gradient(self.x, self.s)
        dy_ds = np.gradient(self.y, self.s)
        d2x_ds2 = np.gradient(dx_ds, self.s)
        d2y_ds2 = np.gradient(dy_ds, self.s)

        num = dx_ds * d2y_ds2 - dy_ds * d2x_ds2
        denom = (dx_ds**2 + dy_ds**2)**1.5
        curv = num / denom

        return curv

    def get_vmax_for_curv(self, v_max:float, c:Union[float, np.ndarray] ) -> Union[float, np.ndarray]:
        """
        computes the maximum velocity based on curvature and the maximum velocity on a straight
        the maximum velocity decreases with increasing curvature similar to a Gaussian with a minimum of 0.05

        Parameters:
            v_max : float
                maximum velocity on a straight path [m/s]
            c : (float, np.ndarray)
                curvature value as float or values as 1D np array [1/m]

        Returns:
            v_max_curv : (float, np.ndarray)
                maximum allowable velocity for the given curvature
        """
        ret = v_max * np.exp(-0.4 * c**2)
        return np.maximum(ret, 0.05)

    def get_amax_for_curv(self, a_max:float, c:Union[float, np.ndarray] ) -> Union[float, np.ndarray]:
        """
        computes the maximum acceleration based on curvature and the maximum acceleration on a straight path
        the maximum acceleration decreases with increasing curvature similar to a Gaussian

        Parameters:
            a_max : float
                maximum acceleration on a straight path [m/s²]
            c : (float, np.ndarray)
                curvature value as float or values as 1D np array [1/m]

        Returns:
            a_max_curv : (float, np.ndarray)
                maximum allowable acceleration for the given curvature
        """
        ret =  a_max * np.exp(-0.7 * c**2)
        return np.maximum(ret, 0.02)

    def get_amin_for_curv(self, a_min:float, c:Union[float, np.ndarray] ) -> Union[float, np.ndarray]:
        """
        computes the minimum (braking) acceleration based on curvature and the minimum acceleration on a straight path
        the min. acceleration increases with increasing curvature similar to a inverted Gaussian with a maximum of -0.01 m/s²

        Parameters:
            a_min : float
                minimum (braking) acceleration on a straight path [m/s²]
            c : (float, np.ndarray)
                Curvature value as float or values as 1D numpy array [1/m]

        Returns:
            a_min_curv : (float, np.ndarray)
                Minimum allowable acceleration for the given curvature
        """
        ret = a_min * np.exp(-0.7 * c**2)
        return np.minimum(ret, -0.02)

    def plot(self, v:np.ndarray, t:np.ndarray) -> None:
        """
        plots the velocity, position, and acceleration profiles along the path

        Parameters:
            v : np.ndarray
                velocity profile along the path as 1D np array
            t : np.ndarray
                timestamps for each s as 1D np array
        """

        plt.close()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        ax1.plot(self.s, v, 'r', label='s-v')
        ax1.plot(self.s, self.v_max, 'b--', label='v_max')
        ax1.set_xlabel('s (m)')
        ax1.set_ylabel('v (m/s)')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(t, self.s, 'g', label='t-s')
        ax2.set_xlabel('t (s)')
        ax2.set_ylabel('s (m)')
        ax2.legend()
        ax2.grid(True)

        a = np.diff(v) / np.diff(t)
        a = np.insert(a, 0, 0)
        ax3.plot(self.s, a, 'r', label='s-a')
        ax3.plot(self.s, self.a_max, 'b--', label='a_max')
        ax3.plot(self.s, self.a_min, 'g--', label='a_min')
        ax3.set_xlabel('s (m)')
        ax3.set_ylabel('a (m/s²)')
        ax3.legend()
        ax3.grid(True)
        print("plotted")
        plt.savefig("plot.png")
