import numpy as np

from robotutils.geom.se3 import tf2euler

from navigation.trajectory_planner.path_planners.dubins_path.dubins_path_planner import plan_dubins_path
from navigation.wheelchair_interface.wheelchair_utils import stopwatch


@stopwatch
def plan(
    Ws_T_C: np.ndarray,
    Ws_T_despose: np.ndarray,
    step_size: float = 0.1,
    min_length_straight_end: float = 0.2,
    min_length_straight_start: float = 0.0,
    min_curvature: float = 0.1,
    max_curvature: float = 2.0,
    straight_sampling: float = 0.2
) -> tuple[np.ndarray]:
    """
    Calculates viapoints of a constrained Dubins path.
    This function uses the basic Dubins path planner from Python Robotics
    (https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/DubinsPath/dubins_path_planner.py).
    The resulting path is a combination of the Dubins curve from the basic planner and a straight segment at the end to reach the target pose.
    Optionally, a straight segment of fixed length along the start pose heading is prepended to the path.

    In detail this path planner tries to find the best compromise between curvature and length of the straight segment:
    -> Smaller curvature reduces the length of the straight segment, and vice versa.

    Therefore, several paths with different straight lengths and curvatures are calculated, and the best one (according to a heuristic) is chosen.

    Parameters
    ----------
    Ws_T_C : starting pose (odometry) as a 4x4 np array
    Ws_T_despose : target pose as a 4x4 np array
    step_size : step size for path sampling, default 0.1
    min_length_straight_end : minimal length of the straight segment at the end of the path [m], default 0.2
    min_length_straight_start : length of the straight segment at the beginning of the path, driven along
        the start pose heading [m], default 0.0 (no start straight segment)

    max_curvature : maximum possible curvature [1/m], default 2.0
    straight_sampling : sampling rate of straight distances [m], default 0.2

    Returns
    -------
    tuple containing
        - path_x : X coordinates of the determined path as 1D np array
        - path_y : Y coordinates of the determined path as 1D np array
        - path_yaw : Yaw coordinates of the determined path as 1D np array
    """

    x_start, y_start = None, None
    if min_length_straight_start > 0:
        if min_length_straight_start < step_size:
            print(
                f"min_length_straight_start ({min_length_straight_start} m) is smaller than "
                f"step_size ({step_size} m), start straight segment is ignored"
            )
        else:
            x_start, y_start = get_straight_startpoints(Ws_T_C, min_length_straight_start, step_size)

    straights = np.array([min_length_straight_end])

    despose_T_C = np.linalg.inv(Ws_T_despose) @ Ws_T_C
    t_x = despose_T_C[0, 3] * -1
    if t_x > min_length_straight_end:
        max_len_straight = max((t_x / 2), min_length_straight_end)
        straights = np.arange(
            min_length_straight_end,
            max_len_straight + straight_sampling,
            straight_sampling,
        )[::-1]

    # get the least possible curvature to get a feasable path with a straight of minimal length
    least_curv_min_straight = get_least_possible_curv(
        Ws_T_despose, Ws_T_C, straights[-1], min_curvature, max_curvature, step_size,
        x_start=x_start, y_start=y_start
    )
    # get the least possible curvature to get a feasable path with a straight of maximal length
    least_curv_max_straight = get_least_possible_curv(
        Ws_T_despose, Ws_T_C, straights[0], min_curvature, max_curvature, step_size,
        x_start=x_start, y_start=y_start
    )

    if least_curv_min_straight and least_curv_max_straight:
        curvature = (least_curv_min_straight + least_curv_max_straight) / 2
    elif least_curv_min_straight:
        curvature = (least_curv_min_straight + max_curvature) / 2
    else:
        return None, None, None

    curvature = min(curvature, max_curvature)

    return find_best_with_fixed_curvature(
        Ws_T_despose, Ws_T_C, curvature, straights, step_size, x_start=x_start, y_start=y_start
    )


def find_best_with_fixed_curvature(
    Ws_T_despose: np.ndarray,
    Ws_T_C: np.ndarray,
    curvature: float,
    straights: np.ndarray,
    step_size: float = 0.1,
    x_start: np.ndarray | None = None,
    y_start: np.ndarray | None = None,
) -> tuple[np.ndarray]:
    """
    Computes paths with varying straight lengths at the end, for a fixed curvature, and selects the best path according to a cost heuristic.
    Optionally a fixed straight segment at the beginning (given by x_start/y_start) is prepended and the
    Dubins curve is planned from its end point instead of the start pose.

    Parameters
    ----------
    Ws_T_despose : target pose as a 4x4 np array
    Ws_T_C : starting pose (odometry) as a 4x4 np array
    curvature : path curvature [1/m]
    straights : list of straight lengths (e.g., [1.5, 1.4, ..., 0.3]) as a 1D np array
    step_size : step size for path sampling, default 0.1
    x_start : X coordinates of the start straight segment as 1D array, default None (no start straight)
    y_start : Y coordinates of the start straight segment as 1D array, default None (no start straight)

    Returns
    -------
    tuple containing
        - path_x : X coordinates of the determined path as 1D np array
        - path_y : Y coordinates of the determined path as 1D np array
        - path_yaw : Yaw coordinates of the determined path as 1D np array
    """
    best_path = (None, None, None)
    best_cost = np.inf

    s_yaw = tf2euler(Ws_T_C)[5]
    t_yaw = tf2euler(Ws_T_despose)[5]

    if x_start is not None:
        # the dubins curve starts at the end of the start straight segment
        s_x, s_y = x_start[-1], y_start[-1]
        dubins_skip = 1  # first dubins point coincides with the last start straight point
    else:
        s_x, s_y = tf2euler(Ws_T_C)[0], tf2euler(Ws_T_C)[1]
        dubins_skip = 0

    for dx in straights:
        x_end, y_end = get_straight_endpoints(Ws_T_despose, length=dx, sampling=step_size)

        t_x, t_y = x_end[0], y_end[0]

        path_x, path_y, path_yaw, _, _ = plan_dubins_path(s_x, s_y, s_yaw, t_x, t_y, t_yaw, curvature, step_size)

        if x_start is not None:
            path_x = np.concatenate((x_start, path_x[dubins_skip:]))
            path_y = np.concatenate((y_start, path_y[dubins_skip:]))
            path_yaw = np.concatenate((np.full(len(x_start), s_yaw), path_yaw[dubins_skip:]))

        path_x = np.append(path_x, x_end[1:])
        path_y = np.append(path_y, y_end[1:])
        path_yaw = np.append(path_yaw, [path_yaw[-1]] * len(x_end[1:]))

        path_length = len(path_x) * step_size  # approx path length
        cost = path_length - dx * 3 / 4  # adjust weight here

        if cost < best_cost and check_suitability(path_yaw):
            best_cost = cost
            best_path = (path_x, path_y, path_yaw)

    return best_path


def get_least_possible_curv(
    Ws_T_despose: np.ndarray,
    Ws_T_C: np.ndarray,
    len_straight: float = 0.2,
    min_curvature: float = 0.1,
    max_curvature: float = 2.0,
    step_size: float = 0.1,
    curvature_sampling: float = 0.1,
    x_start: np.ndarray | None = None,
    y_start: np.ndarray | None = None,
) -> float:
    """
    finds the least possible curvature to reach the target pose with a straight of given length at the end.
    If a start straight segment is given (x_start/y_start), the dubins curve is checked from its end point.

    Parameters
    ----------
    Ws_T_despose : target pose as a 4x4 numpy array
    Ws_T_C : starting pose (odometry) as a 4x4 numpy array
    len_straight : length of the straight segment at the end [m], default 0.2
    min_curvature : minimal possible curvature [1/m], default 0.1
    max_curvature : maximum possible curvature [1/m], default 2.0
    step_size : step size for path sampling, default 0.1
    curvature_sampling : sampling interval for curvature values [1/m], default 0.1
    x_start : X coordinates of the start straight segment as 1D array, default None (no start straight)
    y_start : Y coordinates of the start straight segment as 1D array, default None (no start straight)

    Returns
    -------
    curvature : least possible curvature [1/m] if a feasible path is found, otherwise None
    """
    curvatures = np.arange(min_curvature, max_curvature, curvature_sampling)

    s_yaw = tf2euler(Ws_T_C)[5]
    t_yaw = tf2euler(Ws_T_despose)[5]
    if x_start is not None:
        # the dubins curve starts at the end of the start straight segment
        s_x, s_y = x_start[-1], y_start[-1]
    else:
        s_x, s_y = tf2euler(Ws_T_C)[0], tf2euler(Ws_T_C)[1]

    x_end, y_end = get_straight_endpoints(Ws_T_despose, length=len_straight, sampling=step_size)
    t_x, t_y = x_end[0], y_end[0]

    for curvature in curvatures:
        _, _, path_yaw, _, _ = plan_dubins_path(s_x, s_y, s_yaw, t_x, t_y, t_yaw, curvature, step_size)
        if check_suitability(path_yaw):
            return curvature
    return None


def get_straight_endpoints(Ws_T_despose: np.ndarray, length: float, sampling: float = 0.1) -> tuple[np.ndarray]:
    """
    Calculates viapoints of a straight path segment reaching the target pose.
    The straight path is sampled at intervals defined by the sampling parameter.

    Note: length >= sampling

    Parameters
    ----------
    Ws_T_despose : target pose as a 4x4 numpy array
    length : length of the straight path segment [m]
    sampling : the distance between viapoints [m], default 0.1

    Returns
    -------
    tuple containing
        - x_end : X coordinates of the straight as 1D np array
        - y_end : Y coordinates of the straight as 1D np array

    Raises
    ------
    AssertionError
        if length divided by sampling is less than 1.
    """

    assert (
        length / sampling >= 1
    ), "Minimum amount of sample points must be >1. Increase straight length or decrease sampling distance"

    x_end = []
    y_end = []
    despose_T_endcurv = np.identity(4)
    for i in range(int(length / sampling)):
        despose_T_endcurv[0, 3] = i * -sampling
        Ws_T_endcurv = Ws_T_despose @ despose_T_endcurv
        x_end.append(Ws_T_endcurv[0, 3])
        y_end.append(Ws_T_endcurv[1, 3])
    despose_T_endcurv[0, 3] = -length
    Ws_T_endcurv = Ws_T_despose @ despose_T_endcurv
    x_end.append(Ws_T_endcurv[0, 3])
    y_end.append(Ws_T_endcurv[1, 3])

    return x_end[::-1], y_end[::-1]


def get_straight_startpoints(Ws_T_C: np.ndarray, length: float, sampling: float = 0.1) -> tuple[np.ndarray]:
    """
    Calculates viapoints of a straight path segment starting at the start pose, driven along its heading.
    The straight path is sampled at intervals defined by the sampling parameter.
    This is the counterpart of get_straight_endpoints at the beginning of the path.

    Note: length >= sampling

    Parameters
    ----------
    Ws_T_C : starting pose as a 4x4 numpy array
    length : length of the straight path segment [m]
    sampling : the distance between viapoints [m], default 0.1

    Returns
    -------
    tuple containing
        - x_start : X coordinates of the straight as 1D np array
        - y_start : Y coordinates of the straight as 1D np array

    Raises
    ------
    AssertionError
        if length divided by sampling is less than 1.
    """

    assert (
        length / sampling >= 1
    ), "Minimum amount of sample points must be >1. Increase straight length or decrease sampling distance"

    x_start = []
    y_start = []
    C_T_fwd = np.identity(4)
    for i in range(int(length / sampling)):
        C_T_fwd[0, 3] = i * sampling
        Ws_T_fwd = Ws_T_C @ C_T_fwd
        x_start.append(Ws_T_fwd[0, 3])
        y_start.append(Ws_T_fwd[1, 3])
    C_T_fwd[0, 3] = length
    Ws_T_fwd = Ws_T_C @ C_T_fwd
    x_start.append(Ws_T_fwd[0, 3])
    y_start.append(Ws_T_fwd[1, 3])

    return x_start, y_start


def check_suitability(path_yaw: np.ndarray) -> bool:
    """
    Checks the suitability of a path by analyzing the yaw values of viapoints.
    The sum of yaw differences must be less than 3/2 * pi (270 degrees) for the path to be considered.

    Parameters
    ----------
    path_yaw : Yaw angles of the viapoints in radians as 1D np array

    Returns
    -------
    bool : True if the path is suitable, False otherwise.
    """

    threshold = 3 * np.pi / 2

    diffs = np.diff(path_yaw)
    for i, diff in enumerate(diffs):
        if diff > np.pi:
            diffs[i] += -2 * np.pi
        elif diff < -np.pi:
            diffs[i] += 2 * np.pi

    return not (abs(sum(diffs)) > threshold or sum(abs(diffs)) > threshold)
