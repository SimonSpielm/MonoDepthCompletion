"""Small ROS helpers shared by the mapping nodes.

The capture timestamp (seconds, time.time() domain) from the LN camera topic is
carried through the whole pipeline as ``header.stamp``. These helpers convert it
losslessly enough (< 1 us) in both directions, so the SynchronizationBuffer can be
queried with the original timestamp further down the pipeline.
"""
import math

from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped

from robotutils.geom.so3 import rotmat2quat


def to_ros_time(t: float) -> Time:
    """Seconds (float) -> builtin_interfaces/Time."""
    sec = math.floor(t)
    nanosec = int(round((t - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec, nanosec = sec + 1, nanosec - 1_000_000_000
    return Time(sec=int(sec), nanosec=nanosec)


def to_sec(stamp: Time) -> float:
    """builtin_interfaces/Time -> seconds (float)."""
    return stamp.sec + stamp.nanosec * 1e-9


def tf_from_matrix(T, parent_frame: str, child_frame: str, stamp: Time) -> TransformStamped:
    """4x4 homogeneous matrix -> TransformStamped."""
    tf_msg = TransformStamped()
    tf_msg.header.stamp = stamp
    tf_msg.header.frame_id = parent_frame
    tf_msg.child_frame_id = child_frame
    tf_msg.transform.translation.x = float(T[0, 3])
    tf_msg.transform.translation.y = float(T[1, 3])
    tf_msg.transform.translation.z = float(T[2, 3])

    q = rotmat2quat(T[:3, :3])
    tf_msg.transform.rotation.x = float(q[0])
    tf_msg.transform.rotation.y = float(q[1])
    tf_msg.transform.rotation.z = float(q[2])
    tf_msg.transform.rotation.w = float(q[3])
    return tf_msg
