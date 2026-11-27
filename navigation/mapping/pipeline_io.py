"""Communication rules shared by all stages of the mapping pipeline (rclpy only).

Every message of one camera frame carries the LN capture time as ``header.stamp`` through the
whole pipeline; the stamp is the only key used to match messages. Each stage is triggered by a
new depth image, runs in a single thread (processing happens in the depth callback) and uses
the QoS profiles below on both the publisher and the subscriber side:

    qos_trigger()   depth images that trigger a stage: reliable, keep last 1. Nothing is lost
                    silently, and while a stage is busy DDS keeps only the newest image, so a
                    slow stage drops old frames instead of building up latency.
    qos_buffer(n)   colour images: reliable, keep last n, for the lookup by stamp.
    qos_latched()   camera_info: reliable, transient local. Sent once; late subscribers still
                    receive it.

Stages that get their depth later than the matching colour image (filter, nvblox) look the
colour image up with ColorByStamp. StageMonitor prints the same status line for every stage.

No dependencies beyond rclpy and message_filters, so the depth completion node can import it
from its own conda environment.
"""
import time

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

COLOR_TOPIC = "/camera/color/image_raw"        # bridge, rgb8
DEPTH_TOPIC = "/camera/depth/image_raw"        # bridge, 16UC1 [mm], registered to colour
INFO_TOPIC = "/camera/color/camera_info"       # bridge, latched; colour and depth share K
COMPLETED_DEPTH_TOPIC = "/camera/completed/depth/image_raw"
FILTERED_DEPTH_TOPIC = "/camera/filtered/depth/image_raw"
COLOR_BUFFER = 10    # colour images kept for the lookup (~0.7 s at 15 Hz, covers the latency)


def qos_trigger():
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)


def qos_buffer(n=COLOR_BUFFER):
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=n)


def qos_latched():
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


def stamp_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def same_stamp(a, b):
    return a.sec == b.sec and a.nanosec == b.nanosec


class ColorByStamp:
    """Last colour images of the bridge, looked up by the stamp of a (later) depth image.

    Uses message_filters.Cache as the buffer; it only stores, it never drops a depth image.
    on_arrival(msg), if given, is called for every new colour image (e.g. to process a depth
    image that arrived before its colour image).
    """

    def __init__(self, node, topic=COLOR_TOPIC, size=COLOR_BUFFER, on_arrival=None):
        from message_filters import Cache, Subscriber
        from sensor_msgs.msg import Image
        self._sub = Subscriber(node, Image, topic, qos_profile=qos_buffer(size))
        self._cache = Cache(self._sub, cache_size=size)
        if on_arrival is not None:
            self._cache.registerCallback(on_arrival)

    def get(self, stamp):
        msg = self._cache.getElemAfterTime(Time.from_msg(stamp))   # first image with stamp >= t
        return msg if msg is not None and same_stamp(msg.header.stamp, stamp) else None


class StageMonitor:
    """One status line per stage every `period` seconds:

        <name>: in 14.2 Hz | out 8.1 Hz | latency 180 ms | model 53.0 | fusion 12.1 ms

    in = trigger images received, out = frames published, latency = publish time minus the
    capture time (stamp), followed by the mean duration of the named steps. Frames lost between
    two stages show up as the difference between the 'out' of the earlier and the 'in' of the
    later stage.
    """

    def __init__(self, log, name, period=5.0):
        self.log, self.name, self.period = log, name, period
        self._reset(time.time())

    def _reset(self, now):
        self.t0, self.n_in, self.n_out, self.lat = now, 0, 0, 0.0
        self.steps, self.counts = {}, {}

    def received(self):
        self.n_in += 1

    def count(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1

    def published(self, stamp, **step_ms):
        now = time.time()
        self.n_out += 1
        self.lat += now - stamp_sec(stamp)
        for k, v in step_ms.items():
            self.steps[k] = self.steps.get(k, 0.0) + v
        dt = now - self.t0
        if dt < self.period:
            return
        parts = [f"in {self.n_in / dt:5.1f} Hz", f"out {self.n_out / dt:5.1f} Hz",
                 f"latency {1000 * self.lat / self.n_out:4.0f} ms"]
        if self.steps:
            parts.append(" | ".join(f"{k} {v / self.n_out:.1f}" for k, v in self.steps.items()) + " ms")
        parts += [f"{k} {v}" for k, v in self.counts.items()]
        self.log(f"{self.name}: " + " | ".join(parts))
        self._reset(now)
