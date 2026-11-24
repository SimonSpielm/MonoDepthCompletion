"""
Synchronization buffer as an LN service provider.

Buffers the last SIZE (default 3000, approx. 3 s at 1 kHz) synchronized
(Ws_T_Wt frame, arm configuration q, timestamp) samples from the platform model
and serves them via LN services:

    SynchronizationBuffer.get_q_for_t     (navigation/get_q_for_t)
    SynchronizationBuffer.get_odom_for_t  (navigation/get_odom_for_t)

Both services take a `double timestamp` in seconds (time.time() domain) and return
the sample closest to it (plus `error_message` if the buffer is still empty).

Two threads guarantee that a service call never starves the buffering:
  * fill thread (daemon): reads the LN inports and stores samples in the ring buffer
  * main thread: waits for and handles service requests

Ring buffer lookup:
  The ring buffer (SIZE slots, no extra memory) consists of at most two segments
  that are each sorted by time: the older one [start : SIZE] and the newer one
  [0 : start] (only once the buffer has wrapped). One comparison against the
  timestamp in slot 0 decides which segment to search; np.searchsorted on that
  segment (a view, no copy) then finds the sample in O(log n).

Usage (provider, run the file):
    python3 -m navigation.mapping.synchronization_buffer

Usage (client):
    from navigation.mapping.synchronization_buffer import SynchronizationBuffer_LnInterface

    iface = SynchronizationBuffer_LnInterface(ln_clnt)   # or without ln_clnt for its own client
    q = iface.get_closest_q_for_t(time.time())           # -> np.ndarray (8,)
    odom = iface.get_closest_odom_for_t(time.time())     # -> np.ndarray (12,)
"""
import threading
import time
import traceback

import links_and_nodes as ln
import numpy as np

SIZE = 3000  # 3 s at 1 kHz


class SynchronizationBuffer:
    def __init__(self):
        self.ln_clnt = ln.client("SynchronizationBuffer", [])
        self.ln_inport_Ws_T_Wt = self.ln_clnt.subscribe("Ws_to_Wt_no_reset", "ln/frame34")
        self.ln_inport_robot_kin = self.ln_clnt.subscribe("smart_assist_telemetry", "simulink/robot_model/smart_assist_telemetry")

        self.odom_buffer = np.zeros((SIZE, 12), dtype=np.float32)
        self.arm_config_buffer = np.zeros((SIZE, 8), dtype=np.float32)
        self.timestamp_buffer = np.zeros(SIZE, dtype=np.float64)
        self._start_idx = 0  # slot of the oldest sample
        self._n_valid = 0
        self._lock = threading.Lock()
        self._running = False

        svc_q = self.ln_clnt.get_service_provider("SynchronizationBuffer.get_q_for_t", "navigation/get_q_for_t")
        svc_q.set_handler(self.svc_get_q_for_t)
        svc_q.do_register()

        svc_odom = self.ln_clnt.get_service_provider("SynchronizationBuffer.get_odom_for_t", "navigation/get_odom_for_t")
        svc_odom.set_handler(self.svc_get_odom_for_t)
        svc_odom.do_register()

    def svc_get_q_for_t(self, request, req, resp):
        try:
            sample = self._find_closest(float(req.timestamp))
            if sample is None:
                resp.error_message = "no buffered samples yet"
            else:
                q, _, _ = sample
                resp.error_message = ""
                resp.q = q
        except Exception:
            resp.error_message = traceback.format_exc()
        finally:
            request.respond()

    def svc_get_odom_for_t(self, request, req, resp):
        try:
            sample = self._find_closest(float(req.timestamp))
            if sample is None:
                resp.error_message = "no buffered samples yet"
            else:
                _, odom, _ = sample
                resp.error_message = ""
                resp.odom = odom
        except Exception:
            resp.error_message = traceback.format_exc()
        finally:
            request.respond()

    def _find_closest(self, timestamp: float):
        with self._lock:
            n = self._n_valid
            if n == 0:
                return None
            start = self._start_idx
            ts = self.timestamp_buffer

            # Older segment: slots [start, start + n_old), newer segment: slots [0, n_new)
            n_old = min(n, SIZE - start)
            n_new = n - n_old

            # k = logical index (0 = oldest) of the first sample with ts >= timestamp
            if n_new > 0 and timestamp >= ts[0]:
                k = n_old + int(np.searchsorted(ts[:n_new], timestamp))
            else:
                k = int(np.searchsorted(ts[start:start + n_old], timestamp))

            # Pick the closer of the two neighbours k - 1 and k
            if k == 0:
                i = 0
            elif k == n:
                i = n - 1
            else:
                t_right = ts[(start + k) % SIZE]
                t_left = ts[(start + k - 1) % SIZE]
                i = k if (t_right - timestamp) < (timestamp - t_left) else k - 1

            idx = (start + i) % SIZE
            closest_ts = float(ts[idx])
            q = self.arm_config_buffer[idx].copy()
            odom = self.odom_buffer[idx].copy()
        return q, odom, closest_ts

    def _fill_loop(self):
        while self._running:
            self.ln_inport_Ws_T_Wt.read()
            self.ln_inport_robot_kin.read()
            self._store_sample()

    def _store_sample(self):
        t = time.time()
        frame = self.ln_inport_Ws_T_Wt.packet.frame
        q = self.ln_inport_robot_kin.packet.q_msr
        with self._lock:
            idx = (self._start_idx + self._n_valid) % SIZE
            self.odom_buffer[idx] = frame
            self.arm_config_buffer[idx] = q
            self.timestamp_buffer[idx] = t
            if self._n_valid < SIZE:
                self._n_valid += 1
            else:
                self._start_idx = (self._start_idx + 1) % SIZE

    def run(self):
        self._running = True
        fill_thread = threading.Thread(target=self._fill_loop, name="SynchronizationBufferFill", daemon=True)
        fill_thread.start()
        print("SynchronizationBuffer: running (fill thread started, waiting for service requests)")
        try:
            while self._running:
                self.ln_clnt.wait_and_handle_service_group_requests(None, 0.1)
        finally:
            self._running = False
            fill_thread.join(timeout=1.0)


class SynchronizationBuffer_LnInterface:
    def __init__(self, ln_clnt=None, name="SynchronizationBuffer"):
        self.ln_clnt = ln_clnt if ln_clnt is not None else ln.client(name, [])
        self._svc_q = self.ln_clnt.get_service(f"{name}.get_q_for_t", "navigation/get_q_for_t")
        self._svc_odom = self.ln_clnt.get_service(f"{name}.get_odom_for_t", "navigation/get_odom_for_t")

    def get_closest_q_for_t(self, timestamp: float) -> np.ndarray:
        self._call(self._svc_q, timestamp)
        return np.array(self._svc_q.resp.q, dtype=np.float32)

    def get_closest_odom_for_t(self, timestamp: float) -> np.ndarray:
        self._call(self._svc_odom, timestamp)
        return np.array(self._svc_odom.resp.odom, dtype=np.float32)

    def _call(self, svc, timestamp: float):
        svc.req.timestamp = float(timestamp)
        svc.call()
        if svc.resp.error_message:
            raise RuntimeError(f"service '{svc.name}' failed: {svc.resp.error_message}")


if __name__ == "__main__":
    buffer = SynchronizationBuffer()
    buffer.run()