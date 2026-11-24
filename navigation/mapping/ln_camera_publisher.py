"""
Step 1: LN camera publisher (only for a locally connected Azure Kinect).

On the real system the camera driver already publishes these LN topics, so this
node is NOT started there. Locally it reads the Azure Kinect via k4a and publishes
the same topics with the same message definitions, so everything downstream
(ln_ros_camera_bridge.py, ...) works identically in both setups:

    camera.perception/rgb    (perception/rgb)    RGB, uint8 HxWx3, + timestamp
    camera.perception/depth  (perception/depth)  depth registered to color, uint16 mm, + timestamp
    camera.perception/cam_k  (perception/cam_k)  color intrinsics K (3x3, row-major)

The timestamp is the capture time in the time.time() domain, the same clock the
SynchronizationBuffer uses for odometry and joint angles.

No timer: get_capture() blocks until the next frame, so the loop runs at camera rate.

Usage:
    python3 -m navigation.mapping.ln_camera_publisher
"""
import copy
import time

import links_and_nodes as ln
import numpy as np


class LnCameraPublisher:
    def __init__(self):
        self.setup_device()

        self.ln_clnt = ln.client("LnCameraPublisher", [])
        self.ln_outport_rgb = self.ln_clnt.publish("camera.perception/rgb", "perception/rgb")
        self.ln_outport_depth = self.ln_clnt.publish("camera.perception/depth", "perception/depth")
        self.ln_outport_camk = self.ln_clnt.publish("camera.perception/cam_k", "perception/cam_k")

        intr = self.calibration.color_cam_cal.intrinsics.parameters.param
        self.K = np.array([[intr.fx, 0.0, intr.cx],
                           [0.0, intr.fy, intr.cy],
                           [0.0, 0.0, 1.0]], dtype=float)

    def setup_device(self):
        import k4a
        print("Opening Azure Kinect device...")
        self.device = k4a.Device.open(0)

        device_config = k4a.DeviceConfiguration(
            color_format=k4a.EImageFormat.COLOR_BGRA32,
            color_resolution=k4a.EColorResolution.RES_720P,
            depth_mode=k4a.EDepthMode.WFOV_2X2BINNED,
            camera_fps=k4a.EFramesPerSecond.FPS_15,
            synchronized_images_only=True
        )

        if self.device.start_cameras(device_config) != k4a.EStatus.SUCCEEDED:
            raise IOError("Failed to start Azure Kinect cameras.")

        self.calibration = self.device.get_calibration(
            depth_mode=device_config.depth_mode,
            color_resolution=device_config.color_resolution
        )
        self.transform = k4a.Transformation(self.calibration)
        print("Kinect device setup complete.")

    def get_rgbd_k4a(self):
        capture = self.device.get_capture(-1)  # blocking
        if not capture:
            return None, None, None
        ts = time.time()
        bgra_image = copy.deepcopy(capture.color.data)
        rgb = np.ascontiguousarray(bgra_image[..., :3][..., ::-1])
        depth = copy.deepcopy(self.transform.depth_image_to_color_camera(capture.depth).data)
        return rgb, depth, ts

    def publish(self, rgb, depth, ts):
        p = self.ln_outport_rgb.packet
        p.height, p.width = rgb.shape[:2]
        p.data[:] = rgb.reshape(-1)
        p.timestamp = ts
        self.ln_outport_rgb.write()

        p = self.ln_outport_depth.packet
        p.height, p.width = depth.shape[:2]
        p.data[:] = depth.reshape(-1)
        p.timestamp = ts
        self.ln_outport_depth.write()

        self.ln_outport_camk.packet.data[:] = self.K.ravel()
        self.ln_outport_camk.write()

    def run(self):
        print("Publishing Azure Kinect on LN...")
        last = time.time()
        while True:
            rgb, depth, ts = self.get_rgbd_k4a()
            if depth is None:
                print("Failed to capture RGB or depth.")
                continue
            self.publish(rgb, depth, ts)
            now = time.time()
            print(f"\rPublished with {1.0 / max(now - last, 1e-6):5.1f} Hz", end="", flush=True)
            last = now

    def shutdown(self):
        print("\nShutting down Kinect device...")
        self.device.stop_cameras()
        self.device.close()


def main():
    publisher = LnCameraPublisher()
    try:
        publisher.run()
    except KeyboardInterrupt:
        pass
    finally:
        publisher.shutdown()


if __name__ == "__main__":
    main()
