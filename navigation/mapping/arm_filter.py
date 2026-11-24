import numpy as np
import cv2

import links_and_nodes as ln

from navigation.mapping.synchronization_buffer import SynchronizationBuffer_LnInterface
from python_modules.environment_awareness.arm_collision_model import Arm_Collision_Model

class SelfFilter:
    def __init__(self, c, pointcloud_downsampling=1):
        self.ln_clnt = ln.client("SelfFilter", [])
        self.ln_inport_W_T_Cam = self.ln_clnt.subscribe("user_config_handler.user_info", "python_modules/periphery/user_info")
        self.odom_buffer = SynchronizationBuffer_LnInterface(self.ln_clnt)
        self.ln_inport_W_T_Cam.read()
        self.Wt_T_Cam = self.ln_inport_W_T_Cam.packet.W_T_CAM.reshape((4, 4))

        self.robot = Arm_Collision_Model(use_octomap=False)

        self.c = c
        self.k = self.c_to_K(c)
        self.height = c["height"]
        self.width = c["width"]

        self.pointcloud_downsampling = pointcloud_downsampling

    def c_to_K(self, c):
        fx, fy, cx, cy = map(float, (c["fx"], c["fy"], c["cx"], c["cy"]))
        K = np.array([[fx, 0,  cx],
                    [0,  fy, cy],
                    [0,  0,  1]], dtype=float)
        return K

    def get_joint_angles_at_ts(self, ts):
        return self.odom_buffer.get_closest_q_for_t(ts)

    def project_3d_to_2d(self, point_3d):
        if point_3d[2] == 0:
            return None

        image_point = self.k.dot(point_3d)
        image_point /= image_point[2]

        # Transform to shape (2), round and turn into integer
        image_point = image_point[:2].round().astype(int)  # (x, y)

        return image_point

    def get_bounding_box_vertices(self, q=None):
        W_points = self.robot.get_collision_aabbs_for_q(q=q)
        boxes_cam = []
        for name, points in W_points.items():
            ones = np.ones((points.shape[0], 1), dtype=points.dtype)
            points = np.hstack((points, ones))  # (N,4)
            cam_h = (np.linalg.inv(self.Wt_T_Cam) @ points.T).T   # (8,4) segments in camera frame
            cam_pts = cam_h[:, :3]                             # (8,3)
            boxes_cam.append(cam_pts)

        return boxes_cam
            

    def corners_to_2d_mask(self, corners):

        mask_all = np.zeros((self.height, self.width), dtype=np.uint8)

        for corners_3d in corners:
            pts_2d = []
            for p3d in corners_3d:
                p2d = self.project_3d_to_2d(p3d)
                if p2d is not None:
                    pts_2d.append(p2d.astype(float))

            pts_2d = np.asarray(pts_2d, dtype=np.float32)
            if pts_2d.any():
                hull = cv2.convexHull(pts_2d)  # shape (N,1,2)
                cv2.fillConvexPoly(mask_all, hull.astype(np.int32), 1)

        mask_bool = mask_all > 0
        return mask_bool


    def get_masked_depth(self, depth, ts, method="3d"):
        """Mask the robot arm in the depth image.

        Pixels in the arm mask become 0 and stay 0 (the caller ignores them).
        Only zeros outside the arm mask become NOINFO in the caller.
        method="2d" uses the projected arm area; "3d" adds an AABB check.
        """
        assert method in {"2d", "3d"}

        arm_mask = np.zeros(depth.shape, dtype=bool)

        q = self.get_joint_angles_at_ts(ts)
        corners = self.get_bounding_box_vertices(q)
        raw_mask = self.corners_to_2d_mask(corners)

        if method == "2d":
            zero_in_proj = raw_mask & (depth == 0)
            arm_mask = raw_mask.copy()
            depth[arm_mask] = 0
            arm_mask |= zero_in_proj
            return depth, arm_mask

        zero_in_proj = raw_mask & (depth == 0)

        masked_arm = np.zeros_like(depth)
        masked_arm[raw_mask] = depth[raw_mask]

        points, pix = self.depth_to_pointcloud(masked_arm, ds=1)

        aabbs = []
        for seg_corners in corners:
            if seg_corners is None or len(seg_corners) == 0:
                continue
            aabbs.append((seg_corners.min(axis=0), seg_corners.max(axis=0)))

        inside_any = np.zeros(points.shape[0], dtype=bool)
        for bmin, bmax in aabbs:
            inside_any |= (
                (points[:, 0] >= bmin[0]) & (points[:, 0] <= bmax[0]) &
                (points[:, 1] >= bmin[1]) & (points[:, 1] <= bmax[1]) &
                (points[:, 2] >= bmin[2]) & (points[:, 2] <= bmax[2])
            )

        xs = pix[inside_any, 0]
        ys = pix[inside_any, 1]

        arm_mask[ys, xs] = True
        depth[arm_mask] = 0

        arm_mask |= zero_in_proj
        return depth, arm_mask


    def depth_to_pointcloud(self, depth, ds=None):
        ds = ds or self.pointcloud_downsampling

        if ds != 1:
            depth = cv2.resize(
                depth, None,
                fx=1.0/ds, fy=1.0/ds,
                interpolation=cv2.INTER_NEAREST
            )

        fx = float(self.c["fx"]) / ds
        fy = float(self.c["fy"]) / ds
        cx = float(self.c["cx"]) / ds
        cy = float(self.c["cy"]) / ds

        z = depth.astype(np.float32)  # mm

        valid = (z > 0.0)
        ys, xs = np.nonzero(valid)
        if xs.size == 0:
            return np.empty((0, 3), np.float32), np.empty((0, 2), np.int32)

        z_valid = z[ys, xs]  # (N,) mm

        x = (xs.astype(np.float32) - cx) * z_valid / fx
        y = (ys.astype(np.float32) - cy) * z_valid / fy

        pts = np.empty((xs.size, 3), dtype=np.float32)
        pts[:, 0] = x
        pts[:, 1] = y
        pts[:, 2] = z_valid
        pts *= 0.001  # mm -> m

        mask = np.stack((xs, ys), axis=1).astype(np.int32)
        return pts, mask


        
    @staticmethod
    def plot_to_file(depth):
        norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)

        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        cv2.imwrite("masked_depth_colormap.png", colored)