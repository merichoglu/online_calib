# src/vo_data_loader.py
import os
import cv2
import numpy as np


def read_vo_calib(calib_path, cam=0):
    """
    Read a KITTI VO-style calib.txt and extract:
      - P_cam and P_cam+1 (rectified projection matrices)
      - R0_rect (3×3 rectification rotation)
      - intrinsics K0, K1
      - identity R (baseline orientation)
      - t (3×1 baseline translation in meters)
    """
    P = {}
    R0_rect = np.eye(3)

    with open(calib_path, "r") as f:
        for line in f:
            # projection matrices
            if line.startswith(f"P{cam}:") or line.startswith(f"P{cam+1}:"):
                key, vals = line.split(":", 1)
                P[key.strip()] = np.fromstring(vals, sep=" ").reshape(3, 4)

            # rectification rotation
            elif line.startswith("R0_rect:") or line.startswith("R_rect_00:"):
                _, vals = line.split(":", 1)
                R0_rect = np.fromstring(vals, sep=" ").reshape(3, 3)

    # grab P0, P1
    P0 = P[f"P{cam}"]
    P1 = P[f"P{cam+1}"]

    # intrinsics
    K0 = P0[:3, :3]
    K1 = P1[:3, :3]

    # compute camera centers C = -K^{-1}·p
    C0 = -np.linalg.inv(K0) @ P0[:, 3]
    C1 = -np.linalg.inv(K1) @ P1[:, 3]
    t = (C1 - C0).reshape(3, 1)

    # cameras are parallel → identity rotation between them
    R = np.eye(3)

    return K0, K1, R, t, R0_rect, P0, P1


class KITTI_VOLoader:
    def __init__(
        self,
        sequence="00",
        base_dir="data_vo/dataset",
        calib_dir="data_vo/dataset",
        cam=0,
    ):
        # paths
        seq_dir = os.path.join(base_dir, "sequences", sequence)
        calib_path = os.path.join(calib_dir, "sequences", sequence, "calib.txt")

        if not os.path.isfile(calib_path):
            raise FileNotFoundError(f"VO calibration file not found: {calib_path}")

        # parse calibration
        (
            self.K0,
            self.K1,
            self.R,  # identity
            self.t,  # baseline
            self.R_rect,  # rectification rotation
            self.P0,  # rectified projection left
            self.P1,
        ) = read_vo_calib(calib_path, cam)

        # GT poses
        poses_file = os.path.join(calib_dir, "poses", f"{sequence}.txt")
        self.gt_poses = []
        if os.path.isfile(poses_file):
            with open(poses_file, "r") as pf:
                for line in pf:
                    vals = [float(x) for x in line.strip().split()]
                    if len(vals) == 12:
                        mat = np.array(vals).reshape(3, 4)
                        self.gt_poses.append((mat[:, :3], mat[:, 3:4]))
        else:
            print(f"Warning: GT poses file not found: {poses_file}")

        # raw image folders
        self.left_dir = os.path.join(seq_dir, "image_0")
        self.right_dir = os.path.join(seq_dir, "image_1")
        if not os.path.isdir(self.left_dir) or not os.path.isdir(self.right_dir):
            raise FileNotFoundError(
                f"VO image folders not found: {self.left_dir}, {self.right_dir}"
            )

        # rectification maps (lazy init)
        self.map0_x = self.map0_y = None
        self.map1_x = self.map1_y = None

        print(f"KITTI VO Loader initialized for sequence {sequence}, cam {cam}")

    def _init_rectify_maps(self, image_size):
        """
        Build undistort+rectify maps once per sequence.
        """
        D0 = np.zeros(5)  # KITTI grayscale cameras have negligible lens distortion
        D1 = np.zeros(5)

        # initUndistortRectifyMap using the rectification rotation and rectified projection
        self.map0_x, self.map0_y = cv2.initUndistortRectifyMap(
            self.K0, D0, self.R_rect, self.P0, image_size, cv2.CV_32FC1
        )
        self.map1_x, self.map1_y = cv2.initUndistortRectifyMap(
            self.K1, D1, self.R_rect, self.P1, image_size, cv2.CV_32FC1
        )

    def image_pairs(self):
        """
        Yields (frame_index, rectified_left, rectified_right)
        """
        for fname in sorted(os.listdir(self.left_dir)):
            idx = int(os.path.splitext(fname)[0])

            img0 = cv2.imread(os.path.join(self.left_dir, fname), cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(os.path.join(self.right_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img0 is None or img1 is None:
                continue

            # initialize rectify maps at first frame
            if self.map0_x is None:
                h, w = img0.shape
                self._init_rectify_maps((w, h))

            # remap to rectified
            img0_rect = cv2.remap(
                img0, self.map0_x, self.map0_y, interpolation=cv2.INTER_LINEAR
            )
            img1_rect = cv2.remap(
                img1, self.map1_x, self.map1_y, interpolation=cv2.INTER_LINEAR
            )

            yield idx, img0_rect, img1_rect
