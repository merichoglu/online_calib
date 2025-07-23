# src/stereo_data_loader.py
import os
import cv2
import yaml
import numpy as np


def read_stereo_calib(calib_path, cam=2):
    """
    Read KITTI stereo calib_cam_to_cam.txt and extract:
      - K0, D0, K1, D1: intrinsics and distortion coefficients
      - R (identity), t: baseline between cam and cam+1
      - R_rect: rectification rotation matrix
      - P0, P1: rectified projection matrices for left/right
    """
    suffix0 = f"{cam:02d}"
    suffix1 = f"{cam+1:02d}"

    K = {}
    D = {}
    P_rect = {}
    R_rect = None

    with open(calib_path, "r") as f:
        for line in f:
            key, vals = line.split(":", 1)
            key = key.strip()
            vals = np.fromstring(vals, sep=" ")

            if key == f"K_{suffix0}":
                K[0] = vals.reshape(3, 3)
            elif key == f"K_{suffix1}":
                K[1] = vals.reshape(3, 3)
            elif key == f"D_{suffix0}":
                D[0] = vals
            elif key == f"D_{suffix1}":
                D[1] = vals
            elif key == f"R_rect_{suffix0}":
                R_rect = vals.reshape(3, 3)
            elif key == f"P_rect_{suffix0}":
                P_rect[0] = vals.reshape(3, 4)
            elif key == f"P_rect_{suffix1}":
                P_rect[1] = vals.reshape(3, 4)

    # Compute baseline translation
    C0 = -np.linalg.inv(K[0]) @ P_rect[0][:, 3]
    C1 = -np.linalg.inv(K[1]) @ P_rect[1][:, 3]
    t = (C1 - C0).reshape(3, 1)
    R = np.eye(3)

    return K[0], D[0], K[1], D[1], R, t, R_rect, P_rect[0], P_rect[1]


class KITTI_StereoLoader:
    def __init__(self, split="training", cam=2, config_path="configs/default.yaml"):
        # Load config for raw data directory
        cfg = yaml.safe_load(open(config_path, "r"))
        config_dir = os.path.dirname(os.path.abspath(config_path))
        raw_dir_setting = cfg["data_stereo"]["raw_dir"]
        raw_dir = (
            raw_dir_setting
            if os.path.isabs(raw_dir_setting)
            else os.path.abspath(os.path.join(config_dir, raw_dir_setting))
        )

        # Calibration folder
        calib_dir = os.path.join(raw_dir, split, "calib_cam_to_cam")
        calib_files = sorted(f for f in os.listdir(calib_dir) if f.endswith(".txt"))
        calib_path = os.path.join(calib_dir, calib_files[0])

        # Parse calibration
        (
            self.K0,
            self.D0,
            self.K1,
            self.D1,
            self.R,
            self.t,
            self.R_rect,
            self.P0,
            self.P1,
        ) = read_stereo_calib(calib_path, cam)

        # Expose calib dict for downstream use
        self.calib = {"K0": self.K0, "K1": self.K1, "R": self.R, "t": self.t}

        # Image directories
        self.left_dir = os.path.join(raw_dir, split, f"image_{cam}")
        self.right_dir = os.path.join(raw_dir, split, f"image_{cam+1}")
        if not os.path.isdir(self.left_dir) or not os.path.isdir(self.right_dir):
            raise FileNotFoundError(
                f"Image folders not found: {self.left_dir}, {self.right_dir}"
            )

        # Rectification maps (to be initialized)
        self.map0_x = self.map0_y = None
        self.map1_x = self.map1_y = None

        print(f"KITTI Stereo Loader initialized for {split} split, cam {cam}")

    def _init_rectify_maps(self, image_size):
        """
        Build undistort + rectify maps using intrinsics, distortion, rectification, and projection.
        """
        self.map0_x, self.map0_y = cv2.initUndistortRectifyMap(
            self.K0, self.D0, self.R_rect, self.P0, image_size, cv2.CV_32FC1
        )
        self.map1_x, self.map1_y = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R_rect, self.P1, image_size, cv2.CV_32FC1
        )

    def image_pairs(self):
        """
        Yields (frame_idx, img_left_rect, img_right_rect).
        """
        files = sorted(os.listdir(self.left_dir))
        for fname in files:
            idx = os.path.splitext(fname)[0]
            img0 = cv2.imread(os.path.join(self.left_dir, fname), cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(os.path.join(self.right_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img0 is None or img1 is None:
                continue

            # Initialize maps on first image
            if self.map0_x is None:
                h, w = img0.shape
                self._init_rectify_maps((w, h))

            # Remap (rectify)
            img0_rect = cv2.remap(
                img0, self.map0_x, self.map0_y, interpolation=cv2.INTER_LINEAR
            )
            img1_rect = cv2.remap(
                img1, self.map1_x, self.map1_y, interpolation=cv2.INTER_LINEAR
            )

            yield idx, img0_rect, img1_rect
