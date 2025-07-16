# src/cb_data_loader.py

import os
import cv2
import json
import numpy as np


def euler_to_rot(yaw_deg, pitch_deg, roll_deg):
    """Build a rotation matrix from yaw, pitch, roll (in degrees)."""
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    Rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    return Rz @ Ry @ Rx


class CheckerBoardLoader:
    """
    Loader for the Ford CheckerBoard calibration dataset.

    Expects:

      data_cb/
      ├── CheckerBoard/   # left_*.png, right_*.png
      └── Position.json   # {"left_i": [x,y,z,yaw,pitch,roll], "right_i": [...] , ...}

    Yields (frame_id, imgL, imgR) and provides .calib with intrinsics + baseline.
    """

    def __init__(self, base_dir):
        self.base_dir = base_dir

        # 1) Hard-coded fisheye intrinsics & zero distortion from the sample code:
        K = np.array(
            [
                [550.0395, 0, 960],
                [0, 533.372225, 768],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        D = np.zeros((4, 1), dtype=np.float64)
        self.K0 = K
        self.K1 = K

        # 2) Precompute undistort/rectify maps
        width, height = 1920, 1536
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (width, height), np.eye(3), balance=0.0, new_size=(width, height)
        )
        self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), newK, (width, height), cv2.CV_16SC2
        )

        # 3) Parse JSON extrinsics (one entry per left_i / right_i)
        pose_file = os.path.join(base_dir, "Position.json")
        with open(pose_file, "r") as fh:
            self._poses = json.load(fh)

        # 4) Compute a single “baseline” using the FIRST pair left_1 & right_1
        L = self._poses["left_1"]  # [x,y,z,yaw,pitch,roll]
        R = self._poses["right_1"]
        tL = np.array(L[0:3], dtype=np.float64).reshape(3, 1)
        tR = np.array(R[0:3], dtype=np.float64).reshape(3, 1)
        RL = euler_to_rot(L[3], L[4], L[5])
        RR = euler_to_rot(R[3], R[4], R[5])

        # left→right
        self.R = RL.T @ RR
        self.t = RL.T @ (tR - tL)

        # 5) Collect and sort all left/right image paths
        cb_dir = os.path.join(base_dir, "CheckerBoard")
        lefts = sorted(f for f in os.listdir(cb_dir) if f.startswith("left_"))
        rights = sorted(f for f in os.listdir(cb_dir) if f.startswith("right_"))
        assert len(lefts) == len(rights), "Left/right count mismatch"
        self.left_paths = [os.path.join(cb_dir, f) for f in lefts]
        self.right_paths = [os.path.join(cb_dir, f) for f in rights]

    @property
    def calib(self):
        return {"K0": self.K0, "K1": self.K1, "R": self.R, "t": self.t}

    def image_pairs(self):
        """Yields (frame_id, imgL, imgR) with both images undistorted."""
        for lp, rp in zip(self.left_paths, self.right_paths):
            fid = os.path.splitext(os.path.basename(lp))[0]
            imgL = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
            imgR = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
            # undistort + rectify
            imgL = cv2.remap(
                imgL,
                self._map1,
                self._map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            imgR = cv2.remap(
                imgR,
                self._map1,
                self._map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            yield fid, imgL, imgR
