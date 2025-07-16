# src/online_calibrator.py

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp


class OnlineCalibrator:
    def __init__(self, config_path="../configs/default.yaml"):
        cfg = yaml.safe_load(open(config_path, "r"))
        # max blending weight for static smoothing
        self.max_alpha = cfg.get("smoothing", {}).get("alpha", 0.1)
        # track the highest inlier count seen for adaptive smoothing
        self.max_inliers = 0

        self.R_prev = None
        self.t_prev = None
        print(f"OnlineCalibrator initialized with max_alpha={self.max_alpha}")

    def update(self, R_meas, t_meas, inliers=None):
        """
        Apply a complementary filter on extrinsic measures:
          - SLERP between previous and measured rotations
          - Linear blend of translations
          - Adaptive alpha based on inlier ratio if provided

        Args:
            R_meas: 3x3 measured rotation matrix
            t_meas: 3x1 measured translation vector
            inliers: (optional) number of inlier matches used in this update
        Returns:
            R_filt: 3x3 filtered rotation matrix
            t_filt: 3x1 filtered translation vector
        """
        # First measurement -> initialize state
        if self.R_prev is None:
            self.R_prev = Rotation.from_matrix(R_meas)
            self.t_prev = t_meas.flatten()
            return R_meas, t_meas

        # Determine blending factor
        if inliers is not None:
            # update max inliers and compute ratio
            self.max_inliers = max(self.max_inliers, inliers)
            ratio = inliers / float(self.max_inliers) if self.max_inliers > 0 else 1.0
            alpha = min(self.max_alpha, ratio)
        else:
            alpha = self.max_alpha

        # SLERP interpolation for rotation
        prev_rot = self.R_prev
        meas_rot = Rotation.from_matrix(R_meas)
        key_times = [0.0, 1.0]
        key_rots = Rotation.from_matrix([prev_rot.as_matrix(), R_meas])
        slerp = Slerp(key_times, key_rots)
        R_filt_rot = slerp(alpha)
        R_filt = R_filt_rot.as_matrix()

        # Linear blend for translation
        t_filt = alpha * t_meas.flatten() + (1.0 - alpha) * self.t_prev

        # Update state
        self.R_prev = R_filt_rot
        self.t_prev = t_filt

        return R_filt, t_filt.reshape(3, 1)
