# src/online_calibrator.py

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp


class OnlineCalibrator:
    def __init__(self, config_path="../configs/default.yaml"):
        cfg = yaml.safe_load(open(config_path, "r"))
        self.alpha = cfg.get("smoothing", {}).get("alpha", 0.1)
        self.R_prev = None
        self.t_prev = None
        print(f"OnlineCalibrator initialized with alpha={self.alpha}")

    def update(self, R_meas, t_meas):
        """
        Apply a complementary filter on extrinsic measures:
          - SLERP between previous and measured rotations
          - Linear blend of translations

        Args:
            R_meas: 3x3 measured rotation matrix
            t_meas: 3x1 measured translation vector
        Returns:
            R_filt: 3x3 filtered rotation matrix
            t_filt: 3x1 filtered translation vector
        """
        if self.R_prev is None:
            self.R_prev = Rotation.from_matrix(R_meas)
            self.t_prev = t_meas.flatten()
            return R_meas, t_meas

        R_prev_rot = self.R_prev
        R_meas_rot = Rotation.from_matrix(R_meas)

        # SLERP interpolation
        key_times = [0.0, 1.0]
        key_rots = Rotation.from_matrix([R_prev_rot.as_matrix(), R_meas])
        slerp = Slerp(key_times, key_rots)
        R_filt_rot = slerp(self.alpha)
        R_filt = R_filt_rot.as_matrix()

        # filter on translation
        t_filt = self.alpha * t_meas.flatten() + (1 - self.alpha) * self.t_prev

        self.R_prev = R_filt_rot
        self.t_prev = t_filt

        return R_filt, t_filt.reshape(3, 1)
