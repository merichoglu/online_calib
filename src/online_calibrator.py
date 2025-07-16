# src/online_calibrator.py

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp


class OnlineCalibrator:
    def __init__(self, config_path="../configs/default.yaml"):
        # Load smoothing and threshold configuration
        cfg = yaml.safe_load(open(config_path, "r"))
        smoothing_cfg = cfg.get("smoothing", {})
        # Base complementary filter weight
        self.alpha = smoothing_cfg.get("alpha", 0.1)
        # Rotation jump threshold for adaptive weighting
        self.rot_thresh = smoothing_cfg.get("rot_thresh_deg", 5.0)  # degrees

        # Previous state
        self.R_prev = None  # scipy Rotation
        self.t_prev = None  # (3,) np array

        print(f"OnlineCalibrator initialized: alpha={self.alpha}, \
" \
              f"rot_thresh={self.rot_thresh}°")

    def update(self, R_meas, t_meas, inliers=None):
        """
        Adaptive complementary filter on extrinsic estimates:
          - Compute rotation jump and adapt SLERP weight if jump exceeds threshold
          - Always blend translation linearly

        Args:
            R_meas: 3x3 measured rotation matrix
            t_meas: 3x1 measured translation vector
            inliers: (unused)
        Returns:
            R_filt: 3x3 filtered rotation matrix
            t_filt: 3x1 filtered translation vector
        """
        rot_meas = Rotation.from_matrix(R_meas)
        t_meas_flat = t_meas.flatten()

        # Initialize on first measurement
        if self.R_prev is None:
            self.R_prev = rot_meas
            self.t_prev = t_meas_flat
            return R_meas, t_meas

        # Compute rotation difference (angle in degrees)
        rot_delta = rot_meas * self.R_prev.inv()
        angle_deg = np.degrees(rot_delta.magnitude())

        # Determine effective SLERP weight for rotation
        if angle_deg <= self.rot_thresh:
            alpha_rot = self.alpha
        else:
            # scale down update proportionally
            alpha_rot = self.alpha * (self.rot_thresh / angle_deg)
            alpha_rot = np.clip(alpha_rot, 0.0, self.alpha)

        # SLERP interpolation for rotation
        key_rots = Rotation.from_matrix([self.R_prev.as_matrix(), R_meas])
        slerp = Slerp([0, 1], key_rots)
        R_filt_rot = slerp([alpha_rot])[0]

        # Linear blend for translation with base alpha
        t_filt = (self.alpha * t_meas_flat + (1.0 - self.alpha) * self.t_prev)

        # Update state
        self.R_prev = R_filt_rot
        self.t_prev = t_filt

        return R_filt_rot.as_matrix(), t_filt.reshape(3, 1)
