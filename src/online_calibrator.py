# src/online_calibrator.py

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp


class OnlineCalibrator:
    def __init__(self, config_path="../configs/default.yaml"):
        # Load smoothing and threshold configuration
        cfg = yaml.safe_load(open(config_path, "r"))
        smoothing_cfg = cfg.get("smoothing", {})

        # Complementary filter parameters
        self.alpha = smoothing_cfg.get("alpha", 0.1)
        self.rot_thresh = smoothing_cfg.get("rot_thresh_deg", 5.0)  # degrees

        # Warm‐start: number of frames to batch‐initialize on
        self.warmup_frames = smoothing_cfg.get("warmup_frames", 10)

        # Buffers for the warmup phase
        self.buffered_R = []  # list of scipy Rotation
        self.buffered_t = []  # list of (3,) arrays
        self.frame_count = 0
        self.warmup_done = False

        # Filter state (to be set after warmup)
        self.R_prev = None  # scipy Rotation
        self.t_prev = None  # (3,) np array

        print(
            f"OnlineCalibrator initialized: "
            f"alpha={self.alpha}, rot_thresh={self.rot_thresh}°, "
            f"warmup_frames={self.warmup_frames}"
        )

    def update(self, R_meas, t_meas, inliers=None):
        """
        - For the first self.warmup_frames calls: collect R_meas, t_meas.
        - On the last warmup frame: compute a batch average and set R_prev, t_prev.
        - After warmup: apply SLERP + adaptive complementary filter using inlier ratio.
        """
        rot_meas = Rotation.from_matrix(R_meas)
        t_meas_flat = t_meas.flatten()
        self.frame_count += 1

        # === Warm-start phase ===
        if not self.warmup_done:
            self.buffered_R.append(rot_meas)
            self.buffered_t.append(t_meas_flat)

            # Still collecting
            if self.frame_count < self.warmup_frames:
                return R_meas, t_meas

            # On the final warmup frame: compute batch‐average pose
            t_mean = np.mean(np.stack(self.buffered_t, axis=0), axis=0)

            # Rotation average via quaternions
            quats = np.stack([r.as_quat() for r in self.buffered_R], axis=0)
            q0 = quats[0]
            for i in range(1, quats.shape[0]):
                if np.dot(quats[i], q0) < 0:
                    quats[i] *= -1
            q_mean = np.mean(quats, axis=0)
            q_mean /= np.linalg.norm(q_mean)
            R_mean = Rotation.from_quat(q_mean)

            # Initialize filter state
            self.R_prev = R_mean
            self.t_prev = t_mean
            self.warmup_done = True

            # Return averaged pose at warmup frame
            return R_mean.as_matrix(), t_mean.reshape(3, 1)

        # === Complementary-filter phase ===
        # Compute rotation jump
        rot_delta = rot_meas * self.R_prev.inv()
        angle_deg = np.degrees(rot_delta.magnitude())

        # Base SLERP weight for rotation
        if angle_deg <= self.rot_thresh:
            alpha_rot = self.alpha
        else:
            alpha_rot = self.alpha * (self.rot_thresh / angle_deg)
            alpha_rot = np.clip(alpha_rot, 0.0, self.alpha)

        # Adapt weights based on inlier ratio, paper does it this way
        if inliers is not None and isinstance(inliers, (np.ndarray, list)):
            inliers_arr = np.array(inliers, dtype=bool)
            ratio = inliers_arr.sum() / max(1, inliers_arr.size)
            # scale rotation weight and translation weight
            alpha_rot *= ratio**2
            alpha_rot = np.clip(alpha_rot, 0.0, self.alpha)
            alpha_trans = self.alpha * (ratio**2)
        else:
            alpha_trans = self.alpha

        # SLERP interpolation for rotation
        key_rots = Rotation.from_matrix([self.R_prev.as_matrix(), R_meas])
        slerp = Slerp([0, 1], key_rots)
        R_filt = slerp([alpha_rot])[0]

        # Linear blend for translation
        t_filt = alpha_trans * t_meas_flat + (1.0 - alpha_trans) * self.t_prev

        # Update state
        self.R_prev = R_filt
        self.t_prev = t_filt

        return R_filt.as_matrix(), t_filt.reshape(3, 1)
