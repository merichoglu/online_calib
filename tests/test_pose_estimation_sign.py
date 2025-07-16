import os
import sys
import numpy as np
import cv2
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pose_estimation import PoseEstimator


def test_translation_direction_aligned():
    pe = PoseEstimator(config_path="configs/default.yaml")
    # simple keypoints and matches
    kp0 = [cv2.KeyPoint(float(i), float(i), 1) for i in range(5)]
    kp1 = [cv2.KeyPoint(float(i), float(i), 1) for i in range(5)]
    matches = [cv2.DMatch(i, i, 0) for i in range(5)]

    calib = {
        "K0": np.eye(3),
        "K1": np.eye(3),
        "R": np.eye(3),
        "t": np.array([[1.0], [0.0], [0.0]]),
    }

    def fake_findEssentialMat(pts0, pts1, K, method, prob, threshold):
        return np.eye(3), np.ones((pts0.shape[0], 1), dtype=np.uint8)

    def fake_recoverPose(E, pts0, pts1, cameraMatrix, mask=None):
        mask2 = np.zeros((pts0.shape[0], 1), dtype=np.uint8)
        return 1, np.eye(3), np.array([[-1.0], [0.0], [0.0]]), mask2

    with patch("cv2.findEssentialMat", side_effect=fake_findEssentialMat):
        with patch("cv2.recoverPose", side_effect=fake_recoverPose):
            _, t_est, _ = pe.estimate(kp0, kp1, matches, calib)

    assert t_est[0, 0] > 0
    assert np.allclose(t_est.ravel(), calib["t"].ravel())
