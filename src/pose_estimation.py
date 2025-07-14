import cv2
import numpy as np
import yaml


class PoseEstimator:
    def __init__(self, config_path="../configs/default.yaml"):
        cfg = yaml.safe_load(open(config_path, "r"))
        ransac_cfg = cfg["ransac"]
        self.prob = ransac_cfg["prob"]
        self.thresh = ransac_cfg["thresh"]

    def estimate(self, kp0, kp1, matches, calib):
        """
        kp0, kp1: keypoint lists from left/right images
        matches: list of cv2.DMatch after ratio test
        calib: dict containing:
            - 'K0', 'K1': intrinsics
            - 'R', 't': ground-truth stereo extrinsics
        returns: (R_est, t_scaled, inlier_mask)
        """
        # 1) Collect matched points
        pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
        pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])

        # 2) Estimate essential matrix via RANSAC
        E, mask = cv2.findEssentialMat(
            pts0,
            pts1,
            calib["K0"],
            method=cv2.RANSAC,
            prob=self.prob,
            threshold=self.thresh,
        )

        # 3) Recover pose: returns one of four possible (R,t) with unit-norm t
        _, R_est, t_unit, mask2 = cv2.recoverPose(
            E, pts0, pts1, cameraMatrix=calib["K0"], mask=mask
        )
        mask_bool = mask2.ravel().astype(bool)

        # 4) Disambiguate translation sign via cheirality check
        K = calib["K0"]
        P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

        # Subset inlier points (up to 20 for speed)
        inlier_idxs = np.where(mask_bool)[0]
        if len(inlier_idxs) > 0:
            sel = inlier_idxs[: min(20, len(inlier_idxs))]
            pts0_s = pts0[sel].T  # 2xN
            pts1_s = pts1[sel].T  # 2xN

            # Projection for positive and negative t
            P1_pos = K @ np.hstack((R_est, t_unit))
            P1_neg = K @ np.hstack((R_est, -t_unit))

            # Triangulate
            Xp = cv2.triangulatePoints(P0, P1_pos, pts0_s, pts1_s)
            Xn = cv2.triangulatePoints(P0, P1_neg, pts0_s, pts1_s)
            # Convert to Euclidean
            Xp = Xp[:3] / Xp[3]
            Xn = Xn[:3] / Xn[3]

            # Depth in cam1 frame
            X1p = R_est @ Xp + t_unit
            X1n = R_est @ Xn - t_unit

            # Count points in front of both cameras
            front_p = np.sum((Xp[2] > 0) & (X1p[2] > 0))
            front_n = np.sum((Xp[2] > 0) & (X1n[2] > 0))

            # If more points are front-facing with -t, flip
            if front_n > front_p:
                t_unit = -t_unit

        # 5) Scale translation by known baseline magnitude
        baseline = np.linalg.norm(calib["t"])
        t_scaled = t_unit * baseline

        return R_est, t_scaled, mask_bool
