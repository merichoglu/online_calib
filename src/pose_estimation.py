# src/pose_estimation.py

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
        matches: list of cv2.DMatch after matching
        calib: dict containing:
            - 'K0', 'K1': intrinsics
            - 'R', 't': ground-truth stereo extrinsics
        returns: (R_est, t_scaled, inlier_mask)
        """
        # 1) Collect matched points
        pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
        pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])

        # Not enough correspondences to estimate pose
        if len(matches) < 5:
            return (
                np.eye(3),
                np.zeros((3, 1)),
                np.zeros(len(matches), dtype=bool),
            )

        pts0 = np.ascontiguousarray(pts0)
        pts1 = np.ascontiguousarray(pts1)

        # 2a) Preliminary Essential via eight-point RANSAC
        E8, mask8 = cv2.findEssentialMat(
            pts0,
            pts1,
            calib["K0"],
            None,
            calib["K0"],
            None,
            method=cv2.RANSAC,
            prob=self.prob,
            threshold=self.thresh,
        )
        inliers8 = 0
        if mask8 is not None:
            inliers8 = int(mask8.ravel().astype(bool).sum())

        # 2b) Robust Essential via USAC_MAGSAC
        E, mask_usac = cv2.findEssentialMat(
            pts0,
            pts1,
            calib["K0"],
            None,
            calib["K0"],
            None,
            method=cv2.USAC_MAGSAC,
            prob=self.prob,
            threshold=self.thresh,
        )
        if E is None or mask_usac is None:
            # fallback on eight-point if USAC fails
            E = E8
            mask_usac = mask8 if mask8 is not None else np.zeros((len(matches),1), dtype=np.uint8)
        mask_usac = mask_usac.astype(np.uint8)
        mask_usac = np.ascontiguousarray(mask_usac)
        inliers_usac = int(mask_usac.ravel().astype(bool).sum())

        # 2c) Choose the better initialization
        if inliers8 > inliers_usac:
            E_final = E8
            mask_final = mask8.astype(np.uint8)
        else:
            E_final = E
            mask_final = mask_usac

        # 3) Recover pose from chosen E
        _, R_est, t_unit, mask2 = cv2.recoverPose(
            E_final, pts0, pts1, cameraMatrix=calib["K0"], mask=mask_final
        )
        mask_bool = mask2.ravel().astype(bool)

        # 4) Disambiguate translation sign via cheirality check
        K = calib["K0"]
        P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        inlier_idxs = np.where(mask_bool)[0]
        if len(inlier_idxs) > 0:
            sel = inlier_idxs[: min(20, len(inlier_idxs))]
            pts0_s = pts0[sel].T
            pts1_s = pts1[sel].T

            P1_pos = K @ np.hstack((R_est, t_unit))
            P1_neg = K @ np.hstack((R_est, -t_unit))
            Xp = cv2.triangulatePoints(P0, P1_pos, pts0_s, pts1_s)
            Xn = cv2.triangulatePoints(P0, P1_neg, pts0_s, pts1_s)
            Xp = Xp[:3] / Xp[3]
            Xn = Xn[:3] / Xn[3]
            X1p = R_est @ Xp + t_unit
            X1n = R_est @ Xn - t_unit

            front_p = np.sum((Xp[2] > 0) & (X1p[2] > 0))
            front_n = np.sum((Xp[2] > 0) & (X1n[2] > 0))
            if front_n > front_p:
                t_unit = -t_unit

        # ensure translation aligns with ground truth direction
        if np.dot(t_unit.ravel(), calib["t"].ravel()) < 0:
            t_unit = -t_unit

        # 5) Scale translation by known baseline magnitude
        baseline = np.linalg.norm(calib["t"])
        t_scaled = t_unit * baseline

        return R_est, t_scaled, mask_bool
