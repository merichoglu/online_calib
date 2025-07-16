# src/matcher.py

import cv2
import yaml
import numpy as np


class Matcher:
    def __init__(self, config_path="../configs/default.yaml"):
        cfg = yaml.safe_load(open(config_path, "r"))
        self.ratio = cfg["matcher"]["ratio_test"]
        # maximum allowed row difference for epipolar constraint (in pixels)
        self.max_row_diff = cfg["matcher"].get("max_row_diff", 2)
        # brute-force Hamming matcher (good for ORB)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        print(
            f"Matcher initialized with ratio={self.ratio}, max_row_diff={self.max_row_diff}"
        )

    def match(self, kp0, kp1, des0, des1):
        """
        kp0, kp1: lists of cv2.KeyPoint for image0 and image1
        des0, des1: descriptor arrays from detect_and_compute()
        returns: filtered list of cv2.DMatch objects after ratio test, epipolar filter,
                 mutual cross-check, and fundamental-matrix pre-filter
        """
        # 1) k-NN match with k=2, ratio test and epipolar row-difference filter
        knn = self.bf.knnMatch(des0, des1, k=2)
        good = []
        for m, n in knn:
            if m.distance < self.ratio * n.distance:
                y0 = kp0[m.queryIdx].pt[1]
                y1 = kp1[m.trainIdx].pt[1]
                if abs(y0 - y1) <= self.max_row_diff:
                    good.append(m)

        # 2) mutual cross-check (forward and backward consistency)
        back_matches = {m.queryIdx: m for m in self.bf.match(des1, des0)}
        mutual = []
        for m in good:
            bm = back_matches.get(m.trainIdx, None)
            if bm is not None and bm.trainIdx == m.queryIdx:
                mutual.append(m)
        if len(mutual) >= 8:
            pts0 = np.float32([kp0[m.queryIdx].pt for m in mutual]).reshape(-1, 1, 2)
            pts1 = np.float32([kp1[m.trainIdx].pt for m in mutual]).reshape(-1, 1, 2)

            try:
                F, fm_mask = cv2.findFundamentalMat(
                    pts0,
                    pts1,
                    cv2.USAC_MAGSAC,  # method
                    1.0,  # initial ransacReprojThreshold
                    0.99,  # confidence level
                )
            except cv2.error:
                # fallback to standard RANSAC if MAGSAC fails
                F, fm_mask = cv2.findFundamentalMat(
                    pts0, pts1, cv2.FM_RANSAC, 1.0, 0.99
                )

            if fm_mask is not None:
                mutual = [m for m, inlier in zip(mutual, fm_mask.ravel()) if inlier]

        return mutual
