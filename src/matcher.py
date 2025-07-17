# src/matcher.py

import cv2
import yaml
import numpy as np


class Matcher:
    def __init__(self, config_path="../configs/default.yaml", stitching=False):
        cfg = yaml.safe_load(open(config_path, "r"))
        self.ratio = cfg["matcher"]["ratio_test"]
        # maximum allowed row difference for epipolar constraint (in pixels)
        self.max_row_diff = cfg["matcher"].get("max_row_diff", 2)
        # choose norm type based on feature type
        feature_type = cfg.get("feature", {}).get("type", "ORB").upper()
        if feature_type == "ORB":
            norm = cv2.NORM_HAMMING
        else:
            # use L2 for SIFT, SURF, etc.
            norm = cv2.NORM_L2
        # brute-force matcher
        self.bf = cv2.BFMatcher(norm, crossCheck=False)

        # If True, skip epipolar & F-matrix filtering (for stitching)
        self.stitching = stitching

        print(
            f"Matcher initialized with ratio={self.ratio}, "
            f"max_row_diff={self.max_row_diff}, stitching={self.stitching}, norm={'HAMMING' if norm==cv2.NORM_HAMMING else 'L2'}"
        )

    def match(self, kp0, kp1, des0, des1):
        """
        kp0, kp1: lists of cv2.KeyPoint
        des0, des1: descriptor arrays
        Returns: filtered list of cv2.DMatch
        """
        # 1) k-NN match with k=2
        knn = self.bf.knnMatch(des0, des1, k=2)
        good = []
        for m, n in knn:
            # Lowe's ratio test
            if m.distance < self.ratio * n.distance:
                if not self.stitching:
                    # epipolar row filter only in stereo mode
                    y0 = kp0[m.queryIdx].pt[1]
                    y1 = kp1[m.trainIdx].pt[1]
                    if abs(y0 - y1) <= self.max_row_diff:
                        good.append(m)
                else:
                    good.append(m)

        # 2) mutual cross-check
        back = {m.queryIdx: m for m in self.bf.match(des1, des0)}
        mutual = []
        for m in good:
            bm = back.get(m.trainIdx)
            if bm is not None and bm.trainIdx == m.queryIdx:
                mutual.append(m)

        # 3) fundamental-matrix filter only in stereo mode
        if not self.stitching and len(mutual) >= 8:
            pts0 = np.float32([kp0[m.queryIdx].pt for m in mutual]).reshape(-1,1,2)
            pts1 = np.float32([kp1[m.trainIdx].pt for m in mutual]).reshape(-1,1,2)
            try:
                F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.USAC_MAGSAC, 1.0, 0.99)
            except cv2.error:
                F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, 1.0, 0.99)
            if mask is not None:
                mutual = [m for m, inl in zip(mutual, mask.ravel()) if inl]

        return mutual
