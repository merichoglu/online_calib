# src/matcher.py

import cv2
import yaml
import numpy as np
import torch

from superglue import SuperGlueMatcher
from models.superglue import SuperGlue as SGModel


class Matcher:
    def __init__(self, config_path="../configs/default.yaml", stitching=False):
        cfg = yaml.safe_load(open(config_path, "r"))
        tp = cfg["matcher"]["type"]
        self.type = tp
        self.ratio = cfg["matcher"].get("ratio_test", 0.75)
        self.max_row_diff = cfg["matcher"].get("max_row_diff", 2)

        if tp == "BF":
            # brute‐force Hamming matcher (good for ORB)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            print(
                f"[Matcher] type=BF, ratio={self.ratio}, max_row_diff={self.max_row_diff}"
            )

        elif tp == "SuperGlue":
            sg_cfg = cfg["matcher"]["superglue"]
            model = SGModel(sg_cfg)
            device = sg_cfg.get("device", "cuda")
            self.sg = SuperGlueMatcher(model, device=device)
            self.device = device
            print(f"[Matcher] type=SuperGlue, device={device}")

        else:
            raise ValueError(f"Unknown matcher type {tp}")

    def match(
        self,
        kp0,
        kp1,
        des0,
        des1,
        scores0=None,
        scores1=None,
        image_shape=None,
    ):
        """
        kp0, kp1: lists of cv2.KeyPoint
        des0, des1: descriptor arrays (shape [N, D])
        scores0, scores1: 1D arrays of SuperPoint scores
        image_shape: (H, W) tuple needed for SuperGlue
        returns: list of cv2.DMatch objects
        """
        if self.type == "SuperGlue":
            # 1) keypoint coordinates
            pts0 = np.array([kp.pt for kp in kp0], dtype=np.float32)
            pts1 = np.array([kp.pt for kp in kp1], dtype=np.float32)

            # 2) descriptors: [N, D] -> [D, N]
            d0 = des0.astype(np.float32).T
            d1 = des1.astype(np.float32).T

            # 3) build data dict in expected formats
            data = {
                "keypoints0": torch.from_numpy(pts0).unsqueeze(0).to(self.device),
                "keypoints1": torch.from_numpy(pts1).unsqueeze(0).to(self.device),
                "descriptors0": torch.from_numpy(d0).unsqueeze(0).to(self.device),
                "descriptors1": torch.from_numpy(d1).unsqueeze(0).to(self.device),
                "scores0": torch.from_numpy(scores0).unsqueeze(0).to(self.device),
                "scores1": torch.from_numpy(scores1).unsqueeze(0).to(self.device),
                "image0": torch.empty((1, 1, *image_shape)).to(self.device),
                "image1": torch.empty((1, 1, *image_shape)).to(self.device),
            }

            # 4) run SuperGlue
            with torch.no_grad():
                pred = self.sg.model(data)

            matches0 = pred["matches0"][0].cpu().numpy()
            # 5) convert to cv2.DMatch list
            matches = [
                cv2.DMatch(_queryIdx=i, _trainIdx=int(m), _distance=0)
                for i, m in enumerate(matches0)
                if m > -1
            ]
            return matches

        # ---- BF / ORB branch ----
        # 1) k‐NN match + ratio test + row filter
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

        # 2) mutual cross‐check
        back = {m.queryIdx: m for m in self.bf.match(des1, des0)}
        mutual = []
        for m in good:
            bm = back.get(m.trainIdx)
            if bm and bm.trainIdx == m.queryIdx:
                mutual.append(m)

        # 3) fundamental‐matrix inlier filter
        if len(mutual) >= 8:
            pts0_f = np.float32([kp0[m.queryIdx].pt for m in mutual]).reshape(-1, 1, 2)
            pts1_f = np.float32([kp1[m.trainIdx].pt for m in mutual]).reshape(-1, 1, 2)
            try:
                F, mask = cv2.findFundamentalMat(
                    pts0_f, pts1_f, cv2.USAC_MAGSAC, 1.0, 0.99
                )
            except cv2.error:
                F, mask = cv2.findFundamentalMat(
                    pts0_f, pts1_f, cv2.FM_RANSAC, 1.0, 0.99
                )
            if mask is not None:
                mutual = [m for m, inlier in zip(mutual, mask.ravel()) if inlier]

        return mutual
