# src/matcher.py
import cv2
import yaml
import numpy as np
import torch

from src.superglue import SuperGlueMatcher
from models.superglue import SuperGlue as SGModel


class Matcher:
    def __init__(self, config_path="../configs/default.yaml", stitching=False):
        self.stitching = stitching
        cfg = yaml.safe_load(open(config_path, "r"))

        self.type = cfg["matcher"]["type"]
        self.ratio = cfg["matcher"].get("ratio_test", 0.75)
        self.max_row_diff = cfg["matcher"].get("max_row_diff", 2)

        if self.type == "BF":  # ───────────── Brute-Force branch ──────────────
            # decide metric from feature type
            feat_type = cfg["feature"]["type"].lower()
            if feat_type in {"sift", "superpoint"}:
                norm_type = cv2.NORM_L2
                self.float_descriptors = True
            else:  # ORB, BRISK, AKAZE …
                norm_type = cv2.NORM_HAMMING
                self.float_descriptors = False

            self.bf = cv2.BFMatcher(norm_type, crossCheck=False)
            norm_name = "L2" if norm_type == cv2.NORM_L2 else "Hamming"
            print(
                f"[Matcher] type=BF, metric={norm_name}, "
                f"ratio={self.ratio}, max_row_diff={self.max_row_diff}"
            )

        elif self.type == "SuperGlue":  # ────────── SuperGlue branch ──────────
            sg_cfg = cfg["matcher"]["superglue"]
            model = SGModel(sg_cfg)
            self.device = sg_cfg.get("device", "cuda")
            self.sg = SuperGlueMatcher(model, device=self.device)
            print(f"[Matcher] type=SuperGlue, device={self.device}")

        else:
            raise ValueError(f"Unknown matcher type {self.type}")

    # -------------------------------------------------------------------------
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
        kp0, kp1 : lists[cv2.KeyPoint]
        des0, des1 : (N, D) descriptor arrays
        scores0, scores1 : 1-D SuperPoint scores
        image_shape : (H, W) for SuperGlue dummy images
        returns : list[cv2.DMatch]
        """
        if des0 is None or des1 is None or len(des0) == 0 or len(des1) == 0:
            return []

        # ───────────────────────── SuperGlue branch ──────────────────────────
        if self.type == "SuperGlue":
            pts0 = np.array([kp.pt for kp in kp0], dtype=np.float32)
            pts1 = np.array([kp.pt for kp in kp1], dtype=np.float32)

            d0 = des0.astype(np.float32).T  # [D, N]
            d1 = des1.astype(np.float32).T

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

            with torch.no_grad():
                pred = self.sg.model(data)

            matches0 = pred["matches0"][0].cpu().numpy()
            return [
                cv2.DMatch(_queryIdx=i, _trainIdx=int(m), _distance=0)
                for i, m in enumerate(matches0)
                if m > -1
            ]

        # ────────────────────────── BF-matcher branch ────────────────────────
        # Cast descriptors to the dtype the matcher expects
        if self.float_descriptors:
            des0 = des0.astype(np.float32, copy=False)
            des1 = des1.astype(np.float32, copy=False)
        else:
            des0 = des0.astype(np.uint8, copy=False)
            des1 = des1.astype(np.uint8, copy=False)

        # 1) k-NN match + Lowe ratio + optional row filter
        knn = self.bf.knnMatch(des0, des1, k=2)
        good = []
        for m, n in knn:
            if m.distance < self.ratio * n.distance:
                if (
                    self.stitching
                    or abs(kp0[m.queryIdx].pt[1] - kp1[m.trainIdx].pt[1])
                    <= self.max_row_diff
                ):
                    good.append(m)

        # 2) mutual cross-check
        back = {m.queryIdx: m for m in self.bf.match(des1, des0)}
        mutual = [
            m
            for m in good
            if (bm := back.get(m.trainIdx)) and bm.trainIdx == m.queryIdx
        ]

        # 3) fundamental-matrix inlier filter
        if len(mutual) >= 8:
            pts0_f = np.float32([kp0[m.queryIdx].pt for m in mutual]).reshape(-1, 1, 2)
            pts1_f = np.float32([kp1[m.trainIdx].pt for m in mutual]).reshape(-1, 1, 2)
            try:
                _, mask = cv2.findFundamentalMat(
                    pts0_f, pts1_f, cv2.USAC_MAGSAC, 1.0, 0.99
                )
            except cv2.error:  # fallback for < OpenCV 4.5.2
                _, mask = cv2.findFundamentalMat(
                    pts0_f, pts1_f, cv2.FM_RANSAC, 1.0, 0.99
                )
            if mask is not None:
                mutual = [m for m, inlier in zip(mutual, mask.ravel()) if inlier]

        return mutual
