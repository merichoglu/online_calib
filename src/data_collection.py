# src/data_collection.py

import numpy as np
from collections import defaultdict


class GridDataCollector:
    def __init__(self, img_shape, grid_rows=8, grid_cols=16, N_max=200):
        """
        img_shape: (h, w)
        grid_rows, grid_cols: how many cells to split the image into
        N_max: maximum features per cell (before scaling by rho)
        """
        self.h, self.w = img_shape
        self.R, self.C = grid_rows, grid_cols
        self.N_max = N_max

        # Compute grid‐cell centers
        ys = (np.arange(self.R) + 0.5) * self.h / self.R
        xs = (np.arange(self.C) + 0.5) * self.w / self.C
        self.centers = np.stack(np.meshgrid(xs, ys), -1)
        self.C0 = np.array([self.w / 2, self.h / 2])

        # Precompute alpha for quadratic fall-off so that rho(center)=1, rho(edge)=0
        all_dist2 = ((self.centers.reshape(-1, 2) - self.C0) ** 2).sum(axis=1)
        d_max2 = all_dist2.max()
        self.alpha = 1.0 / d_max2
        self.rho0 = 1.0

    def collect(self, kp0, kp1, matches):
        """
        kp0, kp1: lists of keypoints on left and right
        matches: list of cv2.DMatch
        Returns a filtered subset of matches.
        """
        # 1) Assign each match to a cell & record disparity
        cell_buckets = defaultdict(list)
        disparities = []
        for idx, m in enumerate(matches):
            x0, y0 = kp0[m.queryIdx].pt
            x1, y1 = kp1[m.trainIdx].pt
            d = abs(x1 - x0)
            disparities.append(d)

            ci = int(y0 / self.h * self.R)
            cj = int(x0 / self.w * self.C)
            ci = np.clip(ci, 0, self.R - 1)
            cj = np.clip(cj, 0, self.C - 1)
            cell_buckets[(ci, cj)].append(idx)

        # 2) For each cell, compute capacity and take top‐disparity matches
        keep = []
        for (ci, cj), idxs in cell_buckets.items():
            # compute rho_ij = rho0 − alpha * ||center−C0||^2
            center = self.centers[ci, cj]
            dist2 = np.sum((center - self.C0) ** 2)
            rho = self.rho0 - self.alpha * dist2
            cap = int(np.floor(rho * self.N_max))
            if cap <= 0:
                continue

            # sort indices by disparity descending
            sorted_idxs = sorted(idxs, key=lambda i: disparities[i], reverse=True)
            for i in sorted_idxs[:cap]:
                keep.append(matches[i])

        return keep
