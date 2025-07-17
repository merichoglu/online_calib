# src/superglue.py

import torch
import numpy as np
import cv2


class SuperGlueMatcher:
    def __init__(self, model, device="cuda"):
        self.model = model.eval().to(device)
        self.device = device

    def match(self, kpts0, desc0, kpts1, desc1, image_shape):
        if len(kpts0) < 1 or len(kpts1) < 1:
            return []

        data = {
            "keypoints0": torch.from_numpy(kpts0).float().unsqueeze(0),
            "keypoints1": torch.from_numpy(kpts1).float().unsqueeze(0),
            "descriptors0": torch.from_numpy(desc0).float().unsqueeze(0),
            "descriptors1": torch.from_numpy(desc1).float().unsqueeze(0),
            "image0": torch.empty((1, 1, *image_shape)),  # dummy placeholder
            "image1": torch.empty((1, 1, *image_shape)),
        }
        for k in data:
            data[k] = data[k].to(self.device)

        with torch.no_grad():
            pred = self.model(data)

        matches = pred["matches0"][0].cpu().numpy()
        matched_kp0 = []
        matched_kp1 = []
        for i, m in enumerate(matches):
            if m > -1:
                matched_kp0.append(kpts0[i])
                matched_kp1.append(kpts1[m])

        matches = [cv2.DMatch(i, m, 0) for i, m in enumerate(matches) if m > -1]
        return matched_kp0, matched_kp1, matches
