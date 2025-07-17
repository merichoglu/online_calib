# src/superpoint.py

import torch
import torch.nn.functional as F
import cv2
import numpy as np


class SuperPointFrontend:
    def __init__(self, model, device="cuda"):
        self.model = model.eval().to(device)
        self.device = device

    def run(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32) / 255.0
        inp = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(inp)

        # pred['prob'][0]: shape (C, H, W) where C=65 (including dustbin)
        prob_tensor = pred["prob"][0].cpu().numpy()
        # collapse channels 0..63 into a single 2D heatmap (ignore dustbin at index 64)
        heatmap = prob_tensor[:64].sum(axis=0)

        # now do NMS on the 2D heatmap
        keypoint_coords = self._nms(heatmap, nms_dist=4)
        # filter by confidence threshold
        filtered = [(x, y) for x, y in keypoint_coords if heatmap[y, x] > 0.015]
        scores = [float(heatmap[y, x]) for x, y in filtered]

        # get descriptors corresponding to remaining keypoints
        descriptors = pred["descriptors"][0].cpu().numpy()  # shape (D, H, W)
        desc_list = []
        for x, y in filtered:
            desc_list.append(descriptors[:, y, x])
        descs = (
            np.stack(desc_list, axis=0)
            if desc_list
            else np.zeros((0, descriptors.shape[0]))
        )

        return np.array(filtered), descs, np.array(scores)

    def _nms(self, heatmap, nms_dist):
        """
        Non-maximum suppression on a 2D heatmap.
        :param heatmap: 2D numpy array
        :param nms_dist: minimum distance for suppression
        :return: list of (x, y) keypoint coordinates
        """
        keypoints = []
        H, W = heatmap.shape
        for y in range(H):
            for x in range(W):
                val = heatmap[y, x]
                if val < 1e-6:
                    continue
                x0 = max(x - nms_dist, 0)
                x1 = min(x + nms_dist + 1, W)
                y0 = max(y - nms_dist, 0)
                y1 = min(y + nms_dist + 1, H)
                if val >= np.max(heatmap[y0:y1, x0:x1]):
                    keypoints.append((x, y))
        return keypoints
