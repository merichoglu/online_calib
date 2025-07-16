# src/superpoint.py

import torch
import torch.nn.functional as F
import cv2
import numpy as np


class SuperPointFrontend:
    def __init__(self, model, device='cuda'):
        self.model = model.eval().to(device)
        self.device = device

    def run(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32) / 255.0
        inp = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(inp)

        keypoints, scores = self._extract_keypoints(pred)
        descriptors = pred['descriptors'][0].cpu().numpy()
        return keypoints, descriptors, scores

    def _extract_keypoints(self, pred, conf_thresh=0.015, nms_dist=4):
        prob = pred['prob'][0].cpu().numpy()
        keypoints = self._nms(prob, nms_dist)
        keypoints = [kp for kp in keypoints if prob[kp[1], kp[0]] > conf_thresh]
        scores = [prob[kp[1], kp[0]] for kp in keypoints]
        return np.array(keypoints), np.array(scores)

    def _nms(self, heatmap, dist_thresh):
        keypoints = []
        H, W = heatmap.shape
        for y in range(H):
            for x in range(W):
                val = heatmap[y, x]
                if val < 1e-6:
                    continue
                x0 = max(x - dist_thresh, 0)
                x1 = min(x + dist_thresh + 1, W)
                y0 = max(y - dist_thresh, 0)
                y1 = min(y + dist_thresh + 1, H)
                if val >= np.max(heatmap[y0:y1, x0:x1]):
                    keypoints.append((x, y))
        return keypoints
