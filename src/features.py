# src/features.py

import cv2
import yaml
import numpy as np

from src.superpoint import SuperPointFrontend
from models.superpoint import SuperPoint as SPModel


class FeatureExtractor:
    def __init__(self, config_path="../configs/default.yaml"):
        cfg = yaml.safe_load(open(config_path, "r"))
        tp = cfg["feature"]["type"]
        self.type = tp
        n = cfg["feature"].get("n_features", 2000)

        if tp == "ORB":
            self.det = cv2.ORB_create(nfeatures=n)
        elif tp == "SIFT":
            self.det = cv2.SIFT_create(nfeatures=n)
        elif tp == "SuperPoint":
            sp_cfg = cfg["feature"]["superpoint"]
            model = SPModel(sp_cfg)
            device = sp_cfg.get("device", "cuda")
            self.det = SuperPointFrontend(model, device=device)
        else:
            raise ValueError(f"Unknown feature type {tp}")

        print(f"[FeatureExtractor] type={tp}, n_features={n}")

    def detect_and_compute(self, img):
        """
        Returns:
          kp     : list of cv2.KeyPoint
          des    : ndarray of descriptors
          scores : None (for ORB/SIFT) or ndarray of confidence scores (for SuperPoint)
        """
        if self.type in ("ORB", "SIFT"):
            kp, des = self.det.detectAndCompute(img, None)
            if des is None:
                des = np.empty((0, 128), dtype=np.float32)  # 128-dim for SIFT
            else:
                des = des.astype("float32")
            return kp, des, None

        # SuperPoint branch
        pts, des, scores = self.det.run(img)

        # convert (x,y) pairs into cv2.KeyPoint objects, size=1
        kp = [cv2.KeyPoint(float(x), float(y), 1) for x, y in pts]
        return kp, des, scores
