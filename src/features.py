# src/features.py

import cv2
import yaml


class FeatureExtractor:
    def __init__(self, config_path="../configs/default.yaml"):
        cfg = yaml.safe_load(open(config_path, "r"))
        tp = cfg["feature"]["type"]
        n = cfg["feature"]["n_features"]
        if tp == "ORB":
            self.det = cv2.ORB_create(nfeatures=n)
        elif tp == "SIFT":
            self.det = cv2.SIFT_create(nfeatures=n)
        else:
            raise ValueError(f"Unknown feature type {tp}")
        print(f"FeatureExtractor initialized with {tp} and n_features={n}")

    def detect_and_compute(self, img):
        """Returns keypoints, descriptors."""
        kp, des = self.det.detectAndCompute(img, None)
        return kp, des
