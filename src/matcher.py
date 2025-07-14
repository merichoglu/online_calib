# src/matcher.py

import cv2
import yaml


class Matcher:
    def __init__(self, config_path="../configs/default.yaml"):
        cfg = yaml.safe_load(open(config_path, "r"))
        self.ratio = cfg["matcher"]["ratio_test"]
        # brute-force Hamming matcher by default (good for ORB)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, des0, des1):
        """
        des0, des1: descriptor arrays from detect_and_compute()
        returns: list of good cv2.DMatch objects
        """
        # k-NN match with k=2
        knn = self.bf.knnMatch(des0, des1, k=2)
        good = []
        for m, n in knn:
            if m.distance < self.ratio * n.distance:
                good.append(m)
        return good
