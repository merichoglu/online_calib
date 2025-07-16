# scripts/run_pipeline.py

import os
import sys
import argparse
import yaml
import csv
import cv2
import numpy as np

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(proj_root, "src"))

from features import FeatureExtractor
from matcher import Matcher
from pose_estimation import PoseEstimator
from evaluation import reprojection_error


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run feature pipeline and log pose errors"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dataset",
        choices=["stereo", "vo", "cb"],
        default="stereo",
        help="Dataset: stereo, vo, or cb",
    )
    parser.add_argument(
        "-s",
        "--split",
        default="training",
        choices=["training", "testing"],
        help="Stereo split (ignored for vo/cb)",
    )
    parser.add_argument(
        "--base-dir", default="data_cb", help="Base directory for vo or cb datasets"
    )
    parser.add_argument(
        "--samples", type=int, default=5, help="Number of sample match visualizations"
    )
    parser.add_argument(
        "-o", "--out-dir", default="outputs", help="Directory to save outputs"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    # Loader selection
    if args.dataset == "stereo":
        from stereo_data_loader import KITTI_StereoLoader as Loader

        loader = Loader(split=args.split, config_path=args.config)
        calib = loader.calib
        out_base = os.path.join(args.out_dir, "samples_stereo")

    elif args.dataset == "vo":
        from vo_data_loader import KITTI_VOLoader as Loader

        vo_cfg = cfg["data_vo"]
        loader = Loader(
            sequence=vo_cfg["sequence"],
            base_dir=vo_cfg["base_dir"],
            calib_dir=vo_cfg["calib_dir"],
            cam=vo_cfg["cam"],
        )
        calib = {"K0": loader.K0, "K1": loader.K1, "R": loader.R, "t": loader.t}
        out_base = os.path.join(args.out_dir, "samples_vo")

    else:  # cb
        from cb_data_loader import CheckerBoardLoader as Loader

        loader = Loader(base_dir=args.base_dir)
        calib = loader.calib
        out_base = os.path.join(args.out_dir, "samples_cb")

    FE = FeatureExtractor(config_path=args.config)
    M = Matcher(config_path=args.config)
    PE = PoseEstimator(config_path=args.config)

    os.makedirs(out_base, exist_ok=True)
    stats_file = os.path.join(out_base, "stats.csv")
    sample_dir = os.path.join(out_base, "matches")
    os.makedirs(sample_dir, exist_ok=True)

    fieldnames = [
        "frame",
        "mean_error",
        "median_error",
        "inlier_ratio",
        "rot_error_deg",
        "trans_error_deg",
    ]
    with open(stats_file, "w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()

        for count, (idx, img0, img1) in enumerate(loader.image_pairs()):
            kp0, des0 = FE.detect_and_compute(img0)
            kp1, des1 = FE.detect_and_compute(img1)
            matches = M.match(kp0, kp1, des0, des1)
            num_total = len(matches)

            R_est, t_est, mask_bool = PE.estimate(kp0, kp1, matches, calib)
            inliers = [m for i, m in enumerate(matches) if mask_bool[i]]

            stats = reprojection_error(
                kp0, kp1, inliers, num_total, calib["R"], calib["t"], calib
            )

            R_delta = R_est @ calib["R"].T
            rot_error_deg = np.degrees(
                np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1.0, 1.0))
            )
            t_dir_est = t_est.flatten() / np.linalg.norm(t_est)
            t_dir_gt = calib["t"].flatten() / np.linalg.norm(calib["t"])
            trans_error_deg = np.degrees(
                np.arccos(np.clip(np.dot(t_dir_est, t_dir_gt), -1.0, 1.0))
            )

            row = {
                "frame": idx,
                "mean_error": stats["mean_error"],
                "median_error": stats["median_error"],
                "inlier_ratio": stats["inlier_ratio"],
                "rot_error_deg": rot_error_deg,
                "trans_error_deg": trans_error_deg,
            }
            writer.writerow(row)

            if count < args.samples:
                vis = cv2.drawMatches(
                    img0,
                    kp0,
                    img1,
                    kp1,
                    inliers,
                    None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )
                cv2.imwrite(os.path.join(sample_dir, f"{idx}_matches.png"), vis)

            print(
                f"[{idx}] IR={stats['inlier_ratio']:.2f}, R={rot_error_deg:.2f}°, T={trans_error_deg:.2f}°"
            )

    print(f"Stats saved to {stats_file}")
    print(f"Samples saved to {sample_dir}")


if __name__ == "__main__":
    main()
