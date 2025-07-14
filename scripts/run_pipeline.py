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

from stereo_data_loader import KITTI_StereoLoader
from features import FeatureExtractor
from matcher import Matcher
from pose_estimation import PoseEstimator
from evaluation import reprojection_error


def parse_args():
    p = argparse.ArgumentParser(
        description="Run stereo-feature pipeline on KITTI and log pose errors"
    )
    p.add_argument("--config", "-c", default="configs/default.yaml")
    p.add_argument("--split", "-s", default="training", choices=["training", "testing"])
    p.add_argument("--out-dir", "-o", default="outputs")
    p.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of frames to visualize sample matches for.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    loader = KITTI_StereoLoader(split=args.split, config_path=args.config)
    FE = FeatureExtractor(config_path=args.config)
    M = Matcher(config_path=args.config)
    PE = PoseEstimator(config_path=args.config)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    stats_file = os.path.join(out_dir, "stats.csv")

    # CSV fields including rotation and translation errors
    fieldnames = [
        "frame",
        "mean_error",
        "median_error",
        "inlier_ratio",
        "rot_error_deg",
        "trans_error_deg",
    ]
    csv_f = open(stats_file, "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
    writer.writeheader()

    for count, (idx, img0, img1) in enumerate(loader.image_pairs()):
        # 1) Feature extraction and matching
        kp0, des0 = FE.detect_and_compute(img0)
        kp1, des1 = FE.detect_and_compute(img1)
        matches = M.match(des0, des1)
        num_total = len(matches)

        # 2) Pose estimation (unit-norm translation scaled internally)
        R_est, t_est, mask_bool = PE.estimate(kp0, kp1, matches, loader.calib)
        inlier_matches = [m for i, m in enumerate(matches) if mask_bool[i]]

        # 3) Evaluate reprojection error using ground-truth extrinsics
        R_true, t_true = loader.calib["R"], loader.calib["t"]
        stats = reprojection_error(
            kp0, kp1, inlier_matches, num_total, R_true, t_true, loader.calib
        )

        # 4) Compute rotation error (angle between R_est and R_true)
        R_delta = R_est @ R_true.T
        trace_val = np.trace(R_delta)
        angle_rad = np.arccos(np.clip((trace_val - 1) / 2, -1.0, 1.0))
        rot_error_deg = np.degrees(angle_rad)

        # 5) Compute translation error (angle between directions of t_est and t_true)
        t_est_dir = t_est.flatten() / np.linalg.norm(t_est)
        t_true_dir = t_true.flatten() / np.linalg.norm(t_true)
        cos_t = np.dot(t_est_dir, t_true_dir)
        angle_t = np.arccos(np.clip(cos_t, -1.0, 1.0))
        trans_error_deg = np.degrees(angle_t)

        row = {
            "frame": idx,
            "mean_error": stats["mean_error"],
            "median_error": stats["median_error"],
            "inlier_ratio": stats["inlier_ratio"],
            "rot_error_deg": rot_error_deg,
            "trans_error_deg": trans_error_deg,
        }
        writer.writerow(row)

        # Optionally visualize first N samples
        if count < args.samples:
            vis = cv2.drawMatches(
                img0,
                kp0,
                img1,
                kp1,
                inlier_matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            cv2.putText(
                vis,
                f"ME={stats['mean_error']:.1f}, IR={stats['inlier_ratio']:.2f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )
            cv2.imwrite(os.path.join(sample_dir, f"{idx}_matches.png"), vis)

        print(
            f"[{idx}] ME={stats['mean_error']:.1f}, IR={stats['inlier_ratio']:.2f}, "
            f"Rerr={rot_error_deg:.2f}°, Terr={trans_error_deg:.2f}°"
        )

    csv_f.close()
    print(f"Results saved: {stats_file}")
    print(f"Sample visuals: {sample_dir}")


if __name__ == "__main__":
    main()
