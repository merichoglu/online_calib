# scripts/run_online.py

import os
import sys
import argparse
import yaml
import cv2
import numpy as np
import csv
import pandas as pd

# ensure src is on path
top = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(top, "src"))

from features import FeatureExtractor
from matcher import Matcher
from pose_estimation import PoseEstimator
from online_calibrator import OnlineCalibrator
from scripts.result_saver import save_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Online extrinsic calibration streaming"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dataset",
        choices=["stereo", "vo"],
        default="stereo",
        help="Which dataset to use: stereo or vo",
    )
    parser.add_argument(
        "-s",
        "--split",
        default="training",
        choices=["training", "testing"],
        help="Stereo split (ignored for VO)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0, help="Delay between frames in seconds"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    # select loader + calibration
    if args.dataset == "stereo":
        from stereo_data_loader import KITTI_StereoLoader as Loader

        loader = Loader(split=args.split, config_path=args.config)
        calib = loader.calib  # {'K0','K1','R','t'}
    else:
        from vo_data_loader import KITTI_VOLoader as Loader

        vo_cfg = cfg["data_vo"]
        loader = Loader(
            sequence=vo_cfg["sequence"],
            base_dir=vo_cfg["base_dir"],
            calib_dir=vo_cfg["calib_dir"],
            cam=vo_cfg["cam"],
        )
        calib = {"K0": loader.K0, "K1": loader.K1, "R": loader.R, "t": loader.t}

    FE = FeatureExtractor(config_path=args.config)
    M = Matcher(config_path=args.config)
    PE = PoseEstimator(config_path=args.config)
    EKF = OnlineCalibrator(config_path=args.config)

    # prepare output directories
    if args.dataset == "vo":
        base_out = os.path.join("outputs", "online", "vo_results")
        seq = cfg["data_vo"]["sequence"]
    else:
        base_out = os.path.join("outputs", "online", "stereo_results")
        seq = args.split
    os.makedirs(base_out, exist_ok=True)
    tables_dir = os.path.join(base_out, "tables")
    graphs_dir = os.path.join(base_out, "graphs")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    # CSV path
    csv_path = os.path.join(tables_dir, f"{args.dataset}_seq_{seq}.csv")
    cf = open(csv_path, "w", newline="")
    writer = csv.writer(cf)
    writer.writerow(
        [
            "frame",
            "abs_tx",
            "abs_ty",
            "abs_tz",
            "gt_abs_tx",
            "gt_abs_ty",
            "gt_abs_tz",
            "abs_trans_err",
            "abs_rot_err_deg",
            "rel_tx",
            "rel_ty",
            "rel_tz",
            "gt_rel_tx",
            "gt_rel_ty",
            "gt_rel_tz",
            "rel_trans_err",
            "rel_rot_err_deg",
        ]
    )

    cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Matches", 800, 600)

    prev_R = None
    prev_t = None

    for idx_raw, img0, img1 in loader.image_pairs():
        try:
            idx = int(idx_raw)
        except ValueError:
            idx = idx_raw

        kp0, des0 = FE.detect_and_compute(img0)
        kp1, des1 = FE.detect_and_compute(img1)
        # inject epipolar constraint: pass keypoints to matcher
        matches = M.match(kp0, kp1, des0, des1)

        Rk, tk, mask = PE.estimate(kp0, kp1, matches, calib)
        R_filt, t_filt = EKF.update(Rk, tk)

        gt_R_abs = calib["R"]
        gt_t_abs = calib["t"]

        # absolute errors
        abs_trans_err = np.linalg.norm(gt_t_abs - t_filt)
        Rdiff = gt_R_abs.T @ R_filt
        ang = np.clip((np.trace(Rdiff) - 1) / 2, -1.0, 1.0)
        abs_rot_err = np.degrees(np.arccos(ang))

        # relative errors
        if prev_R is not None:
            est_Rrel = prev_R.T @ R_filt
            est_trel = prev_R.T @ (t_filt - prev_t)
        else:
            est_Rrel = np.eye(3)
            est_trel = np.zeros((3, 1))

        if args.dataset == "vo":
            gt_R_prev, gt_t_prev = (
                loader.gt_poses[idx - 1] if idx > 0 else (gt_R_abs, gt_t_abs)
            )
            gt_Rrel = gt_R_prev.T @ loader.gt_poses[idx][0]
            gt_trel = gt_R_prev.T @ (loader.gt_poses[idx][1] - gt_t_prev)
        else:
            gt_Rrel = np.eye(3)
            gt_trel = np.zeros((3, 1))

        rel_trans_err = np.linalg.norm(gt_trel - est_trel)
        Rdiff_rel = gt_Rrel.T @ est_Rrel
        angr = np.clip((np.trace(Rdiff_rel) - 1) / 2, -1.0, 1.0)
        rel_rot_err = np.degrees(np.arccos(angr))

        # write CSV
        writer.writerow(
            [
                idx,
                t_filt[0, 0],
                t_filt[1, 0],
                t_filt[2, 0],
                gt_t_abs[0, 0],
                gt_t_abs[1, 0],
                gt_t_abs[2, 0],
                abs_trans_err,
                abs_rot_err,
                est_trel[0, 0],
                est_trel[1, 0],
                est_trel[2, 0],
                gt_trel[0, 0],
                gt_trel[1, 0],
                gt_trel[2, 0],
                rel_trans_err,
                rel_rot_err,
            ]
        )

        # display matches
        inliers = [m for i, m in enumerate(matches) if mask[i]]
        vis = cv2.drawMatches(
            img0,
            kp0,
            img1,
            kp1,
            inliers,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.putText(
            vis,
            f"abs_err={abs_trans_err:.2f}m rot={abs_rot_err:.1f}°",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            vis,
            f"rel_err={rel_trans_err:.2f}m rot={rel_rot_err:.1f}°",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.imshow(
            "Matches",
            cv2.resize(vis, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
        )
        if cv2.waitKey(int(args.delay * 1000)) & 0xFF == ord("q"):
            break

        prev_R, prev_t = R_filt, t_filt

    cf.close()
    cv2.destroyAllWindows()

    # post-process: save tables & graphs
    df = pd.read_csv(csv_path)
    save_results(df, base_out, args.dataset, seq)


if __name__ == "__main__":
    main()
