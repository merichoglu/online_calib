# scripts/run_online.py

import os
import sys
import argparse
import yaml
import cv2
import numpy as np
import pandas as pd

# ensure src is on path
top = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(top, "src"))

from features import FeatureExtractor
from matcher import Matcher
from data_collection import GridDataCollector
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
        choices=["stereo", "vo", "cb"],
        default="stereo",
        help="Which dataset to use: stereo, vo, or cb",
    )
    parser.add_argument(
        "-s",
        "--split",
        default="training",
        choices=["training", "testing"],
        help="Stereo split (ignored for vo/cb)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0, help="Delay between frames in seconds"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    # --- loader & calib selection ---
    if args.dataset == "stereo":
        from stereo_data_loader import KITTI_StereoLoader as Loader

        loader = Loader(split=args.split, config_path=args.config)
        calib = loader.calib
        base_out = os.path.join("outputs", "online", "stereo_results")
        seq = args.split

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
        base_out = os.path.join("outputs", "online", "vo_results")
        seq = vo_cfg["sequence"]

    else:
        from cb_data_loader import CheckerBoardLoader as Loader

        loader = Loader(base_dir="data_cb")
        calib = loader.calib
        base_out = os.path.join("outputs", "online", "cb_results")
        seq = "cb"

    # --- initialize pipeline ---
    FE = FeatureExtractor(config_path=args.config)
    M = Matcher(config_path=args.config)
    PE = PoseEstimator(config_path=args.config)
    EKF = OnlineCalibrator(config_path=args.config)

    coll_cfg = cfg.get("collection", {})
    grid_rows = coll_cfg.get("grid_rows", 8)
    grid_cols = coll_cfg.get("grid_cols", 16)
    max_per_cell = coll_cfg.get("max_per_cell", 200)
    collector = None  # init later when we know image size

    # prepare results
    all_results = []

    cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Matches", 800, 600)

    prev_R, prev_t = None, None

    # --- processing loop ---
    for idx_raw, img0, img1 in loader.image_pairs():
        idx = (
            int(idx_raw) if isinstance(idx_raw, str) and idx_raw.isdigit() else idx_raw
        )

        # init collector once we know image size
        if collector is None:
            h, w = img0.shape[:2]
            collector = GridDataCollector(
                img_shape=(h, w),
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                N_max=max_per_cell,
            )

        # 1) feature extraction
        kp0, des0, scores0 = FE.detect_and_compute(img0)
        kp1, des1, scores1 = FE.detect_and_compute(img1)

        # 2) matching
        if M.type == "SuperGlue":
            matches = M.match(
                kp0, kp1, des0, des1, scores0, scores1, image_shape=(h, w)
            )
        else:
            matches = M.match(kp0, kp1, des0, des1, scores0, scores1)

        # 3) grid-based filtering
        matches = collector.collect(kp0, kp1, matches)

        # 4) pose estimation + smoothing
        Rk, tk, mask = PE.estimate(kp0, kp1, matches, calib)
        inliers = int(mask.sum())
        R_filt, t_filt = EKF.update(Rk, tk, inliers=inliers)

        # --- error metrics ---
        gt_R_abs, gt_t_abs = calib["R"], calib["t"]
        abs_trans_err = np.linalg.norm(gt_t_abs - t_filt)
        Rdiff = gt_R_abs.T @ R_filt
        abs_rot_err = np.degrees(np.arccos(np.clip((np.trace(Rdiff) - 1) / 2, -1, 1)))

        if prev_R is not None:
            est_Rrel = prev_R.T @ R_filt
            est_trel = prev_R.T @ (t_filt - prev_t)
        else:
            est_Rrel = np.eye(3)
            est_trel = np.zeros((3, 1))

        if args.dataset == "vo" and isinstance(idx, int) and idx > 0:
            gt_R_prev, gt_t_prev = loader.gt_poses[idx - 1]
            gt_next = loader.gt_poses[idx]
            gt_Rrel = gt_R_prev.T @ gt_next[0]
            gt_trel = gt_R_prev.T @ (gt_next[1] - gt_t_prev)
        else:
            gt_Rrel = np.eye(3)
            gt_trel = np.zeros((3, 1))

        rel_trans_err = np.linalg.norm(gt_trel - est_trel)
        Rdiff_rel = gt_Rrel.T @ est_Rrel
        rel_rot_err = np.degrees(
            np.arccos(np.clip((np.trace(Rdiff_rel) - 1) / 2, -1, 1))
        )

        # accumulate
        all_results.append(
            {
                "frame": idx,
                "abs_tx": t_filt[0, 0],
                "abs_ty": t_filt[1, 0],
                "abs_tz": t_filt[2, 0],
                "gt_abs_tx": gt_t_abs[0, 0],
                "gt_abs_ty": gt_t_abs[1, 0],
                "gt_abs_tz": gt_t_abs[2, 0],
                "abs_trans_err": abs_trans_err,
                "abs_rot_err_deg": abs_rot_err,
                "rel_tx": est_trel[0, 0],
                "rel_ty": est_trel[1, 0],
                "rel_tz": est_trel[2, 0],
                "gt_rel_tx": gt_trel[0, 0],
                "gt_rel_ty": gt_trel[1, 0],
                "gt_rel_tz": gt_trel[2, 0],
                "rel_trans_err": rel_trans_err,
                "rel_rot_err_deg": rel_rot_err,
            }
        )

        # draw + display
        inlier_matches = [m for i, m in enumerate(matches) if mask[i]]
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

    # finalize
    cv2.destroyAllWindows()
    df = pd.DataFrame(all_results)
    save_results(df, base_out, args.dataset, seq)


if __name__ == "__main__":
    main()
