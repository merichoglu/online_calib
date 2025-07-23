# scripts/match_pairs.py

"""
Wrapper to generate match pairs, undistort to a common FoV, then run SuperGlue.
Reads K, D, width & height from your YAML config.
"""
import os
import sys
import json
import argparse
import tempfile
import subprocess

import yaml
import cv2 as cv
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Undistort each pair to a common FoV before SuperGlue",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--img_dir",
        required=True,
        help="Directory of fisheye imgs (e.g. data_cb/CheckerBoard)",
    )
    p.add_argument(
        "--pos_json",
        required=True,
        help="JSON with camera poses like 'front_1': [x,y,z,roll,pitch,yaw]",
    )
    p.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML with intrinsics + matcher params",
    )
    p.add_argument(
        "--output_dir",
        default="outputs/cb_matches_fov",
        help="Where to write undistorted inputs & SuperGlue outputs",
    )
    p.add_argument(
        "--undistort_balance",
        type=float,
        default=0.01,
        help="Balance parameter for fisheye undistortion (0: keep all, 1: crop borders)",
    )
    p.add_argument(
        "--undistort_fov_scale",
        type=float,
        default=0.58,
        help="FOV scale parameter for fisheye undistort mapping",
    )
    p.add_argument(
        "--force_cpu", action="store_true", help="Force SuperGlue to run on CPU"
    )
    p.add_argument(
        "--fast_viz", action="store_true", help="Use OpenCV fast viz (thinner lines)"
    )
    return p.parse_args()


def load_config(path):
    cfg = yaml.safe_load(open(path, "r"))
    K = np.array(cfg["intrinsics"]["K"], dtype=np.float32)
    D = np.array(cfg["intrinsics"]["D"], dtype=np.float32).reshape(1, 4)
    h = int(cfg["intrinsics"]["height"])
    w = int(cfg["intrinsics"]["width"])
    return cfg, K, D, (w, h)


def build_pairs(pos_keys, cams):
    pairs = []
    # same-camera sequential pairs
    for c in cams:
        idxs = sorted(k.split("_")[1] for k in pos_keys if k.startswith(c + "_"))
        for i in range(len(idxs) - 1):
            pairs.append((f"{c}_{idxs[i]}.png", f"{c}_{idxs[i+1]}.png"))
    # cross-camera rings
    order = [("front", "left"), ("left", "rear"), ("rear", "right"), ("right", "front")]
    for a, b in order:
        i1 = {k.split("_")[1] for k in pos_keys if k.startswith(a + "_")}
        i2 = {k.split("_")[1] for k in pos_keys if k.startswith(b + "_")}
        for idx in sorted(i1 & i2, key=int):
            pairs.append((f"{a}_{idx}.png", f"{b}_{idx}.png"))
    return pairs


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # load full config + intrinsics
    cfg, K, D, (width, height) = load_config(args.config)

    # prepare undistort/rectify maps using balance & fov_scale from args
    newcameramtx = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        D,
        (width, height),
        np.eye(3),
        balance=args.undistort_balance,
        fov_scale=args.undistort_fov_scale,
    )
    map1, map2 = cv.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), newcameramtx, (width, height), cv.CV_16SC2
    )

    # load poses & build pair list
    poses = json.load(open(args.pos_json, "r"))
    cams = ["front", "left", "rear", "right"]
    pairs = build_pairs(poses.keys(), cams)

    # undistort & save
    tmp = tempfile.mkdtemp(prefix="undist_inputs_", dir=args.output_dir)
    und_pairs = []
    for a, b in pairs:
        ia = cv.imread(os.path.join(args.img_dir, a), cv.IMREAD_GRAYSCALE)
        ib = cv.imread(os.path.join(args.img_dir, b), cv.IMREAD_GRAYSCALE)
        und_a = cv.remap(
            ia, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT
        )
        und_b = cv.remap(
            ib, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT
        )

        pa = os.path.join(tmp, a)
        pb = os.path.join(tmp, b)
        cv.imwrite(pa, und_a)
        cv.imwrite(pb, und_b)
        und_pairs.append((a, b))

    # write new pairs list
    pairs_file = os.path.join(args.output_dir, "pairs_undist.txt")
    with open(pairs_file, "w") as f:
        for a, b in und_pairs:
            f.write(f"{a} {b}\n")

    # run SuperGlue
    repo = os.path.dirname(os.path.dirname(__file__))
    sg = os.path.join(repo, "models", "superglue_author", "match_pairs.py")
    cmd = [
        sys.executable,
        sg,
        "--input_pairs",
        pairs_file,
        "--input_dir",
        tmp,
        "--output_dir",
        args.output_dir,
        "--superglue",
        cfg["matcher"]["superglue"]["weights"],
        "--resize",
        "-1",
        "--viz",
    ]
    if args.fast_viz:
        cmd.append("--fast_viz")
    if args.force_cpu:
        cmd.append("--force_cpu")
    cmd += ["--match_threshold", str(cfg["matcher"]["superglue"]["match_threshold"])]

    print("Running SuperGlue on undistorted images:", " ".join(cmd))
    subprocess.check_call(cmd)

    print("Done. Undistorted inputs in:", tmp)
    print("Matches in:", args.output_dir)


if __name__ == "__main__":
    main()
