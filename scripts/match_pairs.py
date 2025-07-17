# scripts/match_pairs.py

import os
import sys
import cv2
import json
import argparse
import numpy as np
import yaml

# ensure project root is on PYTHONPATH
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.features import FeatureExtractor
from src.matcher import Matcher


def parse_args():
    p = argparse.ArgumentParser(
        description="Fisheye undistort + ORB matching in stitching mode"
    )
    p.add_argument("--img_dir", required=True,
                   help="Directory of fisheye images (e.g. data_cb/CheckerBoard)")
    p.add_argument("--pos_json", required=True,
                   help="Path to Position.json with keys like 'left_1', etc.")
    p.add_argument("--config", default="configs/default.yaml",
                   help="YAML config (for feature/matcher params & intrinsics)")
    p.add_argument("--output_dir", default="outputs/cb_matches",
                   help="Where to save match visualizations")
    p.add_argument("--max_matches", type=int, default=50,
                   help="Max number of matches to draw per frame")
    p.add_argument("--balance", type=float, default=0.01,
                   help="Balance for undistort (0=crop,1=full FOV)")
    p.add_argument("--fov_scale", type=float, default=0.58,
                   help="FOV scale for fisheye undistort")
    return p.parse_args()


def load_intrinsics(config_path):
    cfg = yaml.safe_load(open(config_path, "r"))
    intr = cfg.get("intrinsics")
    if intr is None or "K" not in intr or "D" not in intr:
        raise KeyError("Config must contain 'intrinsics' with 'K' and 'D'")
    K = np.array(intr["K"], dtype=np.float32)
    D = np.array(intr["D"], dtype=np.float32)
    return K, D


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # load intrinsics
    K, D = load_intrinsics(args.config)

    # load frame positions
    with open(args.pos_json, "r") as f:
        pos = json.load(f)

    # determine image size from first key
    first = next(iter(pos.keys()))
    cam0, idx0 = first.split("_")
    sample = cv2.imread(os.path.join(args.img_dir, f"{cam0}_{idx0}.png"),
                        cv2.IMREAD_GRAYSCALE)
    if sample is None:
        raise RuntimeError(f"Cannot open sample image {cam0}_{idx0}.png")
    h, w = sample.shape

    # build undistort/rectify map once
    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=args.balance, fov_scale=args.fov_scale
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), newK, (w, h), cv2.CV_16SC2
    )

    # init extractor + stitching matcher
    extractor = FeatureExtractor(config_path=args.config)
    matcher = Matcher(config_path=args.config, stitching=True)
    # optionally override ratio from config:
    # matcher.ratio = 0.9

    # define camera order and indices
    cams = ["left", "front", "right", "rear"]
    cam_idxs = {
        c: set(k.split("_")[1] for k in pos if k.startswith(c + "_"))
        for c in cams
    }

    for i in range(len(cams)):
        c0 = cams[i]
        c1 = cams[(i + 1) % len(cams)]
        frames = sorted(cam_idxs[c0] & cam_idxs[c1], key=lambda x: int(x))
        if not frames:
            print(f"No overlap between {c0} and {c1}, skipping.")
            continue
        print(f"Matching {c0} ↔ {c1} on frames: {frames}")

        for idx in frames:
            # load & undistort
            im0 = cv2.imread(os.path.join(args.img_dir, f"{c0}_{idx}.png"),
                             cv2.IMREAD_GRAYSCALE)
            im1 = cv2.imread(os.path.join(args.img_dir, f"{c1}_{idx}.png"),
                             cv2.IMREAD_GRAYSCALE)
            if im0 is None or im1 is None:
                print(f"  → failed to load {c0}_{idx} or {c1}_{idx}")
                continue
            u0 = cv2.remap(im0, map1, map2, interpolation=cv2.INTER_LINEAR)
            u1 = cv2.remap(im1, map1, map2, interpolation=cv2.INTER_LINEAR)

            # detect & compute
            kp0, des0 = extractor.detect_and_compute(u0)
            kp1, des1 = extractor.detect_and_compute(u1)
            if des0 is None or des1 is None:
                print(f"  → no descriptors at frame {idx}")
                continue

            # match in stitching mode
            matches = matcher.match(kp0, kp1, des0, des1)
            if not matches:
                print(f"  → no matches at frame {idx}")
                continue

            # draw top matches
            top = sorted(matches, key=lambda m: m.distance)[:args.max_matches]
            vis = cv2.drawMatches(
                u0, kp0, u1, kp1, top, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            outp = os.path.join(args.output_dir,
                                f"match_{c0}_{c1}_{idx}.png")
            cv2.imwrite(outp, vis)

    print("Done.")


if __name__ == "__main__":
    main()
