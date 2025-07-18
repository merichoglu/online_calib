# scripts/match_pairs.py

import os
import sys
import cv2
import yaml
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

# ensure project root on PYTHONPATH
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.features import FeatureExtractor
from src.matcher  import Matcher


def parse_args():
    p = argparse.ArgumentParser(
        description="Undistort + sequential same‐camera matching"
    )
    p.add_argument("--img_dir",    required=True, help="directory of fisheye imgs")
    p.add_argument("--pos_json",   required=True, help="json with keys like 'front_1'")
    p.add_argument(
        "--config",
        default="configs/default.yaml",
        help="yaml for intrinsics + feature/matcher params",
    )
    p.add_argument(
        "--output_dir",
        default="outputs/cb_matches",
        help="where to save match visualizations",
    )
    p.add_argument(
        "--max_matches", type=int, default=50, help="max matches per frame pair"
    )
    p.add_argument(
        "--balance",
        type=float,
        default=0.5,
        help="undistort balance (unused for same‐cam but kept)",
    )
    p.add_argument(
        "--fov_scale",
        type=float,
        default=1.0,
        help="undistort fov_scale (unused for same‐cam but kept)",
    )
    return p.parse_args()


def load_config(path):
    return yaml.safe_load(open(path, "r"))


def load_intrinsics_all(cfg, cams):
    intr = cfg.get("intrinsics", {})
    intrinsics = {}
    # flat intrinsics for all cams
    if "K" in intr and "D" in intr:
        K = np.array(intr["K"], dtype=np.float32)
        D = np.array(intr["D"], dtype=np.float32)
        w = int(intr.get("width"))
        h = int(intr.get("height"))
        for c in cams:
            intrinsics[c] = (K, D, w, h)
    else:
        # per‐camera intrinsics
        for c in cams:
            v = intr.get(c)
            if v is None or "K" not in v or "D" not in v:
                raise KeyError(f"missing intrinsics for camera '{c}' in config")
            K = np.array(v["K"], dtype=np.float32)
            D = np.array(v["D"], dtype=np.float32)
            w = int(v["width"])
            h = int(v["height"])
            intrinsics[c] = (K, D, w, h)
    return intrinsics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = load_config(args.config)
    # Order here is only for the final per‐cam loop:
    cams = ["left", "front", "right", "rear"]
    intrinsics = load_intrinsics_all(cfg, cams)

    # load frame list from Position.json
    pos = json.load(open(args.pos_json, "r"))
    cam_idxs = {
        c: {k.split("_")[1] for k in pos if k.startswith(c + "_")} for c in cams
    }

    # pick any camera to get image size
    _, _, w, h = intrinsics[cams[0]]

    # precompute undistort‐only rectify maps for each camera
    rect_maps = {}
    for c in cams:
        K, D, _, _ = intrinsics[c]
        Rr = np.eye(3, dtype=np.float32)
        P  = K
        m0, m1 = cv2.fisheye.initUndistortRectifyMap(
            K, D, Rr, P, (w, h), cv2.CV_16SC2
        )
        # same maps for both images in a pair
        rect_maps[(c, c)] = (m0, m1, m0, m1)

    extractor = FeatureExtractor(config_path=args.config)
    matcher   = Matcher(config_path=args.config, stitching=False)

    method = extractor.type.lower()  # e.g. 'orb', 'sift', 'superpoint'
    out_sub = os.path.join(args.output_dir, method)
    os.makedirs(out_sub, exist_ok=True)

    # match sequential frames for each camera in this order:
    cam_order = ["front", "rear", "left", "right"]
    for c in cam_order:
        frames = sorted(cam_idxs[c], key=int)
        if len(frames) < 2:
            print(f"no frames for camera '{c}', skipping")
            continue

        print(f"matching {c} sequential frames: {frames}")
        m00, m01, m10, m11 = rect_maps[(c, c)]
        for i in range(len(frames) - 1):
            idx0, idx1 = frames[i], frames[i + 1]
            im0 = cv2.imread(
                os.path.join(args.img_dir, f"{c}_{idx0}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            im1 = cv2.imread(
                os.path.join(args.img_dir, f"{c}_{idx1}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            if im0 is None or im1 is None:
                print(f"  → failed load {c}_{idx0} or {c}_{idx1}")
                continue

            u0 = cv2.remap(im0, m00, m01, interpolation=cv2.INTER_LINEAR)
            u1 = cv2.remap(im1, m10, m11, interpolation=cv2.INTER_LINEAR)

            kp0, des0, sc0 = extractor.detect_and_compute(u0)
            kp1, des1, sc1 = extractor.detect_and_compute(u1)
            if des0 is None or des1 is None:
                print(f"  → no descriptors at {c}_{idx0} or {c}_{idx1}")
                continue

            matches = matcher.match(kp0, kp1, des0, des1, sc0, sc1, (h, w))
            if not matches:
                print(f"  → no matches at {c}_{idx0}-{idx1}")
                continue

            top = sorted(matches, key=lambda m: m.distance)[: args.max_matches]
            vis = cv2.drawMatches(
                u0,
                kp0,
                u1,
                kp1,
                top,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )

            fname = f"{method}_match_{c}_{idx0}_{idx1}.png"
            cv2.imwrite(os.path.join(out_sub, fname), vis)

    print("done")


if __name__ == "__main__":
    main()
