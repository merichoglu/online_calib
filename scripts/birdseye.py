# scripts/birdseye.py

import os
import sys
import cv2
import numpy as np
import argparse

# ─── ensure src on path ───────────────────────────────────────────────────────
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(proj_root, "src"))

from cb_data_loader import CheckerBoardLoader, euler_to_rot

def detect_corners(img, pattern):
    """
    Try the classic and (if that fails) the SB version of OpenCV's chessboard finder.
    Returns (found, corners).
    """
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(img, pattern, flags)
    if not found:
        found, corners = cv2.findChessboardCornersSB(img, pattern,
                                                     cv2.CALIB_CB_NORMALIZE_IMAGE)
    return found, corners

def compute_homography(K, R_wc, t_wc):
    """
    Build the 3×3 homography from ground-plane (Z=0) → image:
        H [X; Y; 1] ∼ K [ R_wc[:,0:2] | t_wc ]
    """
    return K @ np.hstack((R_wc[:, :2], t_wc))

def main():
    p = argparse.ArgumentParser(
        description="Bird’s-eye view stitching via pose-based warping"
    )
    p.add_argument("--base-dir",     default="data_cb",
                   help="path to CheckerBoard dataset")
    p.add_argument("--scale",        type=float, default=200.0,
                   help="pixels per meter")
    p.add_argument("--size",         type=float, default=5.0,
                   help="half-width of output (meters)")
    p.add_argument("--pattern-size", type=str,   default="10,7",
                   help="checkerboard inner corners cols,rows")
    p.add_argument("--out-path",     default="outputs/birdseye/bird_stitched.png",
                   help="where to save result")
    args = p.parse_args()

    # parse the CLI pattern-size → (cols, rows)
    cols, rows = map(int, args.pattern_size.split(","))
    board_pattern = (cols, rows)

    # load dataset + intrinsics
    loader = CheckerBoardLoader(base_dir=args.base_dir)
    K      = loader.calib["K0"]

    # compute the WORLD-CENTRE so that your boards sit in the middle of the canvas
    pts = []
    for key, pose in loader._poses.items():
        if key.startswith("left_") or key.startswith("right_"):
            x, y, _, _, _, _ = pose
            pts.append((x, y))
    xs, ys = zip(*pts)
    mean_x, mean_y = float(np.mean(xs)), float(np.mean(ys))

    # build our blank canvas (square)
    N = int(2 * args.size * args.scale)
    canvas = np.zeros((N, N), dtype=np.float64)
    count  = np.zeros((N, N), dtype=np.uint16)
    cx = cy = N // 2

    # a little CLAHE to boost contrast for the corner detector
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    total_warps = 0
    for fid, imgL, imgR in loader.image_pairs():
        idx = fid.split("_")[-1]  # e.g. "15" from "left_15"

        for cam_label, img in (("left", imgL), ("right", imgR)):
            # 1) equalize + detect
            img_eq = clahe.apply(img)
            found, corners = detect_corners(img_eq, board_pattern)
            if not found:
                continue

            # 2) refine corner positions
            term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(img_eq, corners, (11,11), (-1,-1), term)

            # 3) mask out the checkerboard region
            pts = corners.reshape(-1, 2).astype(np.float32)
            hull = cv2.convexHull(pts)
            mask0 = np.zeros_like(img, dtype=np.uint8)
            cv2.fillConvexPoly(mask0, hull.astype(np.int32), 1)

            # 4) lookup that camera's pose
            pose_key = f"{cam_label}_{idx}"
            if pose_key not in loader._poses:
                print(f"⚠️  no extrinsics for {pose_key}, skipping")
                continue
            px, py, pz, yaw, pitch, roll = loader._poses[pose_key]
            R_wc = euler_to_rot(yaw, pitch, roll)
            t_wc = np.array([px, py, pz], dtype=np.float64).reshape(3,1)

            # 5) build the homography and the canvas→world transform
            H_plane = compute_homography(K, R_wc, t_wc)
            # S maps canvas-pixels → world (meters), centred at (mean_x,mean_y)
            S = np.array([
                [1/args.scale,        0, mean_x - cx/args.scale],
                [0,        1/args.scale, mean_y - cy/args.scale],
                [0,                  0,                   1     ]
            ], dtype=np.float64)

            M = H_plane @ S

            # 6) warp mask & image
            warped_mask = cv2.warpPerspective(
                mask0, M, (N, N),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            ).astype(bool)

            warped_img = cv2.warpPerspective(
                img, M, (N, N),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )

            # 7) accumulate
            canvas[warped_mask] += warped_img[warped_mask]
            count [warped_mask] += 1
            total_warps += 1
            print(f"✔ warped {cam_label}_{idx}")

    if total_warps == 0:
        print("❌ no valid warps – check your --pattern-size or image quality")
        sys.exit(1)

    # normalize and save
    result = np.zeros_like(canvas, dtype=np.uint8)
    valid  = count > 0
    result[valid] = (canvas[valid] / count[valid]).astype(np.uint8)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    cv2.imwrite(args.out_path, result)
    print(f"✅ stitched {total_warps} views → {args.out_path}")

if __name__ == "__main__":
    main()
