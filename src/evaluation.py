import numpy as np


def reprojection_error(kp0, kp1, inlier_matches, total_matches, R, t, calib):
    """
    Compute reprojection statistics and true inlier ratio.

    Args:
        kp0, kp1: lists of cv2.KeyPoint for left/right images.
        inlier_matches: list of cv2.DMatch objects classified as inliers.
        total_matches: int, total number of putative matches before RANSAC.
        R: 3×3 rotation matrix.
        t: 3×1 translation vector (already scaled by baseline).
        calib: dict with 'K0' and 'K1' intrinsic matrices.

    Returns:
        dict with:
          - mean_error   (float): average reprojection error in pixels.
          - median_error (float): median reprojection error in pixels.
          - inlier_ratio (float): (#inlier_matches) / (total_matches).
    """
    # no putative matches
    if total_matches == 0:
        return {"mean_error": 0.0, "median_error": 0.0, "inlier_ratio": 0.0}

    K0_inv = np.linalg.inv(calib["K0"])
    K1 = calib["K1"]
    errors = []

    for m in inlier_matches:
        u0, v0 = kp0[m.queryIdx].pt
        u1_gt, v1_gt = kp1[m.trainIdx].pt

        # Backproject pixel to camera 0 normalized coordinates
        X0 = K0_inv @ np.array([u0, v0, 1.0])
        # Transform into camera 1 coordinate frame
        X1 = R @ X0 + t.flatten()
        # Project into image plane of camera 1
        proj = K1 @ (X1 / X1[2])
        u1p, v1p = proj[0], proj[1]

        # Euclidean reprojection error
        errors.append(np.hypot(u1p - u1_gt, v1p - v1_gt))

    errs = np.array(errors)
    mean_error = float(errs.mean()) if errs.size > 0 else 0.0
    median_error = float(np.median(errs)) if errs.size > 0 else 0.0
    inlier_ratio = len(inlier_matches) / total_matches

    return {
        "mean_error": mean_error,
        "median_error": median_error,
        "inlier_ratio": inlier_ratio,
    }
