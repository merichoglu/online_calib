import cv2
import numpy as np
import yaml
from scipy.linalg import svd
import cvxpy as cp


def certify_essential(E_local, X0, X1, eps=1e-6):
    """
    SDP-based certification of essential matrix global optimality following §3.3 of the paper:
    - Builds the lifted variable X = [e;1][e;1]^T ∈ S^{10}
    - Minimizes <Cblock, X> s.t. X⪰0, X[9,9]==1
    - Compares primal value with local cost to compute duality gap
    Returns:
      is_certified (bool), dual_gap (float)
    """
    e_vec = E_local.flatten()
    v_local = np.hstack([e_vec, 1.0])  # length 10

    # Build the data matrix C = A^T A for epipolar constraints
    # X0, X1 are 3xN normalized points
    N = X0.shape[1]
    # construct A (N×9)
    A = np.zeros((N, 9))
    x0 = X0[:2, :].T
    x1 = X1[:2, :].T
    for i in range(N):
        u0, v0 = x0[i]
        u1, v1 = x1[i]
        A[i] = [u1 * u0, u1 * v0, u1, v1 * u0, v1 * v0, v1, u0, v0, 1]
    C = A.T @ A  # 9×9

    # Build block-diagonal Cblock = [[C, 0]; [0, 0]] (10×10)
    Cblock = np.zeros((10, 10))
    Cblock[:9, :9] = C

    # SDP variable
    X = cp.Variable((10, 10), PSD=True)
    # Objective: minimize trace(Cblock * X)
    obj = cp.Minimize(cp.trace(Cblock @ X))
    # Constraints: X[9,9] == 1
    constraints = [X[9, 9] == 1]
    # Solve
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    # Primal value
    f_global = prob.value
    # Local cost f_local = v_local^T Cblock v_local
    f_local = float(v_local @ (Cblock @ v_local))
    dual_gap = abs(f_local - f_global)

    is_certified = dual_gap <= eps
    return is_certified, dual_gap


class PoseEstimator:
    def __init__(self, config_path="../configs/default.yaml"):
        cfg = yaml.safe_load(open(config_path, "r"))
        ransac_cfg = cfg["ransac"]
        self.prob = ransac_cfg["prob"]
        self.thresh = ransac_cfg["thresh"]
        self.max_iters = cfg.get("gnc", {}).get("max_iters", 10)
        self.sigma = cfg.get("gnc", {}).get("sigma", self.thresh)
        self.cert_eps = cfg.get("cert", {}).get("eps", 1e-6)

    def estimate(self, kp0, kp1, matches, calib):
        # 1) Matched points
        pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
        pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])
        if len(matches) < 5:
            return np.eye(3), np.zeros((3, 1)), np.zeros(len(matches), dtype=bool)

        # 2) Eight-point RANSAC for init
        E8, mask8 = cv2.findEssentialMat(
            pts0,
            pts1,
            calib["K0"],
            None,
            calib["K0"],
            None,
            method=cv2.RANSAC,
            prob=self.prob,
            threshold=self.thresh,
        )
        # ensure E8 is a 3x3 matrix (handle multiple or wrong-shaped returns)
        if E8 is not None and E8.shape != (3, 3):
            E8 = E8.flatten()[:9].reshape(3, 3)

        if E8 is None or mask8 is None:
            return np.eye(3), np.zeros((3, 1)), np.zeros(len(matches), dtype=bool)
        mask8 = mask8.ravel().astype(bool)

        # 3) Normalize points
        K = calib["K0"]
        x0 = cv2.undistortPoints(pts0.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        x1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        X0 = np.vstack([x0.T, np.ones((1, x0.shape[0]))])
        X1 = np.vstack([x1.T, np.ones((1, x1.shape[0]))])

        # 4) GNC loop with Welsch
        E_curr = E8
        for _ in range(self.max_iters):
            Ex = E_curr @ X0
            num = np.sum(X1 * Ex, axis=0)
            den = np.linalg.norm(Ex, axis=0) + 1e-12
            r = num / den
            w = np.exp(-0.5 * (r / self.sigma) ** 2)
            # weighted 8pt
            Np = len(r)
            A = np.zeros((Np, 9))
            for i in range(Np):
                u0, v0 = x0[i]
                u1, v1 = x1[i]
                A[i] = [u1 * u0, u1 * v0, u1, v1 * u0, v1 * v0, v1, u0, v0, 1]
            W = np.sqrt(w)
            Aw = W[:, None] * A
            U, S, Vt = svd(Aw)
            e = Vt.T[:, -1]
            E_next = e.reshape(3, 3)
            # enforce rank-2
            Ue, Se, Vte = svd(E_next)
            Se[2] = 0
            E_next = Ue @ np.diag(Se) @ Vte
            if np.linalg.norm(E_next - E_curr) < 1e-6:
                break
            E_curr = E_next

        E_refined = E_curr

        # 5) SDP certification
        is_cert, gap = certify_essential(E_refined, X0, X1, eps=self.cert_eps)
        if not is_cert:
            E_final = E8
            mask_final = mask8
        else:
            E_final = E_refined
            mask_final = mask8

        # 6) Recover pose
        _, R_est, t_unit, mask2 = cv2.recoverPose(
            E_final, pts0, pts1, cameraMatrix=K, mask=mask_final.astype(np.uint8)
        )
        mask_bool = mask2.ravel().astype(bool)

        # 7) Cheirality & sign
        P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        idxs = np.where(mask_bool)[0]
        if idxs.size:
            sel = idxs[: min(20, idxs.size)]
            pts0_s = pts0[sel].T
            pts1_s = pts1[sel].T
            P1p = K @ np.hstack([R_est, t_unit])
            P1n = K @ np.hstack([R_est, -t_unit])
            Xp = cv2.triangulatePoints(P0, P1p, pts0_s, pts1_s)
            Xn = cv2.triangulatePoints(P0, P1n, pts0_s, pts1_s)
            Xp = Xp[:3] / Xp[3]
            Xn = Xn[:3] / Xn[3]
            X1p = R_est @ Xp + t_unit
            X1n = R_est @ Xn - t_unit
            if np.sum((Xp[2] > 0) & (X1n[2] > 0)) > np.sum((Xp[2] > 0) & (X1p[2] > 0)):
                t_unit = -t_unit
        if np.dot(t_unit.ravel(), calib["t"].ravel()) < 0:
            t_unit = -t_unit

        # 8) Scale
        t_scaled = t_unit * np.linalg.norm(calib["t"])

        return R_est, t_scaled, mask_bool
