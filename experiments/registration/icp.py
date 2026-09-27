"""Point-to-Point ICP (Iterative Closest Point) registration.

Pure NumPy implementation using cKDTree for nearest-neighbor search and
SVD (Kabsch/Umeyama algorithm) for the closed-form optimal rigid transform.

Convention: we seek a rigid transform T (R, t) such that
    target_i ~= R @ source_i + t
minimizing the mean squared point-to-point error over the current
correspondence set.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _best_fit_transform(src: np.ndarray, dst: np.ndarray):
    """Closed-form optimal rigid transform aligning src onto dst.

    Returns (R (3,3), t (3,), mse) minimizing ||R @ src_i + t - dst_i||^2.
    """
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_c = src - src_mean
    dst_c = dst - dst_mean

    # Cross-covariance matrix H = src_c^T @ dst_c
    H = src_c.T @ dst_c

    U, S, Vt = np.linalg.svd(H)

    # Rotation: R = V @ U^T  (Vt is V^T, so R = Vt.T @ U.T)
    R = Vt.T @ U.T

    # Correct for reflection (ensure proper rotation, det = +1)
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = dst_mean - R @ src_mean

    aligned = src @ R.T + t
    mse = np.mean(np.sum((aligned - dst) ** 2, axis=1))
    return R, t, mse


def icp_registration(source: np.ndarray,
                     target: np.ndarray,
                     max_iterations: int = 100,
                     tolerance: float = 1e-6):
    """Point-to-Point ICP registration.

    Parameters
    ----------
    source : np.ndarray, shape (N, 3)
        Source point cloud to be aligned.
    target : np.ndarray, shape (M, 3)
        Target point cloud (fixed reference).
    max_iterations : int
        Maximum number of ICP iterations.
    tolerance : float
        Convergence threshold on the absolute change of MSE between
        consecutive iterations.

    Returns
    -------
    aligned_source : np.ndarray, shape (N, 3)
        Transformed source point cloud.
    transformation : np.ndarray, shape (4, 4)
        Homogeneous transformation matrix such that
        ``aligned = (R @ source.T).T + t``, equivalently
        ``aligned_homo = transformation @ source_homo``.
    final_mse : float
        Final mean squared point-to-point error after convergence.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    assert source.ndim == 2 and source.shape[1] == 3, \
        f"source must be (N,3), got {source.shape}"
    assert target.ndim == 2 and target.shape[1] == 3, \
        f"target must be (M,3), got {target.shape}"

    # Accumulated transform: starts as identity
    R_total = np.eye(3)
    t_total = np.zeros(3)

    current = source.copy()
    prev_mse = np.inf

    for it in range(max_iterations):
        # 1. Nearest-neighbor correspondence
        tree = cKDTree(target)
        dists, idx = tree.query(current, k=1)
        matched_target = target[idx]

        # 2. Optimal rigid transform for current correspondences (Kabsch)
        R_step, t_step, mse = _best_fit_transform(current, matched_target)

        # 3. Update source
        current = current @ R_step.T + t_step

        # Accumulate: new_point = R_step @ old_point + t_step
        R_total = R_step @ R_total
        t_total = R_step @ t_total + t_step

        # 4. Convergence check
        if abs(prev_mse - mse) < tolerance:
            break
        prev_mse = mse

    # Build 4x4 homogeneous matrix
    transformation = np.eye(4)
    transformation[:3, :3] = R_total
    transformation[:3, 3] = t_total

    # Final MSE on the aligned cloud
    tree = cKDTree(target)
    final_dists, _ = tree.query(current, k=1)
    final_mse = float(np.mean(final_dists ** 2))

    return current, transformation, final_mse


if __name__ == "__main__":
    # ---- Quick sanity test with synthetic (asymmetric ellipsoid) data ----
    rng = np.random.default_rng(42)

    # Original point cloud: points on an asymmetric ellipsoid
    # (unequal axis lengths -> no rotational symmetry, so ICP can recover R)
    n = 800
    vec = rng.standard_normal((n, 3))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    source = vec * np.array([1.0, 0.6, 0.4])  # ellipsoid radii

    # Known rigid transform: 30 deg rotation about z + translation
    angle = np.deg2rad(30.0)
    c, s = np.cos(angle), np.sin(angle)
    R_true = np.array([[c, -s, 0.0],
                       [s, c, 0.0],
                       [0.0, 0.0, 1.0]])
    t_true = np.array([0.5, -0.3, 0.2])
    target = source @ R_true.T + t_true
    target += rng.normal(scale=0.005, size=target.shape)  # small noise

    # MSE before registration
    tree = cKDTree(target)
    pre_dists, _ = tree.query(source, k=1)
    pre_mse = float(np.mean(pre_dists ** 2))

    aligned, T, post_mse = icp_registration(source, target,
                                            max_iterations=200,
                                            tolerance=1e-9)

    print("=== ICP self-test ===")
    print(f"Pre-registration  MSE: {pre_mse:.6f}")
    print(f"Post-registration MSE: {post_mse:.6f}")
    print(f"Converged reduction : {pre_mse - post_mse:.6f}")
    print("Recovered R:\n", np.array2string(T[:3, :3], precision=4))
    print("True R:\n", np.array2string(R_true, precision=4))
    print("Recovered t:", np.array2string(T[:3, 3], precision=4))
    print("True t:", np.array2string(t_true, precision=4))
    assert post_mse < pre_mse, "ICP should reduce the error!"
    print("ICP self-test PASSED.")
