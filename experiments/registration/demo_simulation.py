"""End-to-end simulation validation: ICP registration + metric improvement.

Generates an asymmetric point cloud (ellipsoid), applies a known rigid
transform + noise to create the target, runs our pure-NumPy Point-to-Point
ICP, and reports how the point-cloud metrics (HD95, ASSD) improve.

Run from the project root:
    python -m experiments.registration.demo_simulation
or
    python experiments/registration/demo_simulation
"""

from __future__ import annotations

import sys
import os
import numpy as np

# Allow running as a plain script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from registration.icp import icp_registration          # noqa: E402
from metrics import (dice_score, iou_score,             # noqa: E402
                     hausdorff_distance_95, assd)


def build_ellipsoid(n=1000, radii=(1.0, 0.6, 0.4), seed=42):
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal((n, 3))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return vec * np.array(radii)


def main():
    # 1. Source cloud (asymmetric ellipsoid -> ICP can recover full pose)
    source = build_ellipsoid(n=1000)

    # 2. Known rigid transform + noise -> target
    rng = np.random.default_rng(123)
    angle = np.deg2rad(25.0)
    c, s = np.cos(angle), np.sin(angle)
    R_true = np.array([[c, -s, 0.0],
                       [s, c, 0.0],
                       [0.0, 0.0, 1.0]])
    t_true = np.array([0.4, -0.25, 0.15])
    target = source @ R_true.T + t_true
    target += rng.normal(scale=0.003, size=target.shape)

    print("=" * 60)
    print("ICP Registration + Metrics Simulation Demo")
    print("=" * 60)
    print(f"Source points: {source.shape}, Target points: {target.shape}")

    # --- Metrics BEFORE registration ---
    hd_pre = hausdorff_distance_95(source, target)
    ad_pre = assd(source, target)

    # --- Run ICP ---
    aligned, T, mse_post = icp_registration(source, target,
                                             max_iterations=200,
                                             tolerance=1e-9)

    # --- Metrics AFTER registration ---
    hd_post = hausdorff_distance_95(aligned, target)
    ad_post = assd(aligned, target)

    # Also demonstrate Dice / IoU on a voxelized occupancy grid
    def voxelize(pc, voxel=0.05, grid=64):
        """Bin a point cloud into a 3D occupancy grid (for Dice/IoU demo)."""
        mins = pc.min(axis=0) - 0.05
        shifted = pc - mins
        coords = np.clip((shifted / voxel).astype(int), 0, grid - 1)
        vol = np.zeros((grid, grid, grid), dtype=bool)
        vol[coords[:, 0], coords[:, 1], coords[:, 2]] = True
        return vol

    vol_src = voxelize(source)
    vol_tgt = voxelize(target)
    vol_alg = voxelize(aligned)

    dice_pre = dice_score(vol_src, vol_tgt)
    iou_pre = iou_score(vol_src, vol_tgt)
    dice_post = dice_score(vol_alg, vol_tgt)
    iou_post = iou_score(vol_alg, vol_tgt)

    print("\n--- Point-cloud metrics (HD95 / ASSD) ---")
    print(f"  HD95  before: {hd_pre:.6f}   after: {hd_post:.6f}   "
          f"improvement: {hd_pre - hd_post:+.6f}")
    print(f"  ASSD  before: {ad_pre:.6f}   after: {ad_post:.6f}   "
          f"improvement: {ad_pre - ad_post:+.6f}")

    print("\n--- Voxel-grid overlap metrics (Dice / IoU) ---")
    print(f"  Dice  before: {dice_pre:.4f}   after: {dice_post:.4f}   "
          f"improvement: {dice_post - dice_pre:+.4f}")
    print(f"  IoU   before: {iou_pre:.4f}   after: {iou_post:.4f}   "
          f"improvement: {iou_post - iou_pre:+.4f}")

    print("\n--- Recovered pose vs ground truth ---")
    print(f"  Recovered R:\n{T[:3, :3]}")
    print(f"  True R:\n{R_true}")
    print(f"  Recovered t: {T[:3, 3]}")
    print(f"  True t:      {t_true}")

    # --- Assertions ---
    assert hd_post < hd_pre, "HD95 should decrease after ICP"
    assert ad_post < ad_pre, "ASSD should decrease after ICP"
    assert dice_post >= dice_pre, "Dice should not decrease after ICP"
    assert iou_post >= iou_pre, "IoU should not decrease after ICP"

    print("\n" + "=" * 60)
    print("ALL SIMULATION CHECKS PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
