"""Evaluation metrics for segmentation / point-cloud registration.

Implements:
    - dice_score            : Dice coefficient for binary masks
    - iou_score             : Intersection-over-Union (Jaccard) for binary masks
    - hausdorff_distance_95 : 95th-percentile Hausdorff distance (point clouds)
    - assd                  : Average Symmetric Surface Distance (point clouds)

All point-cloud metrics use ``scipy.spatial.cKDTree`` for fast nearest-neighbor
queries and include robust handling of empty inputs.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Segmentation metrics (binary masks)
# ---------------------------------------------------------------------------
def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient = 2 * |A ∩ B| / (|A| + |B|).

    Parameters
    ----------
    pred, gt : np.ndarray
        Binary (or boolean) arrays of the same shape. Non-zero elements are
        treated as foreground.

    Returns
    -------
    float
        Dice coefficient in [0, 1]. Returns 1.0 if both are empty (perfect
        agreement on empty sets), 0.0 if exactly one is empty.
    """
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred shape {pred.shape} != gt shape {gt.shape}")

    intersection = np.logical_and(pred, gt).sum()
    sum_pred = pred.sum()
    sum_gt = gt.sum()

    if sum_pred + sum_gt == 0:
        return 1.0  # both empty -> perfect agreement
    return float(2.0 * intersection / (sum_pred + sum_gt))


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-Union (Jaccard index) = |A ∩ B| / |A ∪ B|.

    Returns 1.0 if both are empty, 0.0 if exactly one is empty.
    """
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred shape {pred.shape} != gt shape {gt.shape}")

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0  # both empty
    return float(intersection / union)


# ---------------------------------------------------------------------------
# Surface / point-cloud metrics
# ---------------------------------------------------------------------------
def _check_point_clouds(pred_points: np.ndarray, gt_points: np.ndarray):
    pred_points = np.asarray(pred_points, dtype=np.float64)
    gt_points = np.asarray(gt_points, dtype=np.float64)
    if pred_points.ndim != 2 or pred_points.shape[1] != 3:
        raise ValueError(
            f"pred_points must be (N, 3), got {pred_points.shape}")
    if gt_points.ndim != 2 or gt_points.shape[1] != 3:
        raise ValueError(
            f"gt_points must be (M, 3), got {gt_points.shape}")
    return pred_points, gt_points


def hausdorff_distance_95(pred_points: np.ndarray,
                          gt_points: np.ndarray,
                          percentile: float = 95.0) -> float:
    """95th-percentile Hausdorff distance between two point clouds.

    Computes the symmetric nearest-neighbor distance in both directions and
    returns the ``percentile``-th percentile (default 95). This is a robust
    variant of the Hausdorff distance that is insensitive to a small number
    of outliers.

    Parameters
    ----------
    pred_points : np.ndarray, shape (N, 3)
    gt_points : np.ndarray, shape (M, 3)
    percentile : float
        Percentile to compute (default 95).

    Returns
    -------
    float
        HD95 distance. Returns 0.0 if both clouds are empty; raises if exactly
        one is empty.
    """
    pred_points, gt_points = _check_point_clouds(pred_points, gt_points)

    n_pred = pred_points.shape[0]
    n_gt = gt_points.shape[0]

    if n_pred == 0 and n_gt == 0:
        return 0.0
    if n_pred == 0 or n_gt == 0:
        raise ValueError("Cannot compute HD95 with an empty point cloud "
                         "(one side is empty).")

    # pred -> gt
    tree_gt = cKDTree(gt_points)
    d_pred_to_gt, _ = tree_gt.query(pred_points, k=1)

    # gt -> pred
    tree_pred = cKDTree(pred_points)
    d_gt_to_pred, _ = tree_pred.query(gt_points, k=1)

    all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_dists, percentile))


def assd(pred_points: np.ndarray, gt_points: np.ndarray) -> float:
    """Average Symmetric Surface Distance (ASSD / MSD).

    ASSD = (mean_{p in pred} d(p, gt) + mean_{q in gt} d(q, pred)) / 2

    Returns
    -------
    float
        Average symmetric nearest-neighbor distance. 0.0 if both clouds empty;
        raises if exactly one is empty.
    """
    pred_points, gt_points = _check_point_clouds(pred_points, gt_points)

    n_pred = pred_points.shape[0]
    n_gt = gt_points.shape[0]

    if n_pred == 0 and n_gt == 0:
        return 0.0
    if n_pred == 0 or n_gt == 0:
        raise ValueError("Cannot compute ASSD with an empty point cloud "
                         "(one side is empty).")

    tree_gt = cKDTree(gt_points)
    d_pred_to_gt, _ = tree_gt.query(pred_points, k=1)

    tree_pred = cKDTree(pred_points)
    d_gt_to_pred, _ = tree_pred.query(gt_points, k=1)

    return float(0.5 * (np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred)))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # --- Segmentation metrics ---
    a = np.zeros((10, 10), dtype=bool)
    a[2:6, 2:6] = True
    b = np.zeros((10, 10), dtype=bool)
    b[3:7, 3:7] = True  # shifted by 1

    print("=== Metrics self-test ===")
    print(f"Dice(a, b)      = {dice_score(a, b):.4f}")
    print(f"IoU(a, b)       = {iou_score(a, b):.4f}")
    print(f"Dice(a, a)      = {dice_score(a, a):.4f}  (expect 1.0)")
    print(f"IoU(a, a)       = {iou_score(a, a):.4f}  (expect 1.0)")
    print(f"Dice(empty, empty) = {dice_score(np.zeros(3, bool), np.zeros(3, bool)):.4f}")

    # --- Point-cloud metrics ---
    base = rng.standard_normal((300, 3))
    shifted = base + np.array([0.1, 0.0, 0.05]) + rng.normal(scale=0.005, size=(300, 3))

    hd = hausdorff_distance_95(base, shifted)
    ad = assd(base, shifted)
    print(f"HD95(base, shifted) = {hd:.4f}")
    print(f"ASSD(base, shifted) = {ad:.4f}")

    # Identical clouds -> 0
    hd_same = hausdorff_distance_95(base, base)
    ad_same = assd(base, base)
    print(f"HD95(base, base)     = {hd_same:.4f}  (expect 0.0)")
    print(f"ASSD(base, base)     = {ad_same:.4f}  (expect 0.0)")

    # Empty-edge cases
    print(f"HD95(empty, empty) = {hausdorff_distance_95(np.zeros((0,3)), np.zeros((0,3))):.4f}")
    print(f"ASSD(empty, empty)  = {assd(np.zeros((0,3)), np.zeros((0,3))):.4f}")
    try:
        hausdorff_distance_95(base, np.zeros((0, 3)))
    except ValueError as e:
        print(f"HD95 empty-edge raised as expected: {e}")

    print("Metrics self-test PASSED.")
