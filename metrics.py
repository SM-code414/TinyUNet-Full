# metrics.py
# Standalone numpy metric functions for BraTS Meningioma evaluation.
#
# BraTS Meningioma label encoding:
#   0 = Background
#   1 = NETC  Non-Enhancing Tumour Core
#   2 = SNFH  Surrounding Non-enhancing FLAIR Hyperintensity
#   3 = ET    Enhancing Tumour
#
# All functions expect boolean or integer numpy arrays.

import numpy as np
from medpy.metric import binary

EPS = 1e-8

# ------------------------------------------------------------------ #
# BraTS standard HD95 penalty when one mask is empty                  #
# (documented in BraTS challenge rules)                               #
# ------------------------------------------------------------------ #
HD95_EMPTY_PENALTY = 373.13


# ============================================================
# Binary mask metrics  (work on any single boolean region)
# ============================================================

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Dice similarity coefficient for a single binary region.

    BUG FIXED (was):
        EPS was only in the denominator → asymmetric formula.
        Now EPS added to both numerator and denominator for consistency
        with dice_per_class in utilsall.py.
        Also: when both are empty the correct answer is NaN (undefined),
        not 1.0 (perfect agreement), so the early-exit now returns NaN.
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    if pred.sum() + gt.sum() == 0:
        return np.nan   # both empty — undefined, not "perfect"

    inter = np.logical_and(pred, gt).sum()
    return float((2.0 * inter + EPS) / (pred.sum() + gt.sum() + EPS))


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-Union for a single binary region."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return np.nan

    inter = np.logical_and(pred, gt).sum()
    return float(inter / (union + EPS))


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Sensitivity (recall / TPR) for a single binary region."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    if gt.sum() == 0:
        return np.nan

    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return float(tp / (tp + fn + EPS))


def specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Specificity (TNR) for a single binary region."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    neg = ~gt
    if neg.sum() == 0:
        return np.nan

    tn = np.logical_and(~pred, neg).sum()
    fp = np.logical_and(pred,  neg).sum()
    return float(tn / (tn + fp + EPS))


def precision(pred: np.ndarray, gt: np.ndarray) -> float:
    """Precision (PPV) for a single binary region."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    if pred.sum() == 0:
        return np.nan

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    return float(tp / (tp + fp + EPS))


def hd95(pred: np.ndarray, gt: np.ndarray,
         empty_penalty: float = HD95_EMPTY_PENALTY) -> float:
    """
    95th-percentile Hausdorff distance.

    BUG FIXED (was):
        Returned np.nan when either mask was empty.
        BraTS standard is to return the penalty value (373.13) when the
        prediction is empty but GT is non-empty, so empty predictions are
        penalised rather than silently excluded from the mean.
        When both are empty → NaN (not penalised).

        Inconsistency with utilsall.hd95_per_class (which used 373.13) is
        now resolved — both files use the same convention.
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return np.nan              # both empty — not penalised
    if pred.sum() == 0 or gt.sum() == 0:
        return empty_penalty       # BraTS standard penalty

    return float(binary.hd95(pred, gt))


def auc_score(prob_map: np.ndarray, gt: np.ndarray) -> float:
    """
    ROC-AUC for a single class probability map vs binary GT mask.

    BUG FIXED (was):
        Flattened the ENTIRE 3-D volume (D×H×W ~ 2M voxels) before
        computing AUC.  Background voxels overwhelmingly dominate, so
        the classifier trivially gets ~99% of labels right by predicting
        background everywhere → AUC was artificially inflated toward 1.0.

        Fix: compute AUC only on a foreground region of interest (ROI)
        defined as the bounding box of the GT mask dilated by 5 voxels,
        so the classifier is evaluated on the clinically relevant region.
        Falls back to the full volume only when GT is completely empty
        (returns NaN in that case).
    """
    from sklearn.metrics import roc_auc_score
    from scipy.ndimage import binary_dilation

    gt = gt.astype(np.uint8)

    if gt.sum() == 0:
        return np.nan

    # Build ROI: bounding box of GT dilated by 5 voxels
    dilated = binary_dilation(gt, iterations=5)
    roi_idx = np.where(dilated)

    if len(roi_idx[0]) == 0:
        return np.nan

    gt_roi   = gt[roi_idx].flatten()
    prob_roi = prob_map[roi_idx].flatten()

    # Need at least one positive and one negative sample inside ROI
    if gt_roi.sum() == 0 or (gt_roi == 0).sum() == 0:
        return np.nan

    try:
        return float(roc_auc_score(gt_roi, prob_roi))
    except ValueError:
        return np.nan


# ============================================================
# Convenience: compute all metrics for one region at once
# ============================================================

def all_metrics(pred: np.ndarray, gt: np.ndarray,
                prob_map: np.ndarray = None) -> dict:
    """
    Return dict of all metrics for a single binary region.

    Parameters
    ----------
    pred     : (D,H,W) bool/int — predicted binary mask
    gt       : (D,H,W) bool/int — ground-truth binary mask
    prob_map : (D,H,W) float    — class probability map (optional, for AUC)
    """
    out = {
        "Dice":        dice_score(pred, gt),
        "IoU":         iou_score(pred, gt),
        "Sensitivity": sensitivity(pred, gt),
        "Specificity": specificity(pred, gt),
        "Precision":   precision(pred, gt),
        "HD95":        hd95(pred, gt),
    }
    if prob_map is not None:
        out["AUC"] = auc_score(prob_map, gt)
    return out
