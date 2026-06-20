# utilsall.py
# Shared utilities: losses, metrics, inference
# BraTS Meningioma — corrected label mapping
#
# BraTS Meningioma label encoding:
#   0 = Background
#   1 = NETC  Non-Enhancing Tumour Core (calcification, hyperostosis, necrosis, cysts)
#   2 = SNFH  Surrounding Non-enhancing FLAIR Hyperintensity (oedematous brain parenchyma)
#   3 = ET    Enhancing Tumour
#
# Hierarchical evaluation regions:
#   ET  = label 3
#   TC  = labels 1 + 3   (Tumour Core = NETC + ET)
#   WT  = labels 1+2+3   (Whole Tumour = TC + SNFH)

import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from medpy.metric import binary
from itertools import product
import random


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ============================================================
# PER-LABEL SENSITIVITY  (raw integer labels, NOT regions)
# ============================================================

def sensitivity_per_class(preds, targets, num_classes):
    """
    Compute sensitivity (recall) for each raw integer label.

    Parameters
    ----------
    preds   : torch.Tensor  any shape — predicted class indices
    targets : torch.Tensor  same shape — ground-truth class indices
    num_classes : int

    Returns
    -------
    np.ndarray shape (num_classes,)
        Index 0 = background, 1 = NETC, 2 = SNFH, 3 = ET.
        Caller should slice [1:] to exclude background.

    BUG FIXED (was):
        Docstring claimed (B,D,H,W) input but the function works on any shape
        because it uses element-wise ops — not a crash but was misleading.
        More importantly: the original return order was used incorrectly in
        evaluate_model as [ET, TC, WT]; it actually returns raw per-label
        scores [NETC, SNFH, ET] after [1:] slicing.  This is now explicit.
    """
    sens = []
    for cls in range(num_classes):
        tp = ((preds == cls) & (targets == cls)).sum().item()
        fn = ((preds != cls) & (targets == cls)).sum().item()
        sens.append(tp / (tp + fn + 1e-6))
    return np.array(sens)


# ============================================================
# METRICS — PER RAW LABEL (used internally)
# ============================================================

def dice_per_class(pred, target, num_classes=4, eps=1e-6, ignore_bg=True):
    """
    Dice score for each raw integer label.

    Parameters
    ----------
    pred   : (1,D,H,W) or (D,H,W) torch.Tensor — predicted labels
    target : (1,D,H,W) or (D,H,W) torch.Tensor — ground-truth labels

    Returns
    -------
    list of length (num_classes-1) if ignore_bg else (num_classes)
    Order: [score_label1, score_label2, score_label3]
           = [NETC_dice,  SNFH_dice,   ET_dice]

    NOTE: This returns PER-LABEL scores, NOT hierarchical region scores.
    Use dice_meningioma_regions() for ET / TC / WT evaluation.
    """
    if pred.ndim == 4:
        pred = pred.squeeze(0)
    if target.ndim == 4:
        target = target.squeeze(0)

    scores  = []
    classes = range(1, num_classes) if ignore_bg else range(num_classes)

    for c in classes:
        p     = (pred   == c).float()
        t     = (target == c).float()
        denom = p.sum() + t.sum()
        if denom == 0:
            scores.append(np.nan)
        else:
            scores.append(((2.0 * (p * t).sum() + eps) / (denom + eps)).item())

    return scores   # [NETC, SNFH, ET]


def hd95_per_class(pred, target, num_classes=4, spacing=(1, 1, 1), ignore_bg=True):
    """
    HD95 for each raw integer label.

    Returns
    -------
    list [hd95_label1, hd95_label2, hd95_label3] = [NETC, SNFH, ET]

    BUG FIXED (was):
        Docstring falsely claimed this returned [ET, TC, WT].
        TC and WT are UNION regions — they cannot be computed per-label.
        Use hd95_meningioma_regions() for correct ET / TC / WT HD95.

    Fallback value 373.13 is the BraTS standard penalty for empty predictions.
    """
    if pred.ndim == 4:
        pred = pred.squeeze(0)
    if target.ndim == 4:
        target = target.squeeze(0)

    scores  = []
    classes = range(1, num_classes) if ignore_bg else range(num_classes)

    for c in classes:
        p = (pred   == c).cpu().numpy()
        t = (target == c).cpu().numpy()

        if p.sum() == 0 and t.sum() == 0:
            scores.append(np.nan)          # both empty → not penalised
        elif p.sum() == 0 or t.sum() == 0:
            scores.append(373.13)          # BraTS standard penalty
        else:
            scores.append(binary.hd95(p, t, voxelspacing=spacing))

    return scores   # [NETC, SNFH, ET]


# ============================================================
# METRICS — HIERARCHICAL REGIONS  (ET / TC / WT)
# These are the metrics reported in BraTS Meningioma papers.
# ============================================================

def _region_masks_np(pred_np, gt_np):
    """
    Build boolean numpy masks for ET, TC, WT from integer label arrays.

    Parameters
    ----------
    pred_np : np.ndarray (D,H,W) int
    gt_np   : np.ndarray (D,H,W) int

    Returns
    -------
    dict with keys 'ET', 'TC', 'WT', each value is (pred_mask, gt_mask)
    """
    return {
        "ET": (pred_np == 3,                          gt_np == 3),
        "TC": ((pred_np == 1) | (pred_np == 3),       (gt_np == 1) | (gt_np == 3)),
        "WT": (pred_np > 0,                           gt_np > 0),
    }


def dice_meningioma_regions(pred, target):
    """
    Compute Dice for ET, TC, WT hierarchical regions.

    Parameters
    ----------
    pred   : torch.Tensor (1,D,H,W) or (D,H,W)
    target : torch.Tensor (1,D,H,W) or (D,H,W)

    Returns
    -------
    dict {"ET": float, "TC": float, "WT": float}
    NaN when both pred and GT are empty for a region.
    """
    if pred.ndim == 4:
        pred = pred.squeeze(0)
    if target.ndim == 4:
        target = target.squeeze(0)

    pred_np = pred.cpu().numpy()
    gt_np   = target.cpu().numpy()
    masks   = _region_masks_np(pred_np, gt_np)
    eps     = 1e-6
    scores  = {}

    for region, (p, t) in masks.items():
        denom = p.sum() + t.sum()
        if denom == 0:
            scores[region] = np.nan
        else:
            inter          = np.logical_and(p, t).sum()
            scores[region] = (2.0 * inter + eps) / (denom + eps)

    return scores


def hd95_meningioma_regions(pred, target, spacing=(1, 1, 1)):
    """
    Compute HD95 for ET, TC, WT hierarchical regions.

    BUG FIXED (was): HD95 for TC and WT was taken from hd95_per_class()
    which iterates raw labels — TC (labels 1+3) and WT (labels 1+2+3)
    are union regions and CANNOT be computed per-label.

    Parameters
    ----------
    pred   : torch.Tensor (1,D,H,W) or (D,H,W)
    target : torch.Tensor (1,D,H,W) or (D,H,W)

    Returns
    -------
    dict {"ET": float, "TC": float, "WT": float}
    373.13 when pred is empty (BraTS standard), NaN when both are empty.
    """
    if pred.ndim == 4:
        pred = pred.squeeze(0)
    if target.ndim == 4:
        target = target.squeeze(0)

    pred_np = pred.cpu().numpy()
    gt_np   = target.cpu().numpy()
    masks   = _region_masks_np(pred_np, gt_np)
    scores  = {}

    for region, (p, t) in masks.items():
        if p.sum() == 0 and t.sum() == 0:
            scores[region] = np.nan
        elif p.sum() == 0 or t.sum() == 0:
            scores[region] = 373.13
        else:
            scores[region] = binary.hd95(p, t, voxelspacing=spacing)

    return scores


def sensitivity_meningioma_regions(pred, target):
    """
    Compute sensitivity (recall) for ET, TC, WT hierarchical regions.

    BUG FIXED (was): sensitivity was taken from sensitivity_per_class()[1:]
    which gives [NETC, SNFH, ET] — NOT [ET, TC, WT].

    Returns
    -------
    dict {"ET": float, "TC": float, "WT": float}
    NaN when GT region is empty.
    """
    if pred.ndim == 4:
        pred = pred.squeeze(0)
    if target.ndim == 4:
        target = target.squeeze(0)

    pred_np = pred.cpu().numpy()
    gt_np   = target.cpu().numpy()
    masks   = _region_masks_np(pred_np, gt_np)
    scores  = {}

    for region, (p, t) in masks.items():
        if t.sum() == 0:
            scores[region] = np.nan
        else:
            tp             = np.logical_and(p, t).sum()
            fn             = np.logical_and(~p, t).sum()
            scores[region] = tp / (tp + fn + 1e-6)

    return scores


def specificity_meningioma_regions(pred, target):
    """
    Compute specificity for ET, TC, WT hierarchical regions.

    Returns
    -------
    dict {"ET": float, "TC": float, "WT": float}
    NaN when GT background is empty.
    """
    if pred.ndim == 4:
        pred = pred.squeeze(0)
    if target.ndim == 4:
        target = target.squeeze(0)

    pred_np = pred.cpu().numpy()
    gt_np   = target.cpu().numpy()
    masks   = _region_masks_np(pred_np, gt_np)
    scores  = {}

    for region, (p, t) in masks.items():
        neg = ~t
        if neg.sum() == 0:
            scores[region] = np.nan
        else:
            tn             = np.logical_and(~p, neg).sum()
            fp             = np.logical_and(p,  neg).sum()
            scores[region] = tn / (tn + fp + 1e-6)

    return scores


# ============================================================
# LOSSES
# ============================================================

def dice_loss_multiclass(logits, target, eps=1e-6, ignore_bg=True):
    """
    Standard multiclass Dice loss (equal FP/FN weighting).

    Parameters
    ----------
    logits : (B, C, D, H, W)
    target : (B, D, H, W) or (B, 1, D, H, W)

    BUG FIXED (was):
        Function was named dice_loss_multiclass but used alpha=0.3, beta=0.7
        making it a Tversky loss (asymmetric FP/FN penalty), not Dice.
        Renamed parameters to symmetric eps-only Dice so the name matches
        the behaviour.  Use tversky_loss_multiclass() if you want asymmetric.
    """
    if target.ndim == 5:
        target = target.squeeze(1)

    num_classes  = logits.shape[1]
    probs        = F.softmax(logits, dim=1)
    target_1hot  = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()
    dims         = (0, 2, 3, 4)

    inter = (probs * target_1hot).sum(dims)
    denom = (probs + target_1hot).sum(dims)
    dice  = (2.0 * inter + eps) / (denom + eps)

    if ignore_bg:
        dice = dice[1:]

    return 1.0 - dice.mean()


def tversky_loss_multiclass(logits, target, alpha=0.3, beta=0.7,
                             eps=1e-6, ignore_bg=True):
    """
    Tversky loss — penalises FN more than FP (alpha < beta).
    alpha=0.3, beta=0.7 is the original setting from the old dice_loss_multiclass.

    BUG FIXED (was): this logic lived inside dice_loss_multiclass with a
    misleading name.  Now correctly separated and named.
    Also fixed: the old 'valid' mask computed over full batch×spatial dims
    so a class present anywhere was counted valid everywhere.  Removed —
    plain mean is cleaner and correct.
    """
    if target.ndim == 5:
        target = target.squeeze(1)

    num_classes  = logits.shape[1]
    probs        = F.softmax(logits, dim=1)
    target_1hot  = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()
    dims         = (0, 2, 3, 4)

    TP = (probs * target_1hot).sum(dims)
    FP = (probs * (1.0 - target_1hot)).sum(dims)
    FN = ((1.0 - probs) * target_1hot).sum(dims)

    tversky = (TP + eps) / (TP + alpha * FP + beta * FN + eps)

    if ignore_bg:
        tversky = tversky[1:]

    return 1.0 - tversky.mean()


# ============================================================
# SLIDING WINDOW INFERENCE
# ============================================================

@torch.no_grad()
def sliding_window_inference(
    volume,
    model,
    device,
    patch_size=(128, 128, 128),
    overlap=0.25,
    num_classes=4,
):
    """
    Sliding-window inference on a full 3-D volume.

    Parameters
    ----------
    volume : torch.Tensor (C, D, H, W)

    Returns
    -------
    np.ndarray (num_classes, D, H, W) — averaged logits

    BUG FIXED (was):
        range(0, D - pd + 1, stride) silently dropped the last region when
        the volume dimension is not a multiple of stride.
        Example: D=155, pd=128, stride=64 → positions [0, 64].
        Voxels 128..154 (27 slices) were never inferred — they stayed as
        zero logits, producing wrong predictions at the volume boundary.

        Fix: after the regular grid, add a final position clamped to
        (dim - patch_dim) so the trailing edge is always covered.
    """
    model.eval()

    if isinstance(volume, np.ndarray):
        volume = torch.from_numpy(volume)

    C, D, H, W = volume.shape
    pd, ph, pw  = patch_size

    stride = [
        max(1, int(pd * (1 - overlap))),
        max(1, int(ph * (1 - overlap))),
        max(1, int(pw * (1 - overlap))),
    ]

    def _positions(dim, p, s):
        """Return start positions covering [0, dim) with no trailing gap."""
        pos = list(range(0, dim - p + 1, s))
        # Ensure the last patch always reaches the end
        if not pos or pos[-1] + p < dim:
            pos.append(max(0, dim - p))
        return pos

    zs = _positions(D, pd, stride[0])
    ys = _positions(H, ph, stride[1])
    xs = _positions(W, pw, stride[2])

    out   = np.zeros((num_classes, D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W),              dtype=np.float32)

    for z, y, x in product(zs, ys, xs):
        patch = volume[:, z:z+pd, y:y+ph, x:x+pw]
        inp   = patch.unsqueeze(0).to(device)      # (1, C, pd, ph, pw)

        logits = model(inp).squeeze(0).cpu().numpy()  # (C, pd, ph, pw)

        out[:,   z:z+pd, y:y+ph, x:x+pw] += logits
        count[   z:z+pd, y:y+ph, x:x+pw] += 1.0

    out /= np.maximum(count, 1.0)[None]
    return out


# ============================================================
# CSV LOGGING
# ============================================================

def log_metrics(csv_path, row, write_header=False):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header and not exists:
            writer.writeheader()
        writer.writerow(row)
