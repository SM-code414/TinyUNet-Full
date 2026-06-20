# visualization.py — Manuscript-quality BraTS Meningioma overlay & GIF
#
# BraTS Meningioma label encoding:
#   0 = Background
#   1 = NETC  Non-Enhancing Tumour Core  → RED    visible on T1ce
#   2 = SNFH  Surrounding Non-enhancing FLAIR Hyperintensity → GREEN  visible on FLAIR
#   3 = ET    Enhancing Tumour           → BLUE   visible on T1ce
#
# Hierarchical evaluation regions:
#   ET  = label 3               (blue,  T1ce background)
#   TC  = labels 1+3            (red+blue shown via layered rendering, T1ce bg)
#   WT  = labels 1+2+3          (all colours, outermost on FLAIR)
#
# Rendering order: SNFH (green) → NETC (red) → ET (blue)
# Each inner region is painted on top so all three are visible simultaneously.

# ── Backend must be set BEFORE any other matplotlib import ───────────────────
# BUG FIXED (was): matplotlib.use("Agg") was called after import matplotlib,
# which meant it was silently ignored if matplotlib was already imported
# elsewhere, causing buffer_rgba() to potentially fail mid-GIF.
import matplotlib
matplotlib.use("Agg")   # ← must be first matplotlib call

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines
from matplotlib.gridspec import GridSpec
import imageio
import torch
from tqdm import tqdm
import cv2


# ================================================================
# BraTS Meningioma colour scheme  (per paper Fig. 2)
#
# BUG FIXED (was): COLOR_MAP used glioma label ordering
#   old index 1 = NCR → green   (wrong: NETC should be RED)
#   old index 2 = ED  → green   (wrong: SNFH should be GREEN — coincidentally correct)
#   old index 3 = ET  → blue    (correct)
#
# COLOR_MAP is now only used for internal reference / legacy callers.
# All rendering uses LABEL_COLOR and REGION_COLOR below.
# ================================================================

# Per raw-label colours  (index = label value)
LABEL_COLOR = {
    0: np.array([0,   0,   0  ], dtype=np.uint8),   # Background (not drawn)
    1: np.array([255, 0,   0  ], dtype=np.uint8),   # NETC → RED
    2: np.array([0,   204, 0  ], dtype=np.uint8),   # SNFH → GREEN
    3: np.array([0,   102, 255], dtype=np.uint8),   # ET   → BLUE
}

# Keep COLOR_MAP for any legacy code that references it by index
COLOR_MAP = np.array([
    LABEL_COLOR[0],
    LABEL_COLOR[1],   # 1 = NETC → RED
    LABEL_COLOR[2],   # 2 = SNFH → GREEN
    LABEL_COLOR[3],   # 3 = ET   → BLUE
], dtype=np.uint8)

# Hierarchical region colours  (what gets drawn in compartment overlays)
REGION_COLOR = {
    "ET":   LABEL_COLOR[3],   # blue
    "NETC": LABEL_COLOR[1],   # red   (used when drawing TC layer)
    "SNFH": LABEL_COLOR[2],   # green (used when drawing WT/SNFH layer)
    # Aliases used by legend code
    "TC":   LABEL_COLOR[1],   # TC is dominated by NETC colour
    "WT":   LABEL_COLOR[2],   # WT outer boundary is SNFH colour
}

# Boolean mask functions for each region
def _snfh_mask(lbl): return lbl == 2                        # SNFH only
def _netc_mask(lbl): return lbl == 1                        # NETC only
def _et_mask(lbl):   return lbl == 3                        # ET only
def _tc_mask(lbl):   return (lbl == 1) | (lbl == 3)        # TC = NETC + ET
def _wt_mask(lbl):   return lbl > 0                         # WT = all labels

# Used for layered rendering: paint outermost (SNFH) first, ET last on top
RENDER_ORDER = [
    ("SNFH", _snfh_mask, LABEL_COLOR[2]),   # green
    ("NETC", _netc_mask, LABEL_COLOR[1]),   # red
    ("ET",   _et_mask,   LABEL_COLOR[3]),   # blue
]

DIFF_COLOR_FP = np.array([255, 165, 0  ], dtype=np.uint8)   # orange
DIFF_COLOR_FN = np.array([255, 0,   255], dtype=np.uint8)   # magenta

# BraTS channel order
MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]
T1CE_CHANNEL   = 1    # ET and NETC are best seen on T1ce
FLAIR_CHANNEL  = 3    # SNFH is by definition a FLAIR hyperintensity

# Manuscript figure settings
OVERLAY_ALPHA = 0.42   # anatomy still visible through colour
DPI           = 600
FONT_SIZE     = 8


# ================================================================
# Internal utilities
# ================================================================

def _to_numpy(*arrays):
    """Convert any mix of torch.Tensor / np.ndarray to np.ndarray."""
    out = []
    for a in arrays:
        if isinstance(a, torch.Tensor):
            a = a.cpu().numpy()
        out.append(np.asarray(a))
    return out


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Linearly scale array to [0, 1]."""
    img = image.astype(np.float32)
    img = np.nan_to_num(img)
    img -= img.min()
    mx  = img.max()
    if mx > 0:
        img /= mx
    return img


def _make_rgb_from_gray(gray_2d: np.ndarray) -> np.ndarray:
    """(H,W) → (H,W,3) float [0,1]."""
    return np.stack([normalize_image(gray_2d)] * 3, axis=-1)


# ================================================================
# Overlay helpers
# ================================================================

def overlay_color_mask_on_gray(
    image_slice: np.ndarray,
    mask_slice:  np.ndarray,
    alpha: float = OVERLAY_ALPHA,
) -> np.ndarray:
    """
    Blend BraTS Meningioma compartments onto a greyscale MRI slice.

    Rendering order: SNFH (green) → NETC (red) → ET (blue)
    Each inner region is painted on top so all three colours are
    simultaneously visible.

    BUG FIXED (was):
        Comments and mask docstring still said NCR/ED (glioma terminology).
        Now correctly labelled NETC / SNFH.

    Parameters
    ----------
    image_slice : (H,W) float/uint — single MRI channel
    mask_slice  : (H,W) int       — raw label map (0=BG, 1=NETC, 2=SNFH, 3=ET)
    alpha       : float           — colour opacity [0,1]

    Returns
    -------
    (H,W,3) float [0,1]
    """
    canvas = _make_rgb_from_gray(image_slice).copy()

    for _, mask_fn, color in RENDER_ORDER:
        mask  = mask_fn(mask_slice)
        color_f = color / 255.0
        canvas[mask] = (1 - alpha) * canvas[mask] + alpha * color_f

    return np.clip(canvas, 0, 1)


def overlay_diff_map_on_gray(
    image_slice: np.ndarray,
    gt_slice:    np.ndarray,
    pred_slice:  np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Overlay global FP (orange) and FN (magenta) on greyscale background.

    Note: This map shows errors across ALL tumour voxels combined.
    It does not distinguish which compartment the error belongs to.
    The panel title explicitly says 'Global FP/FN' to avoid implying
    per-compartment error.
    """
    base    = _make_rgb_from_gray(image_slice)
    diff    = np.zeros_like(base)
    fp_mask = (pred_slice > 0) & (gt_slice == 0)
    fn_mask = (gt_slice   > 0) & (pred_slice == 0)

    diff[fp_mask] = DIFF_COLOR_FP / 255.0
    diff[fn_mask] = DIFF_COLOR_FN / 255.0

    has_diff = (fp_mask | fn_mask)[..., None]
    blended  = np.where(has_diff, (1 - alpha) * base + alpha * diff, base)
    return np.clip(blended, 0, 1)


# ================================================================
# Legends
# ================================================================

def _add_inline_legend(ax, show_diff: bool = True) -> None:
    """
    Embed a compact colour legend directly in an Axes panel.

    BUG FIXED (was):
        Labels said 'ET (blue)', 'TC (red)', 'WT (green)' — wrong
        terminology for meningioma and redundant colour names.
        Now: 'ET', 'NETC', 'SNFH', 'FP', 'FN'.
    """
    patches = [
        mpatches.Patch(color=LABEL_COLOR[3] / 255.0, label="ET"),
        mpatches.Patch(color=LABEL_COLOR[1] / 255.0, label="NETC"),
        mpatches.Patch(color=LABEL_COLOR[2] / 255.0, label="SNFH"),
    ]
    if show_diff:
        patches += [
            mpatches.Patch(color=DIFF_COLOR_FP / 255.0, label="FP"),
            mpatches.Patch(color=DIFF_COLOR_FN / 255.0, label="FN"),
        ]
    ax.legend(
        handles    = patches,
        loc        = "lower center",
        ncol       = len(patches),
        fontsize   = FONT_SIZE - 1,
        framealpha = 0.75,
        edgecolor  = "none",
        handlelength = 1.0,
        handleheight = 0.8,
        borderpad  = 0.4,
    )


def make_legend_frame_png(save_dir: str, name: str = "legend.png") -> str:
    """
    Save a standalone legend PNG for use as the first GIF frame.

    BUG FIXED (was):
        Used COLOR_MAP[1], COLOR_MAP[2], COLOR_MAP[3] which had wrong
        colours (both 1 and 2 were green).
        Text still said 'NCR+ET' and 'WT, all' (glioma terminology).
        Now uses LABEL_COLOR with correct meningioma names.
    """
    os.makedirs(save_dir, exist_ok=True)

    entries = [
        ("ET  — Enhancing Tumour (T1ce)",             LABEL_COLOR[3]),
        ("NETC — Non-Enhancing Tumour Core (T1ce)",   LABEL_COLOR[1]),
        ("SNFH — Surrounding FLAIR Hyperintensity",   LABEL_COLOR[2]),
        ("FP  — False Positive",                      DIFF_COLOR_FP),
        ("FN  — False Negative",                      DIFF_COLOR_FN),
    ]

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.03, 0.97, "Legend", fontsize=11, weight="bold", va="top")

    for i, (txt, col) in enumerate(entries):
        y = 0.80 - i * 0.16
        ax.add_patch(plt.Rectangle((0.03, y), 0.07, 0.11,
                                   color=col / 255.0, zorder=2))
        ax.text(0.13, y + 0.055, txt, fontsize=9, va="center")

    out = os.path.join(save_dir, name)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ================================================================
# Main: slice overlay  (manuscript quality)
# ================================================================

def save_slice_overlay(
    volume:          "np.ndarray | torch.Tensor",   # (C, D, H, W)
    gt:              "np.ndarray | torch.Tensor",   # (D, H, W)
    pred:            "np.ndarray | torch.Tensor",   # (D, H, W)
    patient_id:      str,
    save_dir:        str,
    slice_indices:   "list | range | None" = None,
    show_modalities: "list[int]" = [T1CE_CHANNEL, FLAIR_CHANNEL],
) -> None:
    """
    Save per-slice manuscript-quality PNG figures (600 DPI).

    Layout (left → right)
    ─────────────────────
      T1ce | FLAIR | GT overlay (T1ce bg) | Pred overlay (T1ce bg) | Global FP/FN

    Parameters
    ----------
    volume          : (C, D, H, W)
    gt              : (D, H, W) integer labels
    pred            : (D, H, W) integer labels
    patient_id      : str
    save_dir        : str
    slice_indices   : None = every slice with a non-zero label
    show_modalities : channel indices shown as raw greyscale panels
    """
    os.makedirs(save_dir, exist_ok=True)
    volume, gt, pred = _to_numpy(volume, gt, pred)

    if volume.ndim != 4:
        raise ValueError(f"volume must be (C,D,H,W), got {volume.shape}")
    if gt.ndim != 3 or pred.ndim != 3:
        raise ValueError(f"gt/pred must be (D,H,W), got {gt.shape}, {pred.shape}")

    C, D, H, W   = volume.shape
    show_mods    = [c for c in show_modalities if c < C]
    n_mod        = len(show_mods)
    n_cols       = n_mod + 3                     # modalities + GT + Pred + FP/FN
    col_w        = 2.8
    fig_w        = col_w * n_cols
    fig_h        = col_w * (H / W) + 0.6        # preserve voxel aspect ratio

    if slice_indices is None:
        slice_indices = [i for i in range(D)
                         if gt[i].sum() > 0 or pred[i].sum() > 0]

    print(f"[Overlay] {patient_id}  —  {len(slice_indices)} slices")

    for idx in tqdm(slice_indices, desc=f"Overlay {patient_id}"):
        fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
        fig.suptitle(
            f"{patient_id}   |   axial slice {idx}/{D-1}",
            fontsize=FONT_SIZE + 1, fontweight="bold", y=0.99,
        )

        gs  = GridSpec(1, n_cols, figure=fig,
                       left=0.01, right=0.99,
                       top=0.90,  bottom=0.10, wspace=0.04)
        axs = [fig.add_subplot(gs[0, c]) for c in range(n_cols)]

        # ── Raw modality panels ───────────────────────────────────────────
        for col, ch in enumerate(show_mods):
            axs[col].imshow(volume[ch, idx], cmap="gray",
                            interpolation="lanczos")
            axs[col].set_title(
                MODALITY_NAMES[ch] if ch < len(MODALITY_NAMES) else f"Ch{ch}",
                fontsize=FONT_SIZE)
            axs[col].axis("off")

        # Backgrounds: ET/NETC best on T1ce; SNFH on FLAIR
        # For combined overlay we use T1ce (shows enhancement + dark NETC regions)
        t1ce_img = volume[T1CE_CHANNEL  if T1CE_CHANNEL  < C else 0, idx]

        # ── GT overlay on T1ce ───────────────────────────────────────────
        axs[n_mod].imshow(overlay_color_mask_on_gray(t1ce_img, gt[idx]))
        axs[n_mod].set_title("GT (T1ce)", fontsize=FONT_SIZE)
        axs[n_mod].axis("off")

        # ── Pred overlay on T1ce ─────────────────────────────────────────
        axs[n_mod + 1].imshow(overlay_color_mask_on_gray(t1ce_img, pred[idx]))
        axs[n_mod + 1].set_title("Pred (T1ce)", fontsize=FONT_SIZE)
        axs[n_mod + 1].axis("off")

        # ── Global FP/FN on T1ce ─────────────────────────────────────────
        axs[n_mod + 2].imshow(
            overlay_diff_map_on_gray(t1ce_img, gt[idx], pred[idx]))
        axs[n_mod + 2].set_title("Global FP/FN", fontsize=FONT_SIZE)
        axs[n_mod + 2].axis("off")

        _add_inline_legend(axs[n_mod + 2], show_diff=True)

        # ── Separator line between raw modalities and overlays ────────────
        sep_x = n_mod / n_cols
        fig.add_artist(matplotlib.lines.Line2D(
            [sep_x, sep_x], [0.08, 0.92],
            transform=fig.transFigure,
            color="gray", lw=0.6, linestyle="--", alpha=0.5))

        fname = os.path.join(save_dir, f"{patient_id}_slice{idx:03d}.png")
        fig.savefig(fname, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(f"[Overlay] Done — saved to {save_dir}")


# ================================================================
# GIF
# ================================================================

def save_gif(
    volume:     "np.ndarray | torch.Tensor",
    gt:         "np.ndarray | torch.Tensor",
    pred:       "np.ndarray | torch.Tensor",
    patient_id: str,
    save_dir:   str,
    fps:        int = 6,
) -> None:
    """
    Save an animated GIF (GT | Pred | FP/FN) looping through axial slices.

    BUG FIXED (was):
        1. legend was re-read from disk on EVERY frame inside the loop →
           potential shape mismatches and wasted I/O.
           Legend is now read ONCE before the loop.
        2. Legend text used glioma terminology (NCR+ET, WT).
           Now uses meningioma terms (NETC, SNFH, ET).
        3. matplotlib.use("Agg") moved to top of file so it is always set
           before plt is used, preventing buffer_rgba() failures.
    """
    os.makedirs(save_dir, exist_ok=True)
    volume, gt, pred = _to_numpy(volume, gt, pred)

    C, D, H, W  = volume.shape
    t1ce_ch     = T1CE_CHANNEL if T1CE_CHANNEL < C else 0

    # ── Legend — read ONCE outside the loop ──────────────────────────────
    legend_path = os.path.join(save_dir, "legend.png")
    if not os.path.exists(legend_path):
        make_legend_frame_png(save_dir)

    legend_img = plt.imread(legend_path)
    if legend_img.dtype != np.uint8:
        legend_img = (np.clip(legend_img, 0, 1) * 255).astype(np.uint8)
    if legend_img.ndim == 3 and legend_img.shape[2] == 3:
        legend_img = np.concatenate(
            [legend_img, 255 * np.ones((*legend_img.shape[:2], 1),
                                       dtype=np.uint8)], axis=2)

    # ── Build frames ──────────────────────────────────────────────────────
    frames = []
    print(f"[GIF] Building frames for {patient_id}")

    for i in tqdm(range(D), desc=f"GIF {patient_id}"):
        bg       = volume[t1ce_ch, i]
        gt_rgb   = overlay_color_mask_on_gray(bg, gt[i])
        pred_rgb = overlay_color_mask_on_gray(bg, pred[i])
        diff_rgb = overlay_diff_map_on_gray(bg, gt[i], pred[i])

        fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
        fig.suptitle(f"{patient_id}  slice {i}/{D-1}",
                     fontsize=8, fontweight="bold", y=1.01)

        axes[0].imshow(gt_rgb);   axes[0].set_title("GT",          fontsize=8)
        axes[1].imshow(pred_rgb); axes[1].set_title("Pred",        fontsize=8)
        axes[2].imshow(diff_rgb); axes[2].set_title("Global FP/FN",fontsize=8)
        for ax in axes:
            ax.axis("off")

        _add_inline_legend(axes[2], show_diff=True)
        fig.tight_layout(pad=0.3)
        fig.canvas.draw()

        frame = np.asarray(fig.canvas.buffer_rgba()).astype(np.uint8)
        plt.close(fig)
        frames.append(frame)

    # ── Prepend legend frame ──────────────────────────────────────────────
    frames.insert(0, legend_img)

    # Resize all to smallest common size
    target_h = min(f.shape[0] for f in frames)
    target_w = min(f.shape[1] for f in frames)
    frames   = [cv2.resize(f, (target_w, target_h),
                           interpolation=cv2.INTER_AREA) for f in frames]

    gif_path = os.path.join(save_dir, f"{patient_id}.gif")
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"[GIF] Saved → {gif_path}")
