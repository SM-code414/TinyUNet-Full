"""
manuscript_grid.py
──────────────────
Produces a publication-quality model-comparison grid figure.

Layout
──────
  Rows    : patients (3–5, each having ET + NETC + SNFH present)
  Columns : GT | TinyUNet | UNet | ResUNet | AttUNet
  Content : best axial slice (max tumour area) on T1ce background
            with colour overlay:  ET=blue  NETC=red  SNFH=green
  Extras  : per-cell Dice annotation, row/col labels, shared legend,
            600 DPI PNG ready for journal submission

Usage (from notebook)
──────────────────────
    from manuscript_grid import save_manuscript_grid

    save_manuscript_grid(
        all_predictions = all_predictions,  # list of dicts from main loop
        final_df        = final_df,         # per-patient metrics DataFrame
        test_dataset    = test_dataset,
        model_order     = ["tinyunet", "unet", "resunet", "attentionunet"],
        save_dir        = SAVE_DIR,
        n_patients      = 5,                # 3–5
    )
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ── BraTS Meningioma colour scheme ───────────────────────────────────────────
_ET   = np.array([0,   102, 255], dtype=np.uint8)   # blue
_NETC = np.array([255, 0,   0  ], dtype=np.uint8)   # red
_SNFH = np.array([0,   204, 0  ], dtype=np.uint8)   # green

OVERLAY_ALPHA = 0.45
T1CE_CH       = 1      # channel index in (C,D,H,W) volume
DPI           = 600
CELL_INCHES   = 2.5    # width & height of each thumbnail cell


def _norm(arr):
    """Linearly normalise array to [0,1]."""
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    return arr


def _overlay(t1ce_slice, label_slice):
    """
    Blend ET / NETC / SNFH compartments onto a T1ce greyscale slice.
    Render order: SNFH → NETC → ET  (inner regions on top).

    Parameters
    ----------
    t1ce_slice  : (H, W) float
    label_slice : (H, W) int  — 0=BG 1=NETC 2=SNFH 3=ET

    Returns
    -------
    (H, W, 3) float [0,1]
    """
    base   = np.stack([_norm(t1ce_slice)] * 3, axis=-1)
    canvas = base.copy()

    for mask_fn, colour in [
        (lambda x: x == 2, _SNFH),   # SNFH first (outermost)
        (lambda x: x == 1, _NETC),   # NETC
        (lambda x: x == 3, _ET ),    # ET last (innermost, on top)
    ]:
        m = mask_fn(label_slice)
        c = colour / 255.0
        canvas[m] = (1 - OVERLAY_ALPHA) * canvas[m] + OVERLAY_ALPHA * c

    return np.clip(canvas, 0, 1)


def _best_slice(label_vol):
    """
    Return the axial index with the most labelled voxels.
    Requires all three compartments to be present in the slice.
    Falls back to max-area slice if no single slice has all three.
    """
    D = label_vol.shape[0]
    # prefer slices that contain all three labels
    all_three = [
        i for i in range(D)
        if (label_vol[i] == 1).any()
        and (label_vol[i] == 2).any()
        and (label_vol[i] == 3).any()
    ]
    candidates = all_three if all_three else list(range(D))
    areas = [(label_vol[i] > 0).sum() for i in candidates]
    return candidates[int(np.argmax(areas))]


def _select_patients(all_predictions, final_df, model_order, n_patients):
    """
    Pick n_patients that:
      1. Have predictions from ALL models
      2. Have GT containing ET + NETC + SNFH (labels 1, 2, 3)
      3. Are ranked by mean WT Dice across all models (best cases first)

    Returns list of patient_id strings.
    """
    # Patients that appear in all_predictions for every model
    from collections import defaultdict
    pid_models = defaultdict(set)
    for item in all_predictions:
        pid_models[item["patient_id"]].add(item["model"])

    full_coverage = {
        pid for pid, models in pid_models.items()
        if set(model_order).issubset(models)
    }

    # Among those, keep only patients where GT has all three compartments
    all_compartments = set()
    for item in all_predictions:
        if item["patient_id"] not in full_coverage:
            continue
        gt = item["gt"]
        if (gt == 1).any() and (gt == 2).any() and (gt == 3).any():
            all_compartments.add(item["patient_id"])

    if not all_compartments:
        # fallback: any patient with at least ET present
        for item in all_predictions:
            if item["patient_id"] in full_coverage:
                gt = item["gt"]
                if (gt == 3).any():
                    all_compartments.add(item["patient_id"])

    # Rank by mean WT Dice across all models (descending)
    wt_mean = (
        final_df[final_df["Patient_ID"].isin(all_compartments)]
        .groupby("Patient_ID")["WT_Dice"]
        .mean()
        .sort_values(ascending=False)
    )

    selected = list(wt_mean.index[:n_patients])
    print(f"[Grid] Selected {len(selected)} patients:")
    for pid in selected:
        score = wt_mean[pid]
        labels = set()
        for item in all_predictions:
            if item["patient_id"] == pid:
                labels.update(np.unique(item["gt"]).tolist())
                break
        print(f"  {pid}  mean_WT_Dice={score:.4f}  GT labels={sorted(labels)}")

    return selected


def save_manuscript_grid(
    all_predictions,
    final_df,
    test_dataset,
    model_order,
    save_dir,
    n_patients     = 5,
    force_patients = None,   # list of patient IDs to use instead of auto-select
    filename       = "manuscript_grid.png",
):
    """
    Save the model-comparison grid figure.

    Parameters
    ----------
    all_predictions : list of dicts  — from main evaluation loop
        Each dict must have keys: model, patient_id, volume (C,D,H,W),
        gt (D,H,W), prediction (D,H,W)
    final_df        : pd.DataFrame   — per-patient metrics
    test_dataset    : BraTSDataset   — for patient list
    model_order     : list[str]      — model names in display order
    save_dir        : str
    n_patients      : int            — 3–5 recommended
    force_patients  : list[str]|None — override auto-selection
    filename        : str
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Select patients ───────────────────────────────────────────────────────
    if force_patients:
        patients = force_patients
        print(f"[Grid] Using {len(patients)} forced patients: {patients}")
    else:
        patients = _select_patients(
            all_predictions, final_df, model_order, n_patients
        )

    n_rows = len(patients)
    n_cols = 1 + len(model_order)   # GT + one per model

    col_labels = ["GT"] + [m.replace("attentionunet", "Att-UNet")
                            .replace("tinyunet",      "TinyUNet")
                            .replace("resunet",       "ResUNet")
                            .replace("unet",          "UNet")
                           for m in model_order]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig_w = CELL_INCHES * n_cols
    fig_h = CELL_INCHES * n_rows + 0.7   # extra for col headers + legend

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)

    # GridSpec: header row (thin) + n_rows data rows + legend row (thin)
    gs = gridspec.GridSpec(
        n_rows + 2, n_cols,
        figure    = fig,
        hspace    = 0.04,
        wspace    = 0.02,
        top       = 0.96,
        bottom    = 0.06,
        left      = 0.06,
        right      = 0.99,
        height_ratios = [0.25] + [1.0] * n_rows + [0.18],
    )

    # ── Column headers ────────────────────────────────────────────────────────
    for col, label in enumerate(col_labels):
        ax = fig.add_subplot(gs[0, col])
        ax.text(0.5, 0.5, label,
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                transform=ax.transAxes)
        ax.axis("off")
        # Underline
        ax.axhline(0, color="#333333", linewidth=0.8)

    # ── Build lookup: (patient_id, model) → item dict ────────────────────────
    lookup = {}
    for item in all_predictions:
        lookup[(item["patient_id"], item["model"])] = item

    # ── Data rows ─────────────────────────────────────────────────────────────
    for row, pid in enumerate(patients):
        # Get GT from first available model prediction
        ref_item = None
        for m in model_order:
            if (pid, m) in lookup:
                ref_item = lookup[(pid, m)]
                break
        if ref_item is None:
            continue

        gt_vol  = ref_item["gt"]       # (D,H,W)
        vol     = ref_item["volume"]   # (C,D,H,W)
        sl_idx  = _best_slice(gt_vol)
        t1ce_sl = vol[T1CE_CH, sl_idx]

        # ── Row label (patient ID, rotated) ───────────────────────────────
        ax_label = fig.add_subplot(gs[row + 1, 0])
        # We'll write the patient label as a y-axis label on the GT cell instead

        # ── GT column ─────────────────────────────────────────────────────
        ax_gt = fig.add_subplot(gs[row + 1, 0])
        ax_gt.imshow(_overlay(t1ce_sl, gt_vol[sl_idx]),
                     interpolation="lanczos", aspect="equal")
        ax_gt.set_ylabel(
            pid.replace("BraTS-MEN-", "").replace("-000", ""),
            fontsize=7, rotation=90, labelpad=3,
            va="center", ha="right",
        )
        ax_gt.set_xticks([]); ax_gt.set_yticks([])
        for spine in ax_gt.spines.values():
            spine.set_linewidth(0.4)

        # Slice index annotation
        ax_gt.text(0.02, 0.02, f"sl {sl_idx}",
                   transform=ax_gt.transAxes,
                   fontsize=5, color="white",
                   va="bottom", ha="left",
                   bbox=dict(facecolor="black", alpha=0.4,
                             pad=1, boxstyle="round,pad=0.15"))

        # ── Model columns ──────────────────────────────────────────────────
        for col, m in enumerate(model_order):
            ax = fig.add_subplot(gs[row + 1, col + 1])

            if (pid, m) in lookup:
                item     = lookup[(pid, m)]
                pred_sl  = item["prediction"][sl_idx]   # (H,W)
                overlay  = _overlay(t1ce_sl, pred_sl)

                ax.imshow(overlay, interpolation="lanczos", aspect="equal")

                # Dice score annotation (WT Dice for this patient + model)
                row_df = final_df[
                    (final_df["Patient_ID"] == pid) &
                    (final_df["Model"]      == m)
                ]
                if not row_df.empty:
                    et_d  = row_df["ET_Dice"].values[0]
                    tc_d  = row_df["TC_Dice"].values[0]
                    wt_d  = row_df["WT_Dice"].values[0]

                    def _fmt(v):
                        return f"{v:.2f}" if not np.isnan(v) else "—"

                    dice_txt = (f"ET {_fmt(et_d)}\n"
                                f"TC {_fmt(tc_d)}\n"
                                f"WT {_fmt(wt_d)}")
                    ax.text(0.98, 0.02, dice_txt,
                            transform=ax.transAxes,
                            fontsize=4.5, color="white",
                            va="bottom", ha="right",
                            linespacing=1.3,
                            bbox=dict(facecolor="black", alpha=0.5,
                                      pad=1, boxstyle="round,pad=0.15"))
            else:
                ax.text(0.5, 0.5, "N/A",
                        transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=8, color="gray")
                ax.set_facecolor("#111111")

            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)

    # ── Legend row ────────────────────────────────────────────────────────────
    ax_leg = fig.add_subplot(gs[n_rows + 1, :])
    ax_leg.axis("off")

    patches = [
        mpatches.Patch(color=_ET   / 255.0, label="ET — Enhancing Tumour"),
        mpatches.Patch(color=_NETC / 255.0, label="NETC — Non-Enhancing Tumour Core"),
        mpatches.Patch(color=_SNFH / 255.0, label="SNFH — Surrounding FLAIR Hyperintensity"),
    ]
    ax_leg.legend(
        handles     = patches,
        loc         = "center",
        ncol        = 3,
        fontsize    = 7,
        frameon     = False,
        handlelength= 1.2,
        handleheight= 0.9,
        columnspacing=1.5,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n[Grid] Saved → {out_path}")
    return out_path
