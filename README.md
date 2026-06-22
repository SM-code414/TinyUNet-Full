# TinyUNet-Full: A Compact Boundary-Aware 3D Encoder-Decoder for Volumetric Meningioma Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-red.svg)](https://pytorch.org/)

Official implementation of **TinyUNet-Full**, as described in:

> **TinyUNet-Full: A Compact Boundary-Aware 3D Encoder-Decoder with Edge-Guided Dual Supervision for Volumetric Meningioma Segmentation**
> [Swarna Malini Johnson, Shenbagarajan Anantharajan] — *, 2026
> DOI: [to be added on acceptance]
> Email: swarnamalini@mepcoeng.ac.in; shenbagarajan@mepcoeng.ac.in
---

## Overview

TinyUNet-Full is a compact three-level 3D convolutional encoder-decoder for volumetric meningioma sub-compartment segmentation from multi-parametric MRI. It is evaluated on the [BraTS 2023 Intracranial Meningioma](https://www.synapse.org/#!Synapse:syn51156910) dataset.

The model integrates three components built on a standard U-Net backbone:

1. **3D Sobel edge fusion** — fixed-weight gradient magnitude maps concatenated at the input (0 learnable parameters)
2. **Deep supervision** — auxiliary segmentation heads on dec3 and dec2, active during training only
3. **Laplacian boundary loss** — 3D discrete Laplacian penalty on the main output (λ = 0.1)

Two configurations are provided:

| Configuration | Params | FLOPs | Mean DSC |
|---|---|---|---|
| TinyUNet-Full-B16 | 1.40 M | 132.0 G | 0.8799 |
| TinyUNet-Full-B32 | 5.61 M | 513.4 G | 0.8896 |

Both configurations outperform 3D U-Net (22.6 M), ResUNet (15.2 M), and Attention U-Net (22.7 M) on all three BraTS 2023 Meningioma sub-regions (ET, TC, WT).

---

## Repository Structure

```
TinyUNet-Full/
├── models/
│   ├── blocks.py                          # ConvGNReLU, DoubleConv3D
│   ├── tinyunet3d.py                      # TinyUNet3D baseline
│   ├── tinyunet_progressive_ablation.py   # A1–A4 ablation variants + TinyUNet_Full
│   ├── tinyunet_ablations.py              # DW / Ghost / SE variant exploration
│   ├── attentionunet3d.py                 # Attention U-Net baseline
│   ├── resunet3d.py                       # ResUNet baseline
│   └── unet3d.py                          # 3D U-Net baseline
├── losses/
│   └── boundary_loss.py                   # 3D Laplacian boundary loss
├── train.py                               # Training script
├── evaluate.py                            # Held-out test set evaluation
├── dataset.py                             # BraTSDataset + class-aware patch sampling
├── metrics.py                             # DSC, HD95, Sensitivity
├── registry.py                            # Model registry
├── utilsall.py                            # Sliding-window inference, seeding, utilities
├── precache.py                            # Optional: pre-cache normalised volumes
├── model_stats.py                         # Parameter count + FLOPs measurement
├── visualization.py                       # Slice overlays, GIF export
├── manuscript_grid.py                     # Figure grid generation (manuscript figures)
├── model_complexity.csv                   # Measured params/FLOPs for all models (Table 3)
├── notebooks/
│   └── tinyunetablation.ipynb             # Step-by-step ablation walkthrough
└── requirements.txt
```

---

## Requirements

```
Python >= 3.8
CUDA >= 12.4  (tested on NVIDIA L40S-16Q, 16 GB VRAM)
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `torch==2.4.0`, `monai`, `nibabel`, `medpy`, `scipy`, `pandas`, `tqdm`.

---

## Dataset

This repository does **not** include the BraTS 2023 dataset. Download it from Synapse:

- **Portal:** https://www.synapse.org/#!Synapse:syn51156910
- **Challenge page:** https://www.synapse.org/brats2023
- **Citation:** LaBella et al. (2023, 2024) — see paper references

After downloading, the expected directory layout is:

```
/path/to/brats2023/
└── ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData/
    ├── BraTS-MEN-00001-000/
    │   ├── BraTS-MEN-00001-000-t1c.nii.gz
    │   ├── BraTS-MEN-00001-000-t1n.nii.gz
    │   ├── BraTS-MEN-00001-000-t2f.nii.gz
    │   ├── BraTS-MEN-00001-000-t2w.nii.gz
    │   └── BraTS-MEN-00001-000-seg.nii.gz
    ├── BraTS-MEN-00002-000/
    └── ...
```

Create split files listing one case ID per line:

```bash
# Example: splits/train.txt, splits/val.txt, splits/test.txt
# Partition used in the paper: 800 / 100 / 100
```

---

## Training

All experiments use `seed=42`, `batch_size=2`, `patch_size=128³`, `max_epochs=250`, `early_stopping_patience=30`.

### Reproduce Table 2 — Comparative Evaluation

**TinyUNet-Full (B=16)** — primary result:
```bash
python train.py \
  --data_dir /path/to/brats2023 \
  --train_split splits/train.txt \
  --val_split splits/val.txt \
  --model tinyunet_full \
  --base_channels 16 \
  --batch_size 2 \
  --epochs 250 \
  --lr 2e-4 \
  --patience 30 \
  --out_dir outputs/tinyunet_full_b16
```

**TinyUNet-Full (B=32)**:
```bash
python train.py \
  --data_dir /path/to/brats2023 \
  --train_split splits/train.txt \
  --val_split splits/val.txt \
  --model tinyunet_full \
  --base_channels 32 \
  --batch_size 2 \
  --epochs 250 \
  --lr 2e-4 \
  --patience 30 \
  --out_dir outputs/tinyunet_full_b32
```

**Baselines** (3D U-Net, ResUNet, Attention U-Net at B=32):
```bash
for MODEL in unet resunet attentionunet; do
  python train.py \
    --data_dir /path/to/brats2023 \
    --train_split splits/train.txt \
    --val_split splits/val.txt \
    --model $MODEL \
    --base_channels 32 \
    --batch_size 2 \
    --epochs 250 \
    --lr 2e-4 \
    --patience 30 \
    --out_dir outputs/$MODEL
done
```

### Reproduce Table 4 — Progressive Ablation Study

Run all four ablation stages at B=16 in order:

```bash
# A1 — TinyUNet-Base (compact baseline)
python train.py \
  --model tinyunet_baseline --base_channels 16 \
  --data_dir /path/to/brats2023 \
  --train_split splits/train.txt --val_split splits/val.txt \
  --out_dir outputs/ablation_a1_b16

# A2 — + Sobel edge fusion
python train.py \
  --model tinyunet_edge --base_channels 16 \
  --data_dir /path/to/brats2023 \
  --train_split splits/train.txt --val_split splits/val.txt \
  --out_dir outputs/ablation_a2_b16

# A3 — + Deep supervision
python train.py \
  --model tinyunet_edge_ds --base_channels 16 \
  --data_dir /path/to/brats2023 \
  --train_split splits/train.txt --val_split splits/val.txt \
  --out_dir outputs/ablation_a3_b16

# A4 — + Laplacian boundary loss (= TinyUNet-Full)
python train.py \
  --model tinyunet_full --base_channels 16 \
  --data_dir /path/to/brats2023 \
  --train_split splits/train.txt --val_split splits/val.txt \
  --out_dir outputs/ablation_a4_b16

# Reference: B=32 baseline and full
python train.py --model tinyunet_baseline --base_channels 32 \
  --data_dir /path/to/brats2023 \
  --train_split splits/train.txt --val_split splits/val.txt \
  --out_dir outputs/ablation_base_b32

python train.py --model tinyunet_full --base_channels 32 \
  --data_dir /path/to/brats2023 \
  --train_split splits/train.txt --val_split splits/val.txt \
  --out_dir outputs/ablation_full_b32
```

### Resume a Interrupted Training Run

```bash
python train.py \
  --model tinyunet_full --base_channels 16 \
  --data_dir /path/to/brats2023 \
  --train_split splits/train.txt --val_split splits/val.txt \
  --out_dir outputs/tinyunet_full_b16 \
  --resume outputs/tinyunet_full_b16/best_model.pth
```

---

## Evaluation

Run on the held-out test set (100 patients) to reproduce Tables 2, 3, and 4.

```bash
# Example: evaluate TinyUNet-Full B=16
python evaluate.py \
  --data_dir /path/to/brats2023 \
  --test_split splits/test.txt \
  --model tinyunet_full \
  --base_channels 16 \
  --model_dir outputs/tinyunet_full_b16 \
  --out_dir outputs/tinyunet_full_b16/eval
```

Results are written to `outputs/tinyunet_full_b16/eval/test_metrics.csv`.

Column mapping to paper tables:

| CSV column | Paper metric |
|---|---|
| `dice_c1` | TC-DSC (tumour core) |
| `dice_c2` | WT-DSC (whole tumour ) |
| `dice_c3` | ET-DSC (enhancing tumour) |
| `hd95_c1` | TC-HD95 |
| `hd95_c2` | WT-HD95 |
| `hd95_c3` | ET-HD95 |
| `mean_inference_time` | Inference time (ms/patch), Table 3 |

Evaluate all models in a loop:

```bash
declare -A MODELS=(
  ["unet"]="unet,32"
  ["resunet"]="resunet,32"
  ["attentionunet"]="attentionunet,32"
  ["tinyunet_full_b16"]="tinyunet_full,16"
  ["tinyunet_full_b32"]="tinyunet_full,32"
  ["ablation_a1_b16"]="tinyunet_baseline,16"
  ["ablation_a2_b16"]="tinyunet_edge,16"
  ["ablation_a3_b16"]="tinyunet_edge_ds,16"
  ["ablation_a4_b16"]="tinyunet_full,16"
)

for RUN in "${!MODELS[@]}"; do
  IFS=',' read -r MODEL BASE <<< "${MODELS[$RUN]}"
  python evaluate.py \
    --data_dir /path/to/brats2023 \
    --test_split splits/test.txt \
    --model $MODEL \
    --base_channels $BASE \
    --model_dir outputs/$RUN \
    --out_dir outputs/$RUN/eval
done
```

---

## Model Complexity (Table 3)

To reproduce the parameter counts and FLOPs:

```bash
python model_stats.py
```

Or during training with:

```bash
python train.py ... --compute_stats
```

Pre-computed values are in `model_complexity.csv`.

---


## Hardware

All experiments were run on a single **NVIDIA L40S-16Q GPU (16 GB VRAM)**. Training time per model is approximately 6–12 hours at B=16 and 18–28 hours at B=32, depending on early stopping.

The code will run on any CUDA-capable GPU with ≥ 16 GB VRAM at batch size 2 and patch size 128³. For GPUs with less memory, reduce batch size to 1 (expect slightly lower performance due to GroupNorm with very small batches).

---

## Licence

This code is released under the [MIT License](LICENSE).

The BraTS 2023 dataset is subject to its own data use agreement. See https://www.synapse.org/brats2023 for terms. The dataset is not redistributed here.

---

## Citation

If you use this code or the TinyUNet-Full model in your research, please cite:

```bibtex
@article{tinyunetfull2026,
  title   = {TinyUNet-Full: A Compact Boundary-Aware 3D Encoder-Decoder with
             Edge-Guided Dual Supervision for Volumetric Meningioma Segmentation},
  author  = {[Author names]},
  journal = {PeerJ Computer Science},
  year    = {2026},
  doi     = {[to be added on acceptance]}
}
```

Also cite the BraTS 2023 Meningioma dataset:

```bibtex
@article{labella2023brats,
  title   = {The ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge 2023:
             Intracranial Meningioma},
  author  = {LaBella, Dominic and others},
  journal = {arXiv preprint arXiv:2305.07642},
  year    = {2023}
}

@article{labella2024analysis,
  title   = {Analysis of the BraTS 2023 Intracranial Meningioma Segmentation Challenge},
  author  = {LaBella, Dominic and others},
  journal = {arXiv preprint arXiv:2405.09787},
  year    = {2024}
}
```

---

## Contact

For questions about the code or paper, please open a GitHub issue or contact the corresponding author at [email placeholder].
