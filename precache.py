#!/usr/bin/env python3
# precache.py
# Read BRA TS .nii.gz files, per-patient per-modality z-score normalize, save .npy (float16) volumes and labels.

import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import argparse

def zscore_normalize(vol):
    mean = vol.mean()
    std = vol.std()
    if std < 1e-7:
        std = 1.0
    return (vol - mean) / std

def find_prefix(patient_dir):
    files = sorted([f for f in os.listdir(patient_dir) if f.endswith('.nii.gz')])
    if not files:
        return None
    # Example filename: BraTS-MEN-00020-000-t1c.nii.gz
    # prefix should be BraTS-MEN-00020-000
    first = files[0]
    # look for '-t' marker or '-t1' etc; fallback to first 17 chars up to the 3-digit id
    if '-t' in first:
        prefix = first.split('-t')[0]
    elif '-seg' in first:
        prefix = first.split('-seg')[0]
    else:
        prefix = first.rsplit('-', 1)[0]
    return prefix

def precache(raw_dir, out_dir, modalities=('t1n','t1c','t2w','t2f'), cast_dtype=np.float16):
    os.makedirs(out_dir, exist_ok=True)
    patients = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    print(f"[precache] Found {len(patients)} patient folders in {raw_dir}")
    for patient in tqdm(patients, desc="Precaching"):
        patient_dir = os.path.join(raw_dir, patient)
        prefix = find_prefix(patient_dir)
        if prefix is None:
            print(f"[precache] skip {patient_dir} (no nii.gz files)")
            continue

        # load modalities
        modal_arrays = []
        for mod in modalities:
            # try different filename patterns
            candidates = [
                f"{prefix}-t{mod}.nii.gz",
                f"{prefix}-{mod}.nii.gz",
                f"{prefix}_{mod}.nii.gz",
            ]
            mod_path = None
            for c in candidates:
                p = os.path.join(patient_dir, c)
                if os.path.exists(p):
                    mod_path = p
                    break
            if mod_path is None:
                raise FileNotFoundError(f"[precache] modality {mod} not found under {patient_dir}, tried {candidates}")
            img = nib.load(mod_path)
            data = img.get_fdata().astype(np.float32)
            data = zscore_normalize(data)
            modal_arrays.append(data)

        volume = np.stack(modal_arrays, axis=0)  # (C,D,H,W)
        out_vol = os.path.join(out_dir, f"{prefix}.npy")
        np.save(out_vol, volume.astype(cast_dtype))

        # segmentation
        seg_candidates = [
            os.path.join(patient_dir, f"{prefix}-seg.nii.gz"),
            os.path.join(patient_dir, f"{prefix}_seg.nii.gz"),
        ]
        seg_path = None
        for s in seg_candidates:
            if os.path.exists(s):
                seg_path = s
                break
        if seg_path is None:
            seg_files = [f for f in os.listdir(patient_dir) if 'seg' in f and f.endswith('.nii.gz')]
            if seg_files:
                seg_path = os.path.join(patient_dir, seg_files[0])
        if seg_path is None:
            raise FileNotFoundError(f"[precache] segmentation not found for {patient_dir}")
        seg = nib.load(seg_path).get_fdata().astype(np.uint8)  # (D,H,W)
        np.save(os.path.join(out_dir, f"{prefix}_label.npy"), seg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="./data/BraTS-MEN-Train/")
    parser.add_argument("--out_dir", type=str, default="./preprocessed/")
    args = parser.parse_args()
    precache(args.raw_dir, args.out_dir)
    print("[precache] done.")
