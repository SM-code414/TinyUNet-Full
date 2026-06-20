import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import label

from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotated,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandZoomd,
    RandGaussianSmoothd,
    RandShiftIntensityd,
)


# --------------------------------------------------
# Edge map generation
# --------------------------------------------------
def make_edge_from_mask(mask, iterations=1):
    bin_mask = (mask > 0).astype(np.uint8)
    struct = ndimage.generate_binary_structure(3, 1)
    dil = ndimage.binary_dilation(bin_mask, structure=struct, iterations=iterations)
    erd = ndimage.binary_erosion(bin_mask, structure=struct, iterations=iterations)
    return (dil.astype(np.uint8) - erd.astype(np.uint8))

# --------------------------------------------------
# Class-aware voxel sampling helpers
# --------------------------------------------------
def get_class_voxel_indices(label):
    """
    label: [D, H, W]
    returns dict {class_id: N x 3 array or None}
    """
    class_indices = {}
    for c in [1, 2, 3]:
        idx = np.where(label == c)
        if len(idx[0]) > 0:
            class_indices[c] = np.stack(idx, axis=1)
        else:
            class_indices[c] = None
    return class_indices


def sample_patch_center(class_indices, shape, min_region_voxels=10):
    """
    Region-based, patient-adaptive sampling:
    - Sample ONLY from classes that exist in this patient
    - Sample REGION first, then voxel inside region
    - Ignore tiny regions (noise / 1-voxel tumors)
    """

    available_classes = []

    # Build binary masks per class
    region_maps = {}

    for c in [3, 2, 1]:  # prioritize higher labels
        if class_indices[c] is None or len(class_indices[c]) == 0:
            continue

        mask = np.zeros(shape, dtype=np.uint8)
        mask[
            class_indices[c][:, 0],
            class_indices[c][:, 1],
            class_indices[c][:, 2],
        ] = 1

        labeled, num_regions = label(mask)

        regions = []
        for r in range(1, num_regions + 1):
            coords = np.argwhere(labeled == r)
            if coords.shape[0] >= min_region_voxels:
                regions.append(coords)

        if len(regions) > 0:
            region_maps[c] = regions
            available_classes.append(c)

    # ---------- Tumor-centered sampling ----------
    if len(available_classes) > 0:
        chosen_class = random.choice(available_classes)
        chosen_region = random.choice(region_maps[chosen_class])
        center = chosen_region[np.random.randint(len(chosen_region))]
        return center

    # ---------- Background fallback ----------
    return np.array([
        np.random.randint(0, shape[0]),
        np.random.randint(0, shape[1]),
        np.random.randint(0, shape[2]),
    ])

# --------------------------------------------------
# Dataset
# --------------------------------------------------
class BraTSDataset(Dataset):
    """
    Modes:
      - train : random tumor-aware patches
      - val   : full volumes (NO PATCHING)
      - test  : full volumes
    """

    def __init__(
        self,
        data_dir,
        split_file,
        patch_size=(128, 128, 128),
        n_patches_per_volume=8,
        mode="train",
        seed=42,
        use_augment=False,
        use_edge=False,
    ):
        super().__init__()

        assert mode in ("train", "val", "test")

        self.data_dir = data_dir
        self.mode = mode
        self.patch_size = patch_size
        self.n_patches = n_patches_per_volume
        self.use_augment = use_augment and mode == "train"
        self.use_edge = use_edge

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        with open(split_file) as f:
            self.patients = [l.strip() for l in f if l.strip()]

        # Training → repeat patients for patch sampling
        if self.mode == "train":
            self.index = []
            for pid in self.patients:
                for _ in range(self.n_patches):
                    self.index.append(pid)
        else:
            self.index = self.patients

        # MONAI augmentations
        if self.use_augment:
            self.augment = Compose([
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
                RandRotated(
                    keys=["image", "label"],
                    range_x=np.pi / 12,
                    prob=0.5,
                    mode=["bilinear", "nearest"]
                ),
                RandZoomd(
                    keys=["image", "label"],
                    min_zoom=0.9,
                    max_zoom=1.1,
                    prob=0.3,
                    mode=["trilinear", "nearest"]
                ),
                RandGaussianNoised(keys=["image"], prob=0.3, std=0.01),
                RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
                RandGaussianSmoothd(keys=["image"], prob=0.3),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
            ])
        else:
            self.augment = None

    # --------------------------------------------------
    def __len__(self):
        return len(self.index)

    # --------------------------------------------------
    def load_full_volume(self, patient_id):
        img_p = os.path.join(self.data_dir, f"{patient_id}.npy")
        lbl_p = os.path.join(self.data_dir, f"{patient_id}_label.npy")

        if not os.path.exists(img_p) or not os.path.exists(lbl_p):
            raise FileNotFoundError(patient_id)

        vol = np.load(img_p).astype(np.float32)      # [C, D, H, W]
        lbl = np.load(lbl_p).astype(np.uint8)        # [D, H, W]
        return vol, lbl

    # --------------------------------------------------
    def get_full_volume(self, idx):
        pid = self.patients[idx]
        vol, lbl = self.load_full_volume(pid)

        vol = torch.from_numpy(vol).unsqueeze(0)     # [1, C, D, H, W]
        lbl = torch.from_numpy(lbl).unsqueeze(0)     # [1, D, H, W]

        return vol, lbl

    # --------------------------------------------------
    def random_patch_coords(self, vol_shape):
        _, D, H, W = vol_shape
        pd, ph, pw = self.patch_size
        z = random.randint(0, D - pd)
        y = random.randint(0, H - ph)
        x = random.randint(0, W - pw)
        return z, y, x

    # --------------------------------------------------
    def extract_patch_centered(self, vol, lbl, center):
        """
        vol: [C, D, H, W]
        lbl: [D, H, W]
        center: (z, y, x)
        """
        pd, ph, pw = self.patch_size
        cz, cy, cx = center
    
        z = max(0, min(cz - pd // 2, lbl.shape[0] - pd))
        y = max(0, min(cy - ph // 2, lbl.shape[1] - ph))
        x = max(0, min(cx - pw // 2, lbl.shape[2] - pw))
    
        return (
            vol[:, z:z+pd, y:y+ph, x:x+pw],
            lbl[z:z+pd, y:y+ph, x:x+pw],
        )

    def extract_patch(self, vol, lbl, coords):
        z, y, x = coords
        pd, ph, pw = self.patch_size
        return (
            vol[:, z:z+pd, y:y+ph, x:x+pw],
            lbl[z:z+pd, y:y+ph, x:x+pw]
        )

    # --------------------------------------------------
    def __getitem__(self, idx):

        pid = self.index[idx]
        vol, lbl = self.load_full_volume(pid)

       # ---------- TRAIN: class-aware patch sampling ----------
      
        if self.mode == "train" and self.patch_size is not None:
            class_indices = get_class_voxel_indices(lbl)
            center = sample_patch_center(class_indices, lbl.shape)
            vol, lbl = self.extract_patch_centered(vol, lbl, center)
            #unique = np.unique(lbl)
            #if 3 in unique:
             #   print(f"[DEBUG] {pid}: class-3 voxels =", (lbl == 3).sum())    

        # ---------- Augmentation ----------
        vol = torch.from_numpy(vol)
        lbl = torch.from_numpy(lbl)

        if self.augment:
            data = self.augment({"image": vol, "label": lbl})
            vol, lbl = data["image"], data["label"]

        out = {
            "image": vol.float(),
            "label": lbl.long().unsqueeze(0)
        }

        if self.use_edge:
            edge = make_edge_from_mask(lbl.numpy())
            out["edge"] = torch.from_numpy(edge).unsqueeze(0).float()

        return out

