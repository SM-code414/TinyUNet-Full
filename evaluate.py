import os
os.environ["MPLBACKEND"] = "Agg"

import argparse
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import BraTSDataset
from registry import get_model
from utilsall import dice_per_class, hd95_per_class, sensitivity_per_class, sliding_window_inference
from visualization import (
    save_gif,
    save_slice_overlay
)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()

    dice_all, hd95_all, sens_all = [], [], []
    inference_times = []

    for batch in tqdm(loader, desc="Testing"):
        images = batch["image"].to(device)
        masks = batch["label"].squeeze(1).to(device)

        start = time.time()
        logits = sliding_window_inference(
        images[0],         # shape (C, D, H, W)
        model,
        device,
        patch_size=(128,128,128),  # you can now use 128³
        overlap=0.5,
        num_classes=num_classes
        )
        if device == "cuda":
            torch.cuda.synchronize()
        inference_times.append(time.time() - start)
   # logits shape: (num_classes, D, H, W) -> add batch dim for metrics
        logits = torch.from_numpy(logits).unsqueeze(0).to(device)  # (1, num_classes, D, H, W)
        preds = torch.argmax(logits, dim=1)

        dice_all.append(dice_per_class(preds, masks, num_classes, ignore_bg=True))
        hd95_all.append(hd95_per_class(preds, masks, num_classes, ignore_bg=True))
        sens_all.append(sensitivity_per_class(preds, masks, num_classes)[1:])

    dice = np.nanmean(dice_all, axis=0)
    hd95 = np.nanmean(hd95_all, axis=0)
    sens = np.nanmean(sens_all, axis=0)

    return {
        "dice": dice,
        "hd95": hd95,
        "sens": sens,
        "mean_dice": np.nanmean(dice),
        "mean_hd95": np.nanmean(hd95),
        "mean_sens": np.nanmean(sens),
        "mean_inf_time": np.mean(inference_times),
    }


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = BraTSDataset(
        data_dir=args.data_dir,
        split_file=args.test_split,
        patch_size=None,
        use_augment=False
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = get_model(
        args.model,
        in_channels=4,
        num_classes=args.num_classes,
        base=args.base_channels
    ).to(device)

    ckpt = os.path.join(args.model_dir, "best_model.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    results = evaluate(model, loader, device, args.num_classes)

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.DataFrame([{
        "mean_dice": results["mean_dice"],
        "dice_c1": results["dice"][0],
        "dice_c2": results["dice"][1],
        "dice_c3": results["dice"][2],
        "mean_hd95": results["mean_hd95"],
        "hd95_c1": results["hd95"][0],
        "hd95_c2": results["hd95"][1],
        "hd95_c3": results["hd95"][2],
        "mean_sens": results["mean_sens"],
        "sens_c1": results["sens"][0],
        "sens_c2": results["sens"][1],
        "sens_c3": results["sens"][2],
        "mean_inference_time": results["mean_inf_time"],
    }])

    out_csv = os.path.join(args.out_dir, "test_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(df)
    print(f"✅ Saved to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--test_split", required=True)
    parser.add_argument("--model", default="unet3d")
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--base_channels", default=32, type=int)
    parser.add_argument("--model_dir", default="./outputs/unet")
    parser.add_argument("--out_dir", default="./outputs/eval")
    args = parser.parse_args()

    main(args)
