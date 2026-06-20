import os
os.environ["MPLBACKEND"] = "Agg"

import argparse
import time
import csv
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import BraTSDataset
from registry import get_model
from utilsall import (
    dice_per_class,
    hd95_per_class,
    sensitivity_per_class,
    set_seed
)
from model_stats import compute_model_stats
from losses.boundary_loss import BoundaryLoss


# ─────────────────────────────────────────────────────────────
# Dice Loss
# ─────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_oh = torch.nn.functional.one_hot(
            targets.long(), num_classes=probs.shape[1]
        ).permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_oh, dims)
        union        = torch.sum(probs + targets_oh, dims)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice.mean()


# ─────────────────────────────────────────────────────────────
# Train one epoch
#
# FIX 1: model_name passed explicitly — no longer reads global args
# FIX 2: boundary_criterion passed explicitly — no longer reads global
# FIX 3: batch_size now uses args.batch_size everywhere
# ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    model_name="",            # FIX 1 & 2: passed from main()
    boundary_criterion=None,  # FIX 2: passed from main()
):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        masks  = batch["label"].squeeze(1).to(device)

        optimizer.zero_grad()
        logits = model(images)

        # ── Deep-supervision models ────────────────────────────
        if isinstance(logits, tuple):
            main_out, aux2, aux3 = logits

            loss_main = criterion(main_out, masks)
            loss_aux2 = criterion(aux2,     masks)
            loss_aux3 = criterion(aux3,     masks)

            loss = loss_main + 0.3 * loss_aux2 + 0.3 * loss_aux3

            # Boundary loss only for the Full model
            if model_name == "tinyunet_full" and boundary_criterion is not None:
                boundary_loss = boundary_criterion(main_out, masks)
                loss = loss + 0.1 * boundary_loss

        # ── Standard (single-output) models ───────────────────
        else:
            loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()

    losses          = []
    dice_all        = []
    hd95_all        = []
    sens_all        = []
    inference_times = []

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        masks  = batch["label"].squeeze(1).to(device)

        start   = time.time()
        outputs = model(images)

        # DS models return only main_out in eval mode,
        # but guard here defensively.
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        if device == "cuda":
            torch.cuda.synchronize()
        inference_times.append(time.time() - start)

        loss = criterion(logits, masks)
        losses.append(loss.item())

        preds = torch.argmax(logits, dim=1)

        dice = dice_per_class(preds, masks, num_classes, ignore_bg=True)
        hd95 = hd95_per_class(preds, masks, num_classes, ignore_bg=True)
        sens = sensitivity_per_class(preds, masks, num_classes)[1:]

        dice_all.append(dice)
        hd95_all.append(hd95)
        sens_all.append(sens)

    dice_arr = np.nanmean(np.array(dice_all), axis=0)
    hd95_arr = np.nanmean(np.array(hd95_all), axis=0)
    sens_arr = np.nanmean(np.array(sens_all), axis=0)

    return {
        "val_loss":      np.mean(losses),
        "dice":          dice_arr,
        "hd95":          hd95_arr,
        "sens":          sens_arr,
        "mean_dice":     np.nanmean(dice_arr),
        "mean_hd95":     np.nanmean(hd95_arr),
        "mean_sens":     np.nanmean(sens_arr),
        "mean_inf_time": np.mean(inference_times),
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Dataset ───────────────────────────────────────────────
    train_set = BraTSDataset(
        data_dir=args.data_dir,
        split_file=args.train_split,
        patch_size=(128, 128, 128),
    )
    val_set = BraTSDataset(
        data_dir=args.data_dir,
        split_file=args.val_split,
        patch_size=(128, 128, 128),
        use_augment=False,
    )

    # FIX 3: args.batch_size is now actually used
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,   # keep at 1 for HD95 correctness
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────
    model = get_model(
        args.model,
        in_channels=4,
        num_classes=args.num_classes,
        base=args.base_channels,
    ).to(device)

    if args.compute_stats:
        compute_model_stats(model)

    # ── Loss ──────────────────────────────────────────────────
    dice_loss = DiceLoss()
    ce_loss   = nn.CrossEntropyLoss()

    def criterion(logits, targets):
        return dice_loss(logits, targets) + 0.5 * ce_loss(logits, targets)

    # FIX 2: boundary_criterion defined here and passed into train_one_epoch
    boundary_criterion = BoundaryLoss().to(device)

    # ── Optimizer / Scheduler ─────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
    )
    start_epoch = 1
    # ── Logging ───────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    
    log_path = os.path.join(args.out_dir, "train_log.csv")
    
    log_exists = (
        args.resume is not None
        and os.path.exists(log_path)
    )
    
    with open(
        log_path,
        "a" if log_exists else "w",
        newline=""
    ) as f:
    
        writer = csv.writer(f)
    
        if not log_exists:
            writer.writerow([
                "epoch", "train_loss", "val_loss",
                "mean_dice", "dice_c1", "dice_c2", "dice_c3",
                "mean_hd95", "hd95_c1", "hd95_c2", "hd95_c3",
                "mean_sens", "sens_c1", "sens_c2", "sens_c3",
                "mean_inference_time",
            ])
    
    best_dice              = 0.0
    epochs_without_improve = 0
    
    if args.resume is not None:
    
        checkpoint = torch.load(
            args.resume,
            map_location=device
        )
    
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
    
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )
    
        best_dice = checkpoint["best_dice"]
        start_epoch = checkpoint["epoch"] + 1
    
        print(
            f"Resumed from epoch {checkpoint['epoch']} "
            f"| best_dice={best_dice:.4f}"
        )

    # ── Training Loop ─────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            model_name=args.model,                  # FIX 1 & 2
            boundary_criterion=boundary_criterion,  # FIX 2
        )

        val = validate(model, val_loader, criterion, device, args.num_classes)
        scheduler.step(val["mean_dice"])

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                val["val_loss"],
                val["mean_dice"], *val["dice"],
                val["mean_hd95"], *val["hd95"],
                val["mean_sens"], *val["sens"],
                val["mean_inf_time"],
            ])

        print(
            f"[Epoch {epoch:03d}] "
            f"Train {train_loss:.4f} | "
            f"Val Dice {val['mean_dice']:.4f}",
            flush=True
        )

        if val["mean_dice"] > best_dice + args.min_delta:
            best_dice = val["mean_dice"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_dice": best_dice,
                },
                os.path.join(args.out_dir, "best_model.pth")
            )
            

            epochs_without_improve = 0
            print(f"✅ New best model saved | Dice = {best_dice:.4f}")
        else:
            epochs_without_improve += 1
            print(f"No improvement for {epochs_without_improve} epoch(s)")

        if epochs_without_improve >= args.patience:
            print(f"\n🛑 Early stopping triggered after {epoch} epochs")
            break

    print(f"✅ Training complete | Best Mean Dice: {best_dice:.4f}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      required=True)
    parser.add_argument("--train_split",   required=True)
    parser.add_argument("--val_split",     required=True)
    parser.add_argument("--model",         default="tinyunet_baseline")
    parser.add_argument("--epochs",        default=250,  type=int)
    parser.add_argument("--batch_size",    default=2,    type=int)   # FIX 3: now wired up
    parser.add_argument("--lr",            default=2e-4, type=float)
    parser.add_argument("--num_classes",   default=4,    type=int)
    parser.add_argument("--base_channels", default=16,   type=int)   # matches original TinyUNet3D
    parser.add_argument("--out_dir",       default="outputs")
    parser.add_argument("--compute_stats", action="store_true")
    parser.add_argument("--patience",      default=30,   type=int)
    parser.add_argument("--min_delta",     default=1e-4, type=float)
    parser.add_argument("--resume", default=None, type=str)
    args = parser.parse_args()

    main(args)
