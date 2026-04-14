"""
train_segmentation.py
======================
Training script for Task 1.3 - U-Net Style Semantic Segmentation.

CHANGES vs original  (search '◄' to jump to every changed line):
  ◄ NEW  --run_name arg            → flexible W&B run name for Experiment 2.3
  ◄ NEW  --log_img_every arg       → log segmentation sample images every N epochs (Exp 2.6)
  ◄ NEW  colorize_mask()           → converts class-index mask to RGB for display
  ◄ NEW  log_seg_samples()         → logs (original | GT mask | pred mask) to W&B (Exp 2.6)
  ◄ NEW  epoch_time logged to W&B  → needed for Experiment 2.3 compute-time comparison
  ◄ NEW  run summary best_val_dice → logged at end of training
"""

import argparse
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import time

from data.pets_dataset_aug import build_dataloaders
from models.vgg11 import VGG11Encoder
from models.segmentation import VGG11UNet, SegmentationLoss


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def dice_score(logits, targets, num_classes=3, smooth=1.0):
    """Mean Dice score across foreground classes (ignores background=0)."""
    preds = logits.argmax(dim=1)
    dice_sum, count = 0.0, 0
    for c in range(1, num_classes):
        pred_c   = (preds == c).float()
        target_c = (targets == c).float()
        inter = (pred_c * target_c).sum()
        denom = pred_c.sum() + target_c.sum()
        if denom > 0:
            dice_sum += (2 * inter + smooth) / (denom + smooth)
            count    += 1
    return (dice_sum / count).item() if count > 0 else 0.0


@torch.no_grad()
def pixel_accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# ◄ NEW — Visualization utilities (Experiment 2.6)
# ─────────────────────────────────────────────────────────────────────────────

# Class-index → RGB color for trimap display
_MASK_PALETTE = {
    0: (50,  50,  50),   # background → dark gray
    1: (0,  200,   0),   # foreground → green
    2: (220,  30,  30),  # boundary   → red
}

def colorize_mask(mask_np: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W) integer mask (values 0/1/2) to an (H, W, 3) uint8 RGB image.
    Called inside log_seg_samples.
    """
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in _MASK_PALETTE.items():
        rgb[mask_np == cls_idx] = color
    return rgb


def denormalize(tensor: torch.Tensor,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)) -> np.ndarray:
    """Convert an ImageNet-normalised (C,H,W) tensor to a (H,W,3) uint8 array."""
    t = tensor.cpu().float().clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    t = t.clamp(0, 1).permute(1, 2, 0).numpy()
    return (t * 255).astype(np.uint8)


@torch.no_grad()
def log_seg_samples(model, loader, device, epoch: int, n_samples: int = 5):
    """
    ◄ NEW — Logs n_samples segmentation examples to W&B as a single panel.

    For each sample the panel shows three side-by-side images:
        1. Original image (denormalised)
        2. Ground-truth trimap (colourised)
        3. Predicted trimap  (colourised)

    Logged key: "seg/sample_predictions"

    This implements the visual requirement of W&B Report Section 2.6.
    """
    import wandb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    collected_images = []
    collected_gt     = []
    collected_pred   = []

    for batch in loader:
        imgs    = batch["image"].to(device)
        targets = batch["mask"].to(device)

        logits = model(imgs)
        preds  = logits.argmax(dim=1)           # (B, H, W)

        for i in range(min(imgs.size(0), n_samples - len(collected_images))):
            collected_images.append(denormalize(imgs[i]))
            collected_gt.append(colorize_mask(targets[i].cpu().numpy()))
            collected_pred.append(colorize_mask(preds[i].cpu().numpy()))

        if len(collected_images) >= n_samples:
            break

    # Build a single matplotlib figure: n_samples rows × 3 cols
    fig, axes = plt.subplots(n_samples, 3,
                             figsize=(9, n_samples * 3),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.15})
    col_titles = ["Original", "GT Trimap", "Predicted Trimap"]

    for row in range(n_samples):
        for col, (img_array, col_title) in enumerate(zip(
                [collected_images[row], collected_gt[row], collected_pred[row]],
                col_titles)):
            ax = axes[row][col]
            ax.imshow(img_array)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_title, fontsize=11, fontweight="bold")

    plt.suptitle(f"Segmentation Samples — Epoch {epoch}", fontsize=13)

    wandb.log(
        {"seg/sample_predictions": wandb.Image(fig, caption=f"epoch_{epoch}")},
        step=epoch,
    )
    plt.close(fig)
    model.train()


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        images  = batch["image"].to(device)
        targets = batch["mask"].to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.autocast(device_type="cuda"):
                logits = model(images)
                loss   = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_pa, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        images  = batch["image"].to(device)
        targets = batch["mask"].to(device)
        logits = model(images)
        loss   = criterion(logits, targets)
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_dice += dice_score(logits, targets) * bs
        total_pa   += pixel_accuracy(logits, targets) * bs
        n += bs
    return total_loss / n, total_dice / n, total_pa / n


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(
        root=args.data_root,
        img_size=224,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        task = "segmentation"
    )

    features = VGG11Encoder()
    if args.cls_ckpt and os.path.exists(args.cls_ckpt):
        state = torch.load(args.cls_ckpt, map_location="cpu")
        feat_state = {k.replace("features.", ""): v
                      for k, v in state.items() if k.startswith("features.")}
        missing, unexpected = features.load_state_dict(feat_state, strict=True)
        print(f"[encoder] loaded from classifier | missing={missing} unexpected={unexpected}")

    freeze_encoder = (args.freeze_mode != "full")
    model = VGG11UNet(pretrained_features=features, num_classes=3,
                      freeze_encoder=freeze_encoder).to(device)

    if args.freeze_mode == "partial":
        model.unfreeze_last_encoder_blocks(num_blocks=2)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}  (mode={args.freeze_mode})")

    criterion  = SegmentationLoss(ce_wt=1.0, dice_wt=1.0)
    optimizer  = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',       # 'min' because we want to monitor validation LOSS
    factor=0.1,      # Reduce LR by 10x (5e-4 -> 5e-5)
    patience=10,       # How many epochs to wait without improvement before dropping
    )
    scaler     = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── W&B init ───────────────────────────────────────────────────────────────
    if args.use_wandb:
        import wandb
        # ◄ MODIFIED: use args.run_name; tag with freeze_mode for easy W&B filtering
        wandb.init(
            entity= args.wandb_entity,
            project="da6401-a2",
            name=args.run_name,          # ◄ NEW
            config=vars(args),
            tags=["segmentation", f"freeze_{args.freeze_mode}"],  # ◄ NEW
        )

    best_val_dice = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.model_name)

    prev_lr = args.lr
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()  # ◄ NEW: track per-epoch compute time (Exp 2.3)

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, scaler)
        val_loss, val_dice, val_pa = evaluate(model, val_loader, criterion, device)
        lr_scheduler.step(val_loss)

        # Inside your training loop (after scheduler.step(val_loss))
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != prev_lr:
            print(f"lr changed from {prev_lr:.2f} to {current_lr:.2f}")
        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
            f"val_pa={val_pa:.4f} time={epoch_time:.1f}s"
        )

        if args.use_wandb:
            import wandb
            # ◄ MODIFIED: added epoch_time for Experiment 2.3 compute comparison
            wandb.log({
                "train/loss"       : train_loss,
                "val/loss"         : val_loss,
                "val/dice"         : val_dice,
                "val/pixel_acc"    : val_pa,
                "epoch_time_s"     : epoch_time,   # ◄ NEW
                "learning_rate"    : prev_lr,
            }, step=epoch)
            prev_lr = current_lr

            # ◄ NEW: log sample segmentation images every log_img_every epochs (Exp 2.6)
            if epoch % args.log_img_every == 0:
                log_seg_samples(model, val_loader, device, epoch, n_samples=5)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"  saved best (val_dice={val_dice:.4f})")

    # ◄ NEW: log test-set metrics and run summary
    if args.use_wandb:
        import wandb
        wandb.run.summary["best_val_dice"]    = best_val_dice  # ◄ NEW
        wandb.run.summary["freeze_mode"]      = args.freeze_mode  # ◄ NEW
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1.3 - Segmentation")
    parser.add_argument("--data_root",     type=str,   default="dataset")
    parser.add_argument("--cls_ckpt",      type=str,   default="checkpoints/classifier.pth")
    parser.add_argument("--freeze_mode",   type=str,   default="partial",
                        choices=["strict", "partial", "full"])
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--save_dir",      type=str,   default="checkpoints")
    parser.add_argument("--use_wandb",     action="store_true")
    parser.add_argument("--model_name",    type=str,   default="unet.pth")
    parser.add_argument("--wandb_key",    type=str,   default=None)
    parser.add_argument("--wandb_entity",    type=str,   default=None)
    # ◄ NEW args ────────────────────────────────────────────────────────────────
    parser.add_argument("--run_name",      type=str,   default="seg-run",
                        help="W&B run name; auto-suffixed with freeze_mode if not set")
    parser.add_argument("--log_img_every", type=int,   default=5,
                        help="Log 5 segmentation sample images every N epochs (Exp 2.6)")
    # ◄ end new args ────────────────────────────────────────────────────────────
    args = parser.parse_args()

    # ◄ NEW: auto-suffix run_name with freeze mode if the user left it as default
    if args.run_name == "seg-run":
        args.run_name = f"seg_{args.freeze_mode}"
    
    if args.use_wandb:
        import wandb
        wandb.login(key = args.wandb_key)

    main(args)