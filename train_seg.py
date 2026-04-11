"""
train_segmentation.py
======================
Training script for Task 1.3 - U-Net Style Semantic Segmentation.

Usage:
    python train_segmentation.py \
        --data_root /path/to/oxford-iiit-pet \
        --cls_ckpt  checkpoints/vgg11_best.pth \
        --freeze_mode strict   # strict | partial | full
        --epochs 30 --batch_size 16
"""

import argparse
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.pets_dataset_aug import build_dataloaders
from models.vgg11 import VGG11Encoder
from models.segmentation import VGG11UNet, SegmentationLoss
import time


# Metrics
@torch.no_grad()
def dice_score(logits, targets, num_classes=3, smooth=1.0):
    """Mean Dice score across foreground classes (ignores background=0)."""
    import torch.nn.functional as F
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)

    dice_sum, count = 0.0, 0
    for c in range(1, num_classes):   # skip background
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


# Train / eval
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
        # print("Hi")

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
        n          += bs

    return total_loss / n, total_dice / n, total_pa / n


# Main
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(
        root=args.data_root,
        img_size=224,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        task='segmentation'
    )

    # encoder
    features = VGG11Encoder()
    if args.cls_ckpt and os.path.exists(args.cls_ckpt):
        state = torch.load(args.cls_ckpt, map_location="cpu")
        feat_state = {k.replace("features.", ""): v
                      for k, v in state.items() if k.startswith("features.")}
        missing, unexpected = features.load_state_dict(feat_state, strict=True)
        print(f"[encoder]   loaded from classifier  | "
              f"missing={missing} | unexpected={unexpected}")
        print(f"Loaded encoder weights from {args.cls_ckpt}")
        print(f"Loaded encoder from {args.cls_ckpt}")
        

    # freeze mode
    freeze_encoder = (args.freeze_mode == "strict")
    model = VGG11UNet(
        pretrained_features=features,
        num_classes=3,
        freeze_encoder=freeze_encoder,
    ).to(device)

    if args.freeze_mode == "partial":
        # Unfreeze last 2 conv blocks only
        model.unfreeze_last_encoder_blocks(num_blocks=2)

    # full: encoder already unfrozen (freeze_encoder=False above and nothing frozen)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}  (mode={args.freeze_mode})")

    criterion = SegmentationLoss(ce_wt=1.0, dice_wt=1.0)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    if args.use_wandb:
        import wandb
        wandb.init(project="da6401-a2",
                   name=f"task3-seg-{args.freeze_mode}",
                   config=vars(args))

    best_val_dice = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir,args.model_name)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_dice, val_pa = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        end_time = time.time()
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
            f"val_pa={val_pa:.4f}"
            f"time={(end_time-start_time):.2f}"
        )

        if args.use_wandb:
            import wandb
            wandb.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/dice": val_dice,
                "val/pixel_acc": val_pa,
            }, step=epoch)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"saved best (val_dice={val_dice:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1.3 - Segmentation")
    parser.add_argument("--data_root",    type=str,   default='dataset')
    parser.add_argument("--cls_ckpt",     type=str,   default="checkpoints/classifier.pth")
    parser.add_argument("--freeze_mode",  type=str,   default="partial",
                        choices=["strict", "partial", "full"])
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--save_dir",     type=str,   default="checkpoints")
    parser.add_argument("--use_wandb",    action="store_true")
    parser.add_argument("--model_name",   type=str,   default='unet.pth')
    main(parser.parse_args())