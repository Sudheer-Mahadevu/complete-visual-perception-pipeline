"""
train_localization.py
======================
Training script for Task 1.2 – Bounding-Box Localization.

Usage:
    python train_localization.py \
        --data_root /path/to/oxford-iiit-pet \
        --cls_ckpt  checkpoints/vgg11_best.pth \
        --epochs 20 --batch_size 32
"""

import argparse
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.pets_dataset_aug import build_dataloaders
from models import VGG11Encoder, VGG11Localizer, LocalizationLoss
import time

# IoU metric (not loss — for reporting)

@torch.no_grad()
def batch_iou(pred, target, eps=1e-6):
    """Mean IoU over a batch.  Both tensors: (B,4) in (cx,cy,w,h) format."""
    def to_xyxy(b):
        x1 = b[:, 0] - b[:, 2] / 2
        y1 = b[:, 1] - b[:, 3] / 2
        x2 = b[:, 0] + b[:, 2] / 2
        y2 = b[:, 1] + b[:, 3] / 2
        return x1, y1, x2, y2

    px1, py1, px2, py2 = to_xyxy(pred)
    tx1, ty1, tx2, ty2 = to_xyxy(target)

    ix1 = torch.max(px1, tx1);  iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2);  iy2 = torch.min(py2, ty2)

    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    pred_a   = ((px2 - px1).clamp(0) * (py2 - py1).clamp(0))
    target_a = ((tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0))
    union    = pred_a + target_a - inter + eps

    return (inter / union).mean().item()


# Train / eval

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_iou, n = 0.0, 0.0, 0

    for batch in loader:
        images = batch["image"].to(device)
        bboxes = batch["bbox"].to(device)

        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, bboxes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_iou  += batch_iou(pred.detach(), bboxes) * bs
        n          += bs
        # print("Hi")

    return total_loss / n, total_iou / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou, n = 0.0, 0.0, 0

    for batch in loader:
        images = batch["image"].to(device)
        bboxes = batch["bbox"].to(device)

        pred = model(images)
        loss = criterion(pred, bboxes)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_iou  += batch_iou(pred, bboxes) * bs
        n          += bs

    return total_loss / n, total_iou / n


# Main

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data
    train_loader, val_loader, test_loader = build_dataloaders(
        root=args.data_root,
        img_size=224,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # encoder: load from classification checkpoint if provided
    features = VGG11Encoder()
    print(args.cls_ckpt)
    if args.cls_ckpt and os.path.exists(args.cls_ckpt):
        state = torch.load(args.cls_ckpt, map_location="cpu")
        # The full VGG11 checkpoint has "features.*" keys
        feat_state = {k.replace("features.", ""): v
                      for k, v in state.items() if k.startswith("features.")}
        missing, unexpected = features.load_state_dict(feat_state, strict=True)
        print(f"[encoder]   loaded from classifier  | "
              f"missing={missing} | unexpected={unexpected}")
        print(f"Loaded encoder weights from {args.cls_ckpt}")

    # model
    model = VGG11Localizer(
        pretrained_features=features,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    criterion = LocalizationLoss(iou_weight=1.0, l1_weight=0.5)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # optional W&B
    if args.use_wandb:
        import wandb
        wandb.init(project="da6401-a2", name="task2-localization",
                   config=vars(args))

    best_val_iou = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        # Optional: unfreeze encoder after warm-up
        if args.unfreeze_epoch and epoch == args.unfreeze_epoch:
            model.unfreeze_encoder()
            print(f"Unfreezing encoder at epoch {epoch}")

        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        end_time = time.time()
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} time={(end_time-start_time):.2f}  "
            f"val_loss={val_loss:.4f} val_iou={val_iou:.4f}"
        )

        if args.use_wandb:
            import wandb
            wandb.log({
                "train/loss": train_loss, "train/iou": train_iou,
                "val/loss": val_loss,     "val/iou": val_iou,
            }, step=epoch)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, args.model_name))
            print(f"Saved best model (val_iou={val_iou:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1.2 – Localization")
    parser.add_argument("--data_root",    type=str,   default='dataset')
    parser.add_argument("--cls_ckpt",       type=str,   default="checkpoints/best_classifier.pth")
    parser.add_argument("--freeze_encoder", action="store_true", default=True)
    parser.add_argument("--unfreeze_epoch", type=int,   default=None,
                        help="Epoch at which to unfreeze the encoder (optional)")
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--save_dir",       type=str,   default="checkpoints")
    parser.add_argument("--use_wandb",      action="store_true")
    parser.add_argument("--model_name",   type=str,   default='best_localizer.pth')
    main(parser.parse_args())
