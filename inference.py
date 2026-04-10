"""Inference and evaluation
"""

from models import MultiTaskPerceptionModel
from sklearn.metrics import f1_score
import torch
from data.pets_dataset_aug import build_dataloaders

@torch.no_grad()
def _batch_iou(pred, target, eps=1e-6):
    def to_xyxy(b):
        x1 = b[:, 0] - b[:, 2] / 2;  y1 = b[:, 1] - b[:, 3] / 2
        x2 = b[:, 0] + b[:, 2] / 2;  y2 = b[:, 1] + b[:, 3] / 2
        return x1, y1, x2, y2
    px1, py1, px2, py2 = to_xyxy(pred)
    tx1, ty1, tx2, ty2 = to_xyxy(target)
    inter = ((torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0) *
             (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0))
    pa = ((px2 - px1).clamp(0) * (py2 - py1).clamp(0))
    ta = ((tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0))
    return (inter / (pa + ta - inter + eps)).mean().item()


@torch.no_grad()
def _dice(logits, targets, num_classes=3, smooth=1.0):
    preds = logits.argmax(dim=1)
    ds, count = 0.0, 0
    for c in range(1, num_classes):
        p = (preds == c).float();  t = (targets == c).float()
        d = p.sum() + t.sum()
        if d > 0:
            ds += (2 * (p * t).sum() + smooth) / (d + smooth)
            count += 1
    return (ds / count).item() if count > 0 else 0.0



@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    agg = dict(loss=0, cls=0, loc=0, seg=0, n=0)
    all_cls_preds, all_cls_labels = [], []
    total_iou, total_dice = 0.0, 0.0

    for batch in loader:
        images  = batch["image"].to(device)
        labels  = batch["label"].to(device)
        bboxes  = batch["bbox"].to(device)
        masks   = batch["mask"].to(device)

        logits = model(images)
        cls_logits = logits["classification"]         
        bbox_pred = logits["localization"]
        seg_logits= logits["segmentation"]

        bs = images.size(0)
        agg["n"]    += bs

        all_cls_preds.extend(cls_logits.argmax(1).cpu().tolist())
        all_cls_labels.extend(labels.cpu().tolist())
        total_iou  += _batch_iou(bbox_pred, bboxes) * bs
        total_dice += _dice(seg_logits, masks) * bs
        print(f"{agg["n"]} images inference done")

    n = agg.pop("n")
    macro_f1 = f1_score(all_cls_labels, all_cls_preds,
                        average="macro", zero_division=0)
    return macro_f1, total_iou / n, total_dice / n


# ********** Main ************
def main():
    classifier_drive_id = "1R_eceTm-8bbKkarbxO0jkUqLrllSDVBy"
    localizer_drive_id = "1Jm-BW5SmUl_bZDZ9V7ebjgWcCTsGwoFX"
    unet_drive_id = "1Lp87qN9qp-KemomiBt369Vp4wTdQhpyl"
    folder_path = "checkpoints"
    data_root = "dataset"
    batch_size = 8
    num_workers = 4

    """
    Model names:
    classifier.pth
    localiser.pth
    unet.pth
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MultiTaskPerceptionModel(classifier_drive_id=classifier_drive_id,
                                    localizer_drive_id=localizer_drive_id,
                                    unet_drive_id=unet_drive_id, 
                                    weights_dir=folder_path)

    train_loader, val_loader, test_loader = build_dataloaders(
            root=data_root,
            img_size=224,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    test_f1, test_iou, test_dice = evaluate(
            model,val_loader, device)

    print(f"\nTest results: "
            f"macro_f1={test_f1:.4f}  iou={test_iou:.4f}  dice={test_dice:.4f}")

if __name__ == '__main__':
    main()