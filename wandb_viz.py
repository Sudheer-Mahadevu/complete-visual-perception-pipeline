from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
# matplotlib.use("Agg")   # non-interactive; must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from api_keys import WANDB_API_KEY, WANDB_ENTITY


# ************** Shared Utilities *********************

_IMG_MEAN = (0.485, 0.456, 0.406)
_IMG_STD  = (0.229, 0.224, 0.225)

# Trimap class-index → RGB colour
_MASK_PALETTE = {
    0: (50,  50,  50),    # background  → dark-grey
    1: (0,  200,   0),    # foreground  → green
    2: (220,  30,  30),   # boundary    → red
}


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """(C,H,W) ImageNet-normalised tensor → (H,W,3) uint8 numpy array."""
    t = tensor.cpu().float().clone()
    for c, (m, s) in enumerate(zip(_IMG_MEAN, _IMG_STD)):
        t[c] = t[c] * s + m
    return (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def colorize_mask(mask_np: np.ndarray) -> np.ndarray:
    """(H,W) int mask with values 0/1/2 → (H,W,3) uint8 RGB."""
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in _MASK_PALETTE.items():
        rgb[mask_np == cls] = color
    return rgb


def to_img_transform(img_size: int = 224) -> T.Compose:
    """Standard preprocessing transform for a PIL image."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=list(_IMG_MEAN), std=list(_IMG_STD)),
    ])


def cxcywh_to_xyxy_pixel(box: torch.Tensor, W: int, H: int):
    """
    Convert a normalised (cx,cy,w,h) box → pixel (x1,y1,x2,y2).
    box: 1-D tensor of length 4, values in [0,1].
    """
    cx, cy, bw, bh = box.tolist()
    x1 = int((cx - bw / 2))
    y1 = int((cy - bh / 2))
    x2 = int((cx + bw / 2))
    y2 = int((cy + bh / 2))
    return (max(0, x1), max(0, y1), min(W, x2), min(H, y2))


def draw_boxes_on_image(img_array: np.ndarray,
                        gt_box, pred_box,
                        line_width: int = 3) -> np.ndarray:
    """
    Draw GT (green) and predicted (red) bounding boxes on a copy of img_array.

    Args:
        img_array : (H,W,3) uint8 array
        gt_box    : pixel-space (x1,y1,x2,y2) tuple — ground-truth
        pred_box  : pixel-space (x1,y1,x2,y2) tuple — prediction
    Returns:
        (H,W,3) uint8 array with boxes drawn
    """
    pil = Image.fromarray(img_array)
    draw = ImageDraw.Draw(pil)

    # Ground-truth box — green
    draw.rectangle(gt_box,   outline=(0, 220, 0),   width=line_width)
    # Prediction box — red
    draw.rectangle(pred_box, outline=(220, 30, 30), width=line_width)

    # Labels
    draw.text((gt_box[0]   + 3, gt_box[1]   + 3), "GT",   fill=(0, 220, 0))
    draw.text((pred_box[0] + 3, pred_box[1] + 3), "Pred", fill=(220, 30, 30))

    return np.array(pil)


def make_feature_grid(feat_map_np: np.ndarray, max_ch: int = 32,
                      title: str = "") -> plt.Figure:
    """
    Render up to max_ch channels of a (C,H,W) feature map as a tiled grid.
    Each tile is normalised to [0,1] independently and shown with viridis colormap.

    Returns a matplotlib Figure (not yet closed — caller must call plt.close(fig)).
    """
    C = min(feat_map_np.shape[0], max_ch)
    n_cols = min(8, C)
    n_rows = (C + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.8, n_rows * 1.8),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.1})
    axes_flat = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for i in range(C):
        fm = feat_map_np[i].astype(np.float32)
        lo, hi = fm.min(), fm.max()
        fm = (fm - lo) / (hi - lo + 1e-8)
        axes_flat[i].imshow(fm, cmap="viridis", interpolation="nearest")
        axes_flat[i].axis("off")
        axes_flat[i].set_title(f"ch{i}", fontsize=5)

    for i in range(C, len(axes_flat)):
        axes_flat[i].axis("off")

    plt.suptitle(title, fontsize=9, fontweight="bold")
    plt.tight_layout()
    return fig


# ********************* Experiment 2.4 
# Feature map visualizationFeature map visualization ******************

def exp_2_4(args):
    """
    W&B Section 2.4 — Visualize first and last conv feature maps.

    Hooked layers:
        First conv  : model.features.block1[0]  → (B, 64, 224, 224)
        Last conv   : model.features.block5[1]  → (B, 512,  14,  14)
          'before the pooling layer' = the second _conv_bn_relu in block5,
          i.e. block5[1], whose output comes before block5[2] (MaxPool2d).

    What is logged:
        feature_maps/first_conv_input_image : the dog image passed through the model
        feature_maps/first_conv_grid        : 32-channel grid of block1[0] output
        feature_maps/last_conv_grid         : 32-channel grid of block5[1] output
        feature_maps/per_channel_first_<i>  : individual channel images (first 8)
        feature_maps/per_channel_last_<i>   : individual channel images (first 8)
    """
    import wandb
    from models import VGG11Classifier
    from data.pets_dataset_aug import build_dataloaders

    wandb.init(entity=WANDB_ENTITY ,project="da6401-a2", name="exp_2_4_feature_maps",
               tags=["analysis", "feature_maps"], config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VGG11Classifier(num_classes=37).to(device)
    if args.cls_ckpt and os.path.exists(args.cls_ckpt):
        model.load_state_dict(
            torch.load(args.cls_ckpt, map_location=device), strict=True)
        print(f"Loaded classifier from {args.cls_ckpt}")
    model.eval()

    # Get one image — ideally a dog — from validation set
    _, val_loader, _ = build_dataloaders(args.data_root, img_size=224,
                                         batch_size=1, num_workers=0)
    val_iter = iter(val_loader)
    for i  in range(23):
        sample = next(val_iter)
        img_tensor  = sample["image"].to(device)   # (1,3,224,224)
        img_display = denormalize(img_tensor[0])    # (224,224,3) uint8
    plt.imshow(img_display)
    store = {}

    def make_hook(name):
        def _h(m, inp, out):
            store[name] = out.detach().cpu().float()  # (1,C,H,W)
        return _h

    h1 = model.features.block1[0].register_forward_hook(make_hook("first"))
    h2 = model.features.block5[1].register_forward_hook(make_hook("last"))

    with torch.no_grad():
        _ = model(img_tensor)

    h1.remove();  h2.remove()

    first_maps = store["first"][0].numpy()   # (64, 224, 224)
    last_maps  = store["last"][0].numpy()    # (512,  14,  14)

    # ── Log input image ───────────────────────────────────────────────────────
    wandb.log({
        "feature_maps/input_image": wandb.Image(
            img_display, caption="Input image (dog)"
        )
    })

    # ── Log tiled grids ───────────────────────────────────────────────────────
    fig_first = make_feature_grid(first_maps, max_ch=32,
                                  title="First Conv (block1) — 64 channels @ 224x224")
    wandb.log({"feature_maps/first_conv_grid": wandb.Image(fig_first)})
    # plt.close(fig_first)

    fig_last = make_feature_grid(last_maps, max_ch=32,
                                 title="Last Conv before pool (block5[1]) — 512 ch @ 14x14")
    wandb.log({"feature_maps/last_conv_grid": wandb.Image(fig_last)})
    # plt.close(fig_last)

    # ── Log first 8 channels individually for fine-grained inspection ─────────
    for i in range(min(8, first_maps.shape[0])):
        fm = first_maps[i]
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
        wandb.log({f"feature_maps/first_conv_ch{i}": wandb.Image(
            (fm * 255).astype(np.uint8), caption=f"First conv, channel {i}"
        )})

    for i in range(min(8, last_maps.shape[0])):
        fm = last_maps[i]
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
        # upscale 14×14 for visibility
        fm_pil = Image.fromarray((fm * 255).astype(np.uint8)).resize(
            (112, 112), Image.NEAREST)
        wandb.log({f"feature_maps/last_conv_ch{i}": wandb.Image(
            fm_pil, caption=f"Last conv before pool, channel {i}"
        )})

    # ── Side-by-side comparison of mean activations ───────────────────────────
    fig_compare, ax = plt.subplots(1, 2, figsize=(10, 4))

    mean_first = first_maps.mean(axis=0)
    mean_last_up = np.array(
        Image.fromarray(((last_maps.mean(axis=0) - last_maps.min()) /
                         (last_maps.max() - last_maps.min() + 1e-8) * 255
                        ).astype(np.uint8)).resize((224, 224), Image.NEAREST))

    ax[0].imshow(mean_first, cmap="viridis")
    ax[0].set_title("Mean activation — First Conv (block1)\n64 channels averaged", fontsize=9)
    ax[0].axis("off")

    ax[1].imshow(mean_last_up, cmap="viridis")
    ax[1].set_title("Mean activation — Last Conv before pool (block5[1])\n"
                    "512 channels averaged, upscaled to 224×224", fontsize=9)
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()
    wandb.log({"feature_maps/mean_activation_comparison": wandb.Image(fig_compare)})
    # plt.close(fig_compare)

    print("Experiment 2.4 logged successfully.")
    wandb.finish()

def exp_2_1():
    pass


# Experiment 2.5 — Object detection table

def exp_2_5(args):
    """
    W&B Section 2.5 — Log a detection table with ≥10 test images.

    Each row contains:
        image       : PIL image with GT box (green) and pred box (red) overlaid
        gt_box      : (cx,cy,w,h) ground-truth as string
        pred_box    : (cx,cy,w,h) predicted  as string
        confidence  : max softmax probability from the classifier (proxy for
                      detection confidence — high score = network is sure about class)
        iou         : Intersection-over-Union between GT and prediction
        failure     : True when confidence>0.7 but IoU<0.3 (high-conf, wrong location)

    What is logged:
        detection/predictions_table : wandb.Table
        detection/failure_case      : image of the worst high-confidence failure
    """
    import wandb
    from models import VGG11Localizer, VGG11Classifier, VGG11Encoder
    from data.pets_dataset_aug import build_dataloaders

    wandb.init(project="da6401-a2", name="exp_2_5_detection_table",
               tags=["analysis", "detection"], config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load localizer
    encoder = VGG11Encoder()
    if args.cls_ckpt and os.path.exists(args.cls_ckpt):
        cls_state = torch.load(args.cls_ckpt, map_location="cpu")
        enc_state = {k.replace("features.", ""): v
                     for k, v in cls_state.items() if k.startswith("features.")}
        encoder.load_state_dict(enc_state, strict=True)

    loc_model = VGG11Localizer(pretrained_features=encoder,
                               freeze_encoder=True).to(device)
    if args.loc_ckpt and os.path.exists(args.loc_ckpt):
        loc_model.load_state_dict(
            torch.load(args.loc_ckpt, map_location=device), strict=True)
        print(f"Loaded localizer from {args.loc_ckpt}")

    cls_model = VGG11Classifier(num_classes=37).to(device)
    if args.cls_ckpt and os.path.exists(args.cls_ckpt):
        cls_model.load_state_dict(
            torch.load(args.cls_ckpt, map_location=device), strict=True)
        print(f"Loaded classifier from {args.cls_ckpt}")

    loc_model.eval();  cls_model.eval()

    _, val_loader, _ = build_dataloaders(args.data_root, img_size=224,
                                         batch_size=1, num_workers=0)

    columns = ["image", "gt_box", "pred_box", "confidence", "iou", "failure_case"]
    table   = wandb.Table(columns=columns)

    worst_conf, worst_iou, worst_img_wandb = 0.0, 1.0, None

    collected = 0
    for batch in val_loader:
        if collected >= args.n_detection_samples:
            break

        img_t  = batch["image"].to(device)     # (1,3,224,224)
        gt_box = batch["bbox"][0]               # (4,) cpu tensor

        with torch.no_grad():
            pred_box_t = loc_model(img_t)[0].cpu()          # (4,)
            cls_logits = cls_model(img_t)[0]                 # (37,)
            confidence = float(cls_logits.softmax(0).max().item())

        H = W = 224

        # pixel boxes
        gt_pix   = cxcywh_to_xyxy_pixel(gt_box,     W, H)
        pred_pix = cxcywh_to_xyxy_pixel(pred_box_t, W, H)

        # compute IoU
        def _iou(a, b, eps=1e-6):
            ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            aa = (a[2]-a[0]) * (a[3]-a[1])
            ab = (b[2]-b[0]) * (b[3]-b[1])
            return inter / (aa + ab - inter + eps)

        iou = round(_iou(gt_pix, pred_pix), 4)

        # draw
        img_display = denormalize(img_t[0].cpu())
        img_boxed   = draw_boxes_on_image(img_display, gt_pix, pred_pix)

        # failure: high confidence but low IoU
        is_failure = (confidence > 0.7) and (iou < 0.3)

        img_wandb = wandb.Image(
            img_boxed,
            caption=f"conf={confidence:.2f}  IoU={iou:.3f}"
                    f"{'  ← FAILURE' if is_failure else ''}"
        )

        table.add_data(
            img_wandb,
            str([round(v, 3) for v in gt_box.tolist()]),
            str([round(v, 3) for v in pred_box_t.tolist()]),
            round(confidence, 4),
            iou,
            is_failure,
        )

        # track worst failure case (high conf + low iou)
        if is_failure and confidence > worst_conf and iou < worst_iou:
            # print(f"this is worst image so far")
            worst_conf = confidence
            worst_iou  = iou
            worst_img_wandb = wandb.Image(
                img_boxed,
                caption=(
                    f"FAILURE CASE — confidence={confidence:.3f} "
                    f"but IoU={iou:.3f}\n"
                    "High confidence in class prediction, but bbox is far off. "
                    "Possible causes: occlusion, scale mismatch, complex background."
                )
            )
        # print(f"Conf = {confidence:.2f}, IOU = {iou:.3f}, failure= {is_failure}")
        # plt.imshow(img_boxed)
        # plt.draw()
        # plt.pause(7)
        collected += 1
        print(f"  Processed {collected}/{args.n_detection_samples}")

    wandb.log({"detection/predictions_table": table})

    if worst_img_wandb is not None:
        wandb.log({"detection/worst_failure_case": worst_img_wandb})
    else:
        print("  No clear failure case (conf>0.7, IoU<0.3) found in this sample.")

    print("Experiment 2.5 logged successfully.")
    wandb.finish()

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2.6 — Segmentation visualization
# ─────────────────────────────────────────────────────────────────────────────

def exp_2_6(args):
    """
    W&B Section 2.6 — Log segmentation sample images.

    Logs a wandb.Table with 5 rows, each containing:
        original_image  : denormalised input
        gt_trimap       : colourised ground-truth mask
        pred_trimap     : colourised predicted mask
        pixel_accuracy  : pixel-wise accuracy for this sample
        dice_score      : Dice coefficient for this sample

    What is logged:
        seg/sample_table              : wandb.Table (5 rows)
        seg/pixel_acc_vs_dice_scatter : scatter plot (Exp 2.6 discussion figure)
    """
    import wandb
    from models.segmentation import VGG11UNet
    from data.pets_dataset_aug import build_dataloaders

    wandb.init(project="da6401-a2", name="exp_2_6_seg_viz",
               tags=["analysis", "segmentation"], config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VGG11UNet(num_classes=3).to(device)
    if args.seg_ckpt and os.path.exists(args.seg_ckpt):
        model.load_state_dict(
            torch.load(args.seg_ckpt, map_location=device), strict=True)
        print(f"Loaded segmentation model from {args.seg_ckpt}")
    model.eval()

    _, val_loader, _ = build_dataloaders(args.data_root, img_size=224,
                                         batch_size=1, num_workers=0)

    columns = ["original_image", "gt_trimap", "pred_trimap",
               "pixel_accuracy", "dice_score"]
    table = wandb.Table(columns=columns)

    # Collect per-sample metrics for the scatter plot
    all_pa, all_dice = [], []

    collected = 0
    with torch.no_grad():
        for batch in val_loader:
            if collected >= args.n_seg_samples:
                break

            img_t   = batch["image"].to(device)
            mask_gt = batch["mask"][0].numpy()   # (H, W)

            logits   = model(img_t)               # (1, 3, H, W)
            pred_cls = logits.argmax(dim=1)[0].cpu().numpy()   # (H, W)

            # Metrics for this single image
            preds_t  = torch.from_numpy(pred_cls).long()
            target_t = torch.from_numpy(mask_gt).long()
            pa   = (preds_t == target_t).float().mean().item()

            # Dice (foreground classes only)
            dice_sum, cnt = 0.0, 0
            for c in range(1, 3):
                p = (preds_t == c).float();  t = (target_t == c).float()
                d = p.sum() + t.sum()
                if d > 0:
                    dice_sum += (2*(p*t).sum() + 1.0) / (d + 1.0)
                    cnt += 1
            dice = (dice_sum / cnt).item() if cnt > 0 else 0.0

            all_pa.append(pa);  all_dice.append(dice)

            img_display = denormalize(img_t[0].cpu())
            gt_colored  = colorize_mask(mask_gt)
            pred_colored = colorize_mask(pred_cls)

            table.add_data(
                wandb.Image(img_display,   caption="Original"),
                wandb.Image(gt_colored,    caption="GT Trimap"),
                wandb.Image(pred_colored,  caption=f"Pred Trimap  PA={pa:.3f} Dice={dice:.3f}"),
                round(pa,   4),
                round(dice, 4),
            )
            collected += 1
            print(f"  Sample {collected}  PA={pa:.3f}  Dice={dice:.3f}")

    wandb.log({"seg/sample_table": table})

    # ── Scatter: Pixel Accuracy vs Dice ──────────────────────────────────────
    # This visualises the key Section 2.6 discussion point: PA appears high
    # while Dice is low in early epochs due to class imbalance.
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(all_pa, all_dice, color="#2196F3", s=60, edgecolors="white",
               linewidths=0.8, zorder=3)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="PA = Dice")
    ax.set_xlabel("Pixel Accuracy", fontsize=12)
    ax.set_ylabel("Dice Score",     fontsize=12)
    ax.set_title("Pixel Accuracy vs Dice Score\n"
                 "(PA is inflated by the dominant background class)", fontsize=10)
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()
    wandb.log({"seg/pixel_acc_vs_dice_scatter": wandb.Image(fig)})
    # plt.close(fig)

    print("Experiment 2.6 logged successfully.")
    wandb.finish()

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2.7 — Pipeline showcase on novel images
# ─────────────────────────────────────────────────────────────────────────────

# Oxford-IIIT Pet Dataset — 37 breed names in class-index order
_PET_BREEDS = [
    "Abyssinian", "American Bulldog", "American Pit Bull Terrier",
    "Basset Hound", "Beagle", "Bengal", "Birman", "Bombay",
    "Boxer", "British Shorthair", "Chihuahua", "Egyptian Mau",
    "English Cocker Spaniel", "English Setter", "German Shorthaired",
    "Great Pyrenees", "Havanese", "Japanese Chin", "Keeshond",
    "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland",
    "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue",
    "Saint Bernard", "Samoyed", "Scottish Terrier", "Shiba Inu",
    "Siamese", "Sphynx", "Staffordshire Bull Terrier",
    "Wheaten Terrier", "Yorkshire Terrier",
]


def exp_2_7(args):
    """
    W&B Section 2.7 — Run the unified pipeline on 3 novel images.

    Place 3 pet images (JPG/PNG) in --novel_dir.  The script:
        1. Preprocesses each image.
        2. Runs MultiTaskPerceptionModel.forward() (single pass).
        3. Overlays predicted bounding box on the image.
        4. Overlays the predicted segmentation mask.
        5. Shows the top-3 predicted breeds.

    What is logged per image (9 panels total, one wandb.Image each):
        showcase/<stem>_bbox_overlay    : image + predicted bounding box
        showcase/<stem>_seg_overlay     : image + colourised segmentation mask
        showcase/<stem>_side_by_side    : original | bbox overlay | seg overlay
        showcase/<stem>_top3_breeds     : bar chart of top-3 class probabilities
    """
    import wandb
    from models.multitask import MultiTaskPerceptionModel

    wandb.init(project="da6401-a2", name="exp_2_7_pipeline_showcase",
               tags=["showcase", "multitask"], config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskPerceptionModel(
        classifier_drive_id=args.classifier_drive_id,
        localizer_drive_id=args.localizer_drive_id,
        unet_drive_id=args.unet_drive_id,
        weights_dir=args.weights_dir,
    ).to(device)
    model.eval()

    tf = to_img_transform(img_size=224)

    # Collect image paths
    novel_dir = Path(args.novel_dir)
    img_paths = sorted([p for p in novel_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not img_paths:
        print(f"ERROR: no images found in {novel_dir}")
        return

    columns = ["ID", "BBox Overlay", "Seg Overlay", "Full Pipeline", "Top-3 Breeds"]
    showcase_table = wandb.Table(columns=columns)

    for img_path in img_paths:
        stem = img_path.stem
        pil_img = Image.open(img_path).convert("RGB")
        img_t   = tf(pil_img).unsqueeze(0).to(device)
        W, H    = pil_img.size
        orig_np = np.array(pil_img.resize((224, 224)))

        with torch.no_grad():
            out = model(img_t)

        cls_logits = out["classification"][0]        # (37,)
        bbox_pred  = out["localization"][0].cpu()    # (4,)
        seg_logits = out["segmentation"]             # (1,3,H,W)

        # ── Classification ────────────────────────────────────────────────────
        probs     = cls_logits.softmax(0).cpu().numpy()
        top3_idx  = probs.argsort()[::-1][:3]
        top3_prob = probs[top3_idx]
        top3_name = [_PET_BREEDS[i] for i in top3_idx]
        pred_breed = top3_name[0]

        # ── Bounding box overlay ──────────────────────────────────────────────
        pred_pix = cxcywh_to_xyxy_pixel(bbox_pred, 224, 224)
        bbox_img = Image.fromarray(orig_np.copy())
        draw_bbox = ImageDraw.Draw(bbox_img)
        draw_bbox.rectangle(pred_pix, outline=(220, 30, 30), width=3)
        draw_bbox.text((pred_pix[0] + 3, pred_pix[1] + 3),
                       f"{pred_breed} ({top3_prob[0]:.2f})",
                       fill=(220, 30, 30))
        bbox_np = np.array(bbox_img)

        # ── Segmentation overlay ──────────────────────────────────────────────
        pred_mask = seg_logits.argmax(dim=1)[0].cpu().numpy()   # (224, 224)
        mask_rgb  = colorize_mask(pred_mask)
        alpha     = 0.4
        seg_overlay = (orig_np * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)

        # ── Top-3 breed bar chart ─────────────────────────────────────────────
        fig_bar, ax = plt.subplots(figsize=(5, 2.5))
        ax.barh(top3_name[::-1], top3_prob[::-1],
                color=["#2196F3", "#90CAF9", "#BBDEFB"])
        ax.set_xlim(0, 1);  ax.set_xlabel("Softmax probability")
        ax.set_title(f"Top-3 breed predictions — {stem}", fontsize=9)
        plt.tight_layout()

        # ── Side-by-side panel ────────────────────────────────────────────────
        fig_side, axes = plt.subplots(1, 3, figsize=(13, 4))
        titles = ["Original (224×224)", "Predicted BBox (red)", "Seg Mask Overlay"]
        for ax_, img_arr, ttl in zip(axes,
                                     [orig_np, bbox_np, seg_overlay], titles):
            ax_.imshow(img_arr);  ax_.axis("off");  ax_.set_title(ttl, fontsize=9)
        plt.suptitle(
            f"Pipeline output: '{stem}'  →  Predicted breed: {pred_breed} "
            f"({top3_prob[0]*100:.1f}%)",
            fontsize=10, fontweight="bold"
        )
        plt.tight_layout()

        # ── Log everything ─────────────────────────────────────────────────────
        # wandb.log({
        #     f"showcase/{stem}_bbox_overlay":
        #         wandb.Image(bbox_np,     caption=f"BBox — {pred_breed}"),
        #     f"showcase/{stem}_seg_overlay":
        #         wandb.Image(seg_overlay, caption="Segmentation overlay"),
        #     f"showcase/{stem}_side_by_side":
        #         wandb.Image(fig_side,    caption=f"{stem} — full pipeline"),
        #     f"showcase/{stem}_top3_breeds":
        #         wandb.Image(fig_bar,     caption="Top-3 breed probabilities"),
        # })
        showcase_table.add_data(
            stem,
            wandb.Image(bbox_np),
            wandb.Image(seg_overlay),
            wandb.Image(fig_side),
            wandb.Image(fig_bar)
        )
        plt.close(fig_bar);  plt.close(fig_side)

        # plt.show()
        print(f"  {stem}: predicted '{pred_breed}' ({top3_prob[0]*100:.1f}%)")
    
    wandb.log({'showcase/mutli-task_results':showcase_table})
    print("Experiment 2.7 logged successfully.")
    wandb.finish()


# *********************** CLI *************************

def main():
    parser = argparse.ArgumentParser(
        description="W&B post-training visualization — Experiments 2.1/2.4/2.5/2.6/2.7"
    )

    parser.add_argument("--exp",         required=True,
                        choices=["2_1", "2_4", "2_5", "2_6", "2_7"],
                        help="Which W&B report experiment to run")

    # --- Data ---
    parser.add_argument("--data_root",   type=str, default="dataset")
    parser.add_argument("--num_workers", type=int, default=0)

    # --- Checkpoints ---
    parser.add_argument("--bn_ckpt",     type=str, default="checkpoints/classifier.pth",
                        help="(2.1) Path to trained BN model checkpoint")
    parser.add_argument("--no_bn_ckpt",  type=str, default="checkpoints/classifier_no_bn.pth",
                        help="(2.1) Path to trained No-BN model checkpoint")
    parser.add_argument("--cls_ckpt",    type=str, default="checkpoints/classifier_scratch_augs.pth",
                        help="(2.4/2.5) Classifier checkpoint")
    parser.add_argument("--loc_ckpt",    type=str, default="checkpoints/localizer.pth",
                        help="(2.5) Localizer checkpoint")
    parser.add_argument("--seg_ckpt",    type=str, default="checkpoints/unet.pth",
                        help="(2.6) Segmentation checkpoint")

    # --- Experiment 2.5 ---
    parser.add_argument("--n_detection_samples", type=int, default=15,
                        help="(2.5) Number of test images in the detection table")

    # --- Experiment 2.6 ---
    parser.add_argument("--n_seg_samples",       type=int, default=5,
                        help="(2.6) Number of segmentation sample images")

    # --- Experiment 2.7 ---
    parser.add_argument("--novel_dir",            type=str, default="novel_images/",
                        help="(2.7) Directory containing 3 novel pet images")
    parser.add_argument("--classifier_drive_id",  type=str, default="<id>")
    parser.add_argument("--localizer_drive_id",   type=str, default="<id>")
    parser.add_argument("--unet_drive_id",         type=str, default="<id>")
    parser.add_argument("--weights_dir",           type=str, default="checkpoints")

    # --- W&B ---
    parser.add_argument("--use_wandb",            action="store_true")

    args = parser.parse_args()

    if not args.use_wandb:
        # Monkey-patch wandb so code runs without actually logging
        import types, sys
        dummy = types.ModuleType("wandb")
        dummy.init       = lambda **kw: None
        dummy.log        = lambda d, **kw: None
        dummy.Image      = lambda *a, **kw: None
        dummy.Histogram  = lambda *a, **kw: None
        dummy.Table      = lambda **kw: type("T", (), {"add_data": lambda *a: None,
                                                        "columns": []})()
        dummy.finish     = lambda: None
        sys.modules["wandb"] = dummy
        print("wandb disabled — running in dry-run mode")
    else:
        import wandb
        wandb.login(key=WANDB_API_KEY)

    dispatch = {"2_1": exp_2_1, "2_4": exp_2_4, "2_5": exp_2_5,
                "2_6": exp_2_6, "2_7": exp_2_7}
    dispatch[args.exp](args)


if __name__ == "__main__":
    main()

