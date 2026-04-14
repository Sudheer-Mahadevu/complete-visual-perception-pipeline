"""Training entrypoint — Classification (Task 1.1)

CHANGES vs original  (search '◄' to jump to every changed line):
  ◄ NEW  --run_name arg            → lets each W&B experiment get its own name
  ◄ NEW  --no_bn flag              → switches to VGG11ClassifierNoBN (Exp 2.1)
  ◄ NEW  --log_grad_freq arg       → log gradient norms every N steps (optional)
  ◄ NEW  import VGG11ClassifierNoBN
  ◄ FILL  'TODO: wandb' block      → per-epoch metric logging (Exp 2.1 / 2.2)
  ◄ NEW  log_activation_histogram() → logs 3rd-conv activation histogram (Exp 2.1)
"""

import torch
from data.pets_dataset_aug import build_dataloaders
from models import VGG11Classifier
from models.classification_nobn import VGG11ClassifierNoBN  # ◄ NEW
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import argparse
from sklearn.metrics import f1_score
import time


# ─────────────────────────────────────────────────────────────────────────────
# ◄ NEW  — Activation histogram helper (Experiment 2.1)
# ─────────────────────────────────────────────────────────────────────────────

def log_activation_histogram(model, loader, device, run_step: int):
    """
    Registers a forward hook on the 3rd convolutional layer (block3[0]),
    passes ONE batch through the model, and logs the resulting activation
    distribution as a wandb.Histogram.

    Call this once per epoch (or at key milestones) when --use_wandb is set.

    VGG11 conv layer numbering:
        block1[0] → Conv 1   (3  → 64)
        block2[0] → Conv 2   (64 → 128)
        block3[0] → Conv 3   (128 → 256)  ← hooked here
        block3[1] → Conv 4   (256 → 256)
        block4[0] → Conv 5   ...
        ...

    Args:
        model     : VGG11Classifier or VGG11ClassifierNoBN (already on device)
        loader    : DataLoader supplying the *same* fixed batch every call
        device    : torch.device
        run_step  : logged as the x-axis value in W&B
    """
    import wandb

    captured = {}

    # block3[0] is the _conv_bn_relu (or _conv_relu) Sequential
    # Its output is the post-BN+ReLU (or post-ReLU) activation — the "activation"
    hook_layer = model.features.block3[0]

    def _hook(module, inp, out):
        captured["act"] = out.detach().cpu().float()

    handle = hook_layer.register_forward_hook(_hook)

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        images = batch["image"].to(device)
        _ = model(images)

    handle.remove()
    model.train()

    act = captured["act"].numpy().flatten()        # flatten all (B,C,H,W)
    wandb.log(
        {"activations/3rd_conv_histogram": wandb.Histogram(act)},
        step=run_step,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Train / Eval (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    val_loss = total_loss / total
    val_acc  = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return val_loss, val_acc, macro_f1


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device : {device}")

    train_loader, val_loader, _ = build_dataloaders(
        root=args.data_root,
        img_size=224,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── model selection ────────────────────────────────────────────────────────
    # ◄ NEW: --no_bn switches to the NoBN variant for Experiment 2.1
    if args.no_bn:
        model = VGG11ClassifierNoBN(
            num_classes=37, dropout_p=args.dropout_p
        ).to(device)
        print("Using VGG11ClassifierNoBN  (BatchNorm disabled — Experiment 2.1)")
    else:
        model = VGG11Classifier(
            num_classes=37, dropout_p=args.dropout_p
        ).to(device)
        print("Using VGG11Classifier  (with BatchNorm)")

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn    = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer  = optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── W&B init ───────────────────────────────────────────────────────────────
    # ◄ MODIFIED: use args.run_name instead of hard-coded string
    if args.use_wandb:
        import wandb
        wandb.init(
            project="da6401-a2",
            name=args.run_name,          # ◄ NEW: flexible run name
            config=vars(args),
            tags=["classification",
                  "no_bn" if args.no_bn else "with_bn",   # ◄ NEW
                  f"dropout_{args.dropout_p}"],            # ◄ NEW: tag for exp 2.2
        )
        # ◄ NEW: watch model to log gradient norms automatically
        wandb.watch(model, log="gradients", log_freq=args.log_grad_freq)

    best_val_f1 = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"Training Epoch {epoch}...")
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, loss_fn, device)
        lr_scheduler.step()

        end_time = time.time()
        epoch_time = end_time - start_time

        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"time={epoch_time:.2f}s  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        # ── ◄ FILL: W&B logging (was 'TODO: wandb') ──────────────────────────
        if args.use_wandb:
            import wandb

            # Core training metrics — needed for Experiments 2.1 and 2.2
            wandb.log({
                "train/loss"     : train_loss,
                "train/acc"      : train_acc,
                "val/loss"       : val_loss,
                "val/acc"        : val_acc,
                "val/macro_f1"   : val_f1,
                "train/val_gap"  : train_loss - val_loss,  # ◄ NEW: generalization gap (Exp 2.2)
                "lr"             : optimizer.param_groups[0]["lr"],
                "epoch_time_s"   : epoch_time,
            }, step=epoch)

            # ◄ NEW: Activation histogram for Experiment 2.1
            # Log every epoch so convergence of the distribution is visible
            if args.log_activations:
                log_activation_histogram(model, val_loader, device,
                                         run_step=epoch)
        # ── end W&B block ──────────────────────────────────────────────────────

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, args.model_name))
            print(f"Saved best model with val_f1: {val_f1:.4f}")

    if args.use_wandb:
        import wandb
        # ◄ NEW: log best validation metric as run summary
        wandb.run.summary["best_val_f1"] = best_val_f1
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1.1: VGG11 Classification")
    parser.add_argument("--data_root",       type=str,   default="dataset")
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--weight_decay",    type=float, default=0)
    parser.add_argument("--dropout_p",       type=float, default=0.5)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--save_dir",        type=str,   default="checkpoints")
    parser.add_argument("--use_wandb",       action="store_true")
    parser.add_argument("--model_name",      type=str,   default="classifier.pth")
    # ◄ NEW args below ──────────────────────────────────────────────────────────
    parser.add_argument("--run_name",        type=str,   default="cls-run",
                        help="W&B run name — set this to distinguish experiments")
    parser.add_argument("--no_bn",           action="store_true",
                        help="Disable BatchNorm (for W&B Experiment 2.1)")
    parser.add_argument("--log_activations", action="store_true",
                        help="Log 3rd-conv activation histogram each epoch (Exp 2.1)")
    parser.add_argument("--log_grad_freq",   type=int,   default=100,
                        help="Log gradient norms every N steps via wandb.watch()")
    # ◄ end new args ────────────────────────────────────────────────────────────
    main(parser.parse_args())