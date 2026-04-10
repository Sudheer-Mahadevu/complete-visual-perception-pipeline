"""Training entrypoint
"""
import torch
from data.pets_dataset_aug import build_dataloaders
from models import VGG11Classifier
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import argparse
from sklearn.metrics import f1_score
import time

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
        # print("Hi")
        
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    train_loss = total_loss/total
    train_acc = correct/total

    return train_loss, train_acc

# this decorator makes the evaluate function to run without 
# autograd of the nn
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

        # store for f1 score
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    val_loss = total_loss/total
    val_acc = correct/total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return val_loss, val_acc, macro_f1

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device : {device}")

    # data
    train_loader, val_loader, _ = build_dataloaders(
        root = args.data_root,
        img_size = 224,
        batch_size= args.batch_size,
        num_workers = args.num_workers,
    )

    # model
    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1) # Why
    optimizer = optim.AdamW(model.parameters(), lr = args.lr,
                            weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min=1e-6) #TODO

    # TODO
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # W&B
    if args.use_wandb:
        import wandb
        wandb.init(project = "da6401-a2",name="task1-classification",
                   config=vars(args))
    
    # Training loop
    best_val_f1 = 0.0
    os.makedirs(args.save_dir, exist_ok = True)

    for epoch in range(1, args.epochs+1):
        print(f"Training Epoch{epoch}...")
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, loss_fn, device)
        lr_scheduler.step()

        end_time = time.time()
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}, time={end_time-start_time}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        # TODO: wandb

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, "best_classifier.pth"))
            print(f"Saved best model with val_f1: {val_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1.1: VGG11 Classification")
    parser.add_argument("--data_root",    type=str,   default='dataset')
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--dropout_p",    type=float, default=0.5)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--save_dir",     type=str,   default="checkpoints")
    parser.add_argument("--use_wandb",    action="store_true")
    main(parser.parse_args())