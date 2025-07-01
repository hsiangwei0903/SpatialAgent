import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from data_loader import InsideDataset
from model import ResNet50Binary
from loss import FocalLoss
import os

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    loss = running_loss / total
    acc = correct / total
    return loss, acc

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        acc = 100 * correct / total if total > 0 else 0
        pbar.set_postfix({"loss": f"{running_loss/total:.4f}", "acc": f"{acc:.2f}%"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def get_val_path(path):
    return path.replace("train", "val")

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training dataset & loader
    dataset = InsideDataset(
        json_path=args.json,
        image_dir=args.image_dir,
        depth_dir=args.depth_dir,
        use_depth=False
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # Validation dataset & loader
    val_json = get_val_path(args.json)
    val_image_dir = get_val_path(args.image_dir)
    val_depth_dir = get_val_path(args.depth_dir)
    val_dataset = InsideDataset(
        json_path=val_json,
        image_dir=val_image_dir,
        depth_dir=val_depth_dir,
        use_depth=False
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = ResNet50Binary(in_channels=5)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = FocalLoss(alpha=0.25, gamma=4.0)

    os.makedirs(args.save_path, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        torch.save(model.state_dict(), f"{args.save_path}/epoch_{epoch+1}.pth")
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="../data/train/inside.json")
    parser.add_argument("--image_dir", type=str, default="../data/train/images")
    parser.add_argument("--depth_dir", type=str, default="../data/train/depths")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="ckpt")
    args = parser.parse_args()
    main(args)
