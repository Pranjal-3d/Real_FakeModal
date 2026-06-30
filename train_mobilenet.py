"""
Train MobileNetV3-Small for real vs screen-recapture classification.

Usage:
    python train_mobilenet.py
"""

import glob
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from model_registry import update_model_entry
from preprocess import collect_image_paths, load_image_bgr, prepare_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(BASE_DIR, "real")
SCREEN_DIR = os.path.join(BASE_DIR, "screen")
MODEL_PATH = os.path.join(BASE_DIR, "mobilenet_liveness.pt")

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 50
PATIENCE = 10
LR = 2e-4


class LivenessDataset(Dataset):
    def __init__(self, paths, labels, transform, train=False):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = load_image_bgr(self.paths[idx])
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = prepare_image(img, use_face_crop=True, output_size=IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label


def collect_paths():
    real_paths = collect_image_paths(REAL_DIR)
    screen_paths = collect_image_paths(SCREEN_DIR)
    paths = real_paths + screen_paths
    labels = [0] * len(real_paths) + [1] * len(screen_paths)
    return paths, labels


def build_model(device):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model.to(device)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, preds, labels_all = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        batch_probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        batch_preds = (batch_probs >= 0.5).astype(int)
        probs.extend(batch_probs.tolist())
        preds.extend(batch_preds.tolist())
        labels_all.extend(labels.numpy().astype(int).tolist())
    acc = accuracy_score(labels_all, preds)
    return acc, preds, labels_all, probs


def make_sampler(labels):
    counts = np.bincount(np.array(labels))
    weights = 1.0 / counts[np.array(labels)]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def main():
    paths, labels = collect_paths()
    print(f"Found {labels.count(0)} real and {labels.count(1)} screen images.")
    if labels.count(0) == 0 or labels.count(1) == 0:
        print("Error: need images in both real/ and screen/ folders.")
        return

    train_paths, val_paths, y_train, y_val = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.04),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        LivenessDataset(train_paths, y_train, train_transform, train=True),
        batch_size=BATCH_SIZE,
        sampler=make_sampler(y_train),
        num_workers=0,
    )
    val_loader = DataLoader(
        LivenessDataset(val_paths, y_val, val_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} (MediaPipe face crop + augmentation)...")

    model = build_model(device)
    pos_weight = torch.tensor([labels.count(0) / max(labels.count(1), 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    best_acc = 0.0
    best_state = None
    stale_epochs = 0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_preds, val_labels, _ = evaluate(model, val_loader, device)
        scheduler.step(val_acc)
        print(f"Epoch {epoch:02d} | loss={train_loss:.4f} | val_acc={val_acc * 100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is None:
        print("Training failed to produce a model.")
        return

    model.load_state_dict(best_state)
    val_acc, val_preds, val_labels, _ = evaluate(model, val_loader, device)
    print(f"\nBest validation accuracy: {val_acc * 100:.2f}%")
    print(classification_report(val_labels, val_preds, target_names=["Real", "Screen"]))

    torch.save({
        "model_state_dict": best_state,
        "img_size": IMG_SIZE,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "val_accuracy": best_acc,
        "use_face_crop": True,
    }, MODEL_PATH)
    update_model_entry("mobilenet", best_acc)

    elapsed = time.time() - start
    print(f"\nSaved MobileNetV3 model to {MODEL_PATH} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
