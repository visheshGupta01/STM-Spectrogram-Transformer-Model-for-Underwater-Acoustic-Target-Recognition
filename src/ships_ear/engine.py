from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from .models import build_model

class FeatureDataset(Dataset):
    def __init__(self, frame, split_col, split):
        self.frame = frame[frame[split_col] == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No samples for {split_col}={split}")
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        arr = np.load(row.feature_path).astype(np.float32)
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        x = torch.nn.functional.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False).squeeze(0)
        return x, int(row.label)

def _loader(frame, split_col, split, batch_size, shuffle):
    return DataLoader(FeatureDataset(frame, split_col, split), batch_size=batch_size, shuffle=shuffle, num_workers=0)

def train_model(manifest_csv, dataset="A", feature="fbank", model_name="resnet18",
                output_dir="outputs", epochs=100, batch_size=24, lr=1e-4, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(manifest_csv)
    split_col = "split_A" if dataset.upper() == "A" else "split_B"
    out = Path(output_dir) / f"dataset_{dataset.upper()}" / feature / model_name
    out.mkdir(parents=True, exist_ok=True)
    train_loader = _loader(frame, split_col, "train", batch_size, True)
    val_loader = _loader(frame, split_col, "val", batch_size, False)
    test_loader = _loader(frame, split_col, "test", batch_size, False)
    net = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_acc, best_path = -1.0, out / "best_model.pt"
    for epoch in range(epochs):
        net.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = net(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
        train_loss = running / len(train_loader.dataset)
        net.eval()
        val_loss, pred, truth = 0.0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = net(x)
                val_loss += criterion(logits, y).item() * len(y)
                pred.extend(logits.argmax(1).cpu().tolist())
                truth.extend(y.cpu().tolist())
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(truth, pred)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(float(val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict": net.state_dict(), "epoch": epoch + 1,
                        "val_accuracy": float(val_acc), "model": model_name,
                        "dataset": dataset, "feature": feature}, best_path)
        print(f"Epoch {epoch+1:03d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
    checkpoint = torch.load(best_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    pred, truth = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = net(x.to(device))
            pred.extend(logits.argmax(1).cpu().tolist())
            truth.extend(y.tolist())
    accuracy = float(accuracy_score(truth, pred))
    report = classification_report(truth, pred, labels=[0,1,2,3,4], target_names=["A","B","C","D","E"], output_dict=True, zero_division=0)
    cm = confusion_matrix(truth, pred, labels=[0,1,2,3,4])
    result = {"dataset": dataset, "feature": feature, "model": model_name,
              "test_accuracy": accuracy, "best_validation_accuracy": best_acc,
              "best_epoch": checkpoint["epoch"], "test_samples": len(truth),
              "classification_report": report, "confusion_matrix": cm.tolist(),
              "device": str(device)}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history["val_accuracy"], label="Validation accuracy")
    axes[1].set_title("Validation accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["A","B","C","D","E"], yticklabels=["A","B","C","D","E"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Test confusion matrix")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=160); plt.close(fig)
    print(f"Test accuracy: {accuracy:.4%}")
    print(classification_report(truth, pred, labels=[0,1,2,3,4], target_names=["A","B","C","D","E"], zero_division=0))
    print("Saved results to", out)
    return result
