import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ships_ear.engine import train_model

def main():
    p = argparse.ArgumentParser(description="Train CNN or ResNet18 on a ShipsEar split and feature.")
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--feature", choices=["stft", "fbank", "mfcc"], required=True)
    p.add_argument("--model", choices=["cnn", "resnet18"], required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    manifest = a.manifest or f"data/processed/manifest_{a.feature}.csv"
    train_model(manifest, a.dataset, a.feature, a.model, a.output_dir,
                a.epochs, a.batch_size, a.lr, a.seed)

if __name__ == "__main__":
    main()
