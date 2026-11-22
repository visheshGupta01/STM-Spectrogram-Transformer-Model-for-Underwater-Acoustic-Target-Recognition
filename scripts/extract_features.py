import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ships_ear.features import cache_features

def main():
    p = argparse.ArgumentParser(description="Precompute STFT, FBank, or MFCC feature arrays.")
    p.add_argument("--manifest", default="data/processed/clips_manifest.csv")
    p.add_argument("--output-dir", default="data/processed")
    p.add_argument("--feature", choices=["stft", "fbank", "mfcc"], required=True)
    a = p.parse_args()
    cache_features(a.manifest, a.output_dir, a.feature)

if __name__ == "__main__":
    main()
