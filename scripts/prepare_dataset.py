import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ships_ear.data import prepare_clips

def main():
    p = argparse.ArgumentParser(description="Create 5-second ShipsEar clips and Dataset A/B manifests.")
    p.add_argument("--metadata", required=True, help="CSV with audio_path,class_label[,recording_order]")
    p.add_argument("--audio-root", default=None)
    p.add_argument("--output-dir", default="data/processed")
    p.add_argument("--sample-seconds", type=float, default=5.0)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    prepare_clips(a.metadata, a.audio_root, a.output_dir, a.sample_seconds, a.sample_rate, a.seed)

if __name__ == "__main__":
    main()
