from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from sklearn.model_selection import train_test_split

LABELS = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
LABEL_NAMES = ["A", "B", "C", "D", "E"]

def load_metadata(path, audio_root=None):
    frame = pd.read_csv(path)
    required = {"audio_path", "class_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing columns: {sorted(missing)}")
    frame["class_label"] = frame["class_label"].astype(str).str.strip().str.upper()
    invalid = sorted(set(frame.class_label) - set(LABELS))
    if invalid:
        raise ValueError(f"Unknown class labels {invalid}; expected A-E.")
    root = Path(audio_root) if audio_root else Path(".")
    def resolve(p):
        p = Path(str(p))
        return p if p.is_absolute() else root / p
    frame["resolved_path"] = frame.audio_path.map(resolve)
    absent = [str(p) for p in frame.resolved_path if not p.exists()]
    if absent:
        raise FileNotFoundError("Missing audio files (first 10): " + ", ".join(absent[:10]))
    if frame.resolved_path.duplicated().any():
        raise ValueError("Metadata contains duplicate audio_path entries.")
    return frame

def _split_recordings(frame, seed):
    # Dataset B follows the paper's chronological source-recording split.
    if "recording_order" in frame.columns and frame.recording_order.notna().all():
        ordered = frame.sort_values("recording_order", kind="stable")
    else:
        print("WARNING: recording_order absent/incomplete; using sorted audio_path as a fallback. Verify this matches the dataset chronology.")
        ordered = frame.sort_values("audio_path", kind="stable")
    paths = ordered.resolved_path.tolist()
    n_train = int(np.floor(0.70 * len(paths)))
    remaining = paths[n_train:]
    n_val = len(remaining) // 2
    train_paths = set(paths[:n_train])
    val_paths = set(remaining[:n_val])
    test_paths = set(remaining[n_val:])
    return train_paths, val_paths, test_paths

def prepare_clips(metadata_csv, audio_root, output_dir, sample_seconds=5.0, target_sr=16000, seed=42):
    meta = load_metadata(metadata_csv, audio_root)
    out = Path(output_dir)
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in meta.itertuples(index=False):
        audio, sr = sf.read(str(row.resolved_path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        length = int(sample_seconds * target_sr)
        usable = len(audio) // length
        for idx in range(usable):
            clip = audio[idx * length:(idx + 1) * length]
            if not np.isfinite(clip).all() or np.max(np.abs(clip)) < 1e-7:
                continue
            stem = Path(row.resolved_path).stem
            clip_name = f"{stem}_{idx:05d}.wav"
            clip_path = clips_dir / clip_name
            sf.write(str(clip_path), clip, target_sr)
            rows.append({
                "clip_path": str(clip_path),
                "source_path": str(row.resolved_path),
                "clip_index": idx,
                "class_label": row.class_label,
                "label": LABELS[row.class_label],
                "recording_order": getattr(row, "recording_order", ""),
            })
    clips = pd.DataFrame(rows)
    if clips.empty:
        raise RuntimeError("No usable 5-second clips were created. Check audio paths and files.")
    # Dataset A: random clip-level split, matching the paper's 70:15:15 ratio.
    train_idx, temp_idx = train_test_split(
        np.arange(len(clips)), test_size=0.30, random_state=seed,
        stratify=clips.label if clips.label.value_counts().min() >= 2 else None
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed,
        stratify=clips.iloc[temp_idx].label if clips.iloc[temp_idx].label.value_counts().min() >= 2 else None
    )
    clips["split_A"] = ""
    clips.loc[train_idx, "split_A"] = "train"
    clips.loc[val_idx, "split_A"] = "val"
    clips.loc[test_idx, "split_A"] = "test"
    # Dataset B: all clips from a source recording stay in one chronological subset.
    train_files, val_files, test_files = _split_recordings(meta, seed)
    clips["split_B"] = np.where(clips.source_path.isin(train_files), "train",
                         np.where(clips.source_path.isin(val_files), "val", "test"))
    clips.to_csv(out / "clips_manifest.csv", index=False)
    summary = clips.groupby(["split_A", "class_label"]).size().unstack(fill_value=0)
    summary.to_csv(out / "dataset_A_counts.csv")
    summary = clips.groupby(["split_B", "class_label"]).size().unstack(fill_value=0)
    summary.to_csv(out / "dataset_B_counts.csv")
    print(f"Created {len(clips)} usable clips from {len(meta)} original recordings.")
    print("Dataset A counts:\n", clips.split_A.value_counts())
    print("Dataset B counts:\n", clips.split_B.value_counts())
    print("Saved manifest:", out / "clips_manifest.csv")
    return clips
