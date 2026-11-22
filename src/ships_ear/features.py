from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

def extract_feature(audio_path, feature="fbank", sample_rate=16000, n_fft=254, hop_ms=10, n_mels=128, n_mfcc=40):
    y, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
    hop = max(1, int(sample_rate * hop_ms / 1000))
    window = np.hanning(n_fft).astype(np.float32)
    if feature == "stft":
        spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, window=window, center=True))
    else:
        mel = librosa.feature.melspectrogram(
            y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop,
            win_length=n_fft, window=window, n_mels=n_mels, power=2.0
        )
        if feature == "fbank":
            spec = librosa.power_to_db(mel, ref=np.max)
        elif feature == "mfcc":
            spec = librosa.feature.mfcc(S=librosa.power_to_db(mel, ref=np.max), n_mfcc=n_mfcc)
        else:
            raise ValueError("feature must be stft, fbank, or mfcc")
    spec = np.asarray(spec, dtype=np.float32)
    # Per-clip standardization; stable for the small ShipsEar sample set.
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec

def cache_features(manifest_csv, output_dir, feature="fbank"):
    import pandas as pd
    frame = pd.read_csv(manifest_csv)
    out = Path(output_dir) / feature
    out.mkdir(parents=True, exist_ok=True)
    feature_paths = []
    for i, row in frame.iterrows():
        dest = out / f"clip_{i:05d}.npy"
        if not dest.exists():
            np.save(dest, extract_feature(row.clip_path, feature=feature))
        feature_paths.append(str(dest))
    frame["feature_path"] = feature_paths
    updated = Path(output_dir) / f"manifest_{feature}.csv"
    frame.to_csv(updated, index=False)
    print(f"Cached {len(frame)} {feature} features at {out}")
    return updated
