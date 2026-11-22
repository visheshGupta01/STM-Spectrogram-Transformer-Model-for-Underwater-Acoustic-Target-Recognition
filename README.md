# ShipsEar CNN & ResNet18 Reproduction

A focused implementation of the CNN and ResNet18 baselines from:

**STM: Spectrogram Transformer Model for Underwater Acoustic Target Recognition**  
Li et al., *Journal of Marine Science and Engineering*, 2022.

> This repository intentionally excludes STM and all Transformer components. It reproduces the paper's CNN and ResNet18 comparisons on the ShipsEar dataset.

## Scope

- Convert original ShipsEar recordings into non-overlapping 5-second clips.
- Build Dataset A (random clip-level 70:15:15 split).
- Build Dataset B (chronological recording-level split: earliest 70% of source recordings for training; remaining recordings divided between validation and test).
- Extract STFT, 128-bin FBank, and MFCC features.
- Train the paper's two-convolution CNN baseline and a single-channel ResNet18.
- Report accuracy, precision, recall, F1, confusion matrices, and training curves.
- Compare reproduced results with the paper without fabricating results.

## Paper reference values

| Experiment | Dataset A | Dataset B |
|---|---:|---:|
| ResNet18 + STFT | 76.9% | 60.1% |
| ResNet18 + FBank | 93.8% | 88.2% |
| ResNet18 + MFCC | 93.8% | 85.5% |
| CNN baseline (paper Table 4) | 84.0% | 39.9% |
| ResNet18 baseline (paper Table 4) | 93.7% | 88.1% |

The paper's Table 2 reports ResNet18 feature-extraction comparisons. Table 4 reports model comparisons. Results produced by this code must be entered separately and must not be represented as paper results.

## Dataset setup

The ShipsEar download is not included in this repository. Obtain it from the course-provided SharePoint link and keep the original recordings outside Git.

Create a metadata CSV with these required columns:

- `audio_path`: path to an original audio recording, absolute or relative to `--audio-root`.
- `class_label`: one of `A`, `B`, `C`, `D`, `E`.
- `recording_order`: optional chronological key (date/time or sortable source-file order). Required for a meaningful Dataset B.

The five classes are:
- A: fishing boats, trawlers, mussel boats, tugboats and dredgers
- B: motorboats, pilot boats and sailboats
- C: passenger ferries
- D: ocean liners and Ro-Ro vessels
- E: background noise

If the supplied Excel metadata has different column names, inspect it and map its filename/vessel/category columns into this CSV. Do not guess labels.

Example:

```csv
audio_path,class_label,recording_order
recordings/file_001.wav,A,2012-10-01T09:00:00
recordings/file_002.wav,B,2012-10-01T10:00:00
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

python scripts/prepare_dataset.py \
  --metadata data/metadata.csv \
  --audio-root data/ShipsEar \
  --output-dir data/processed \
  --sample-seconds 5 \
  --seed 42

python scripts/train.py --dataset A --feature fbank --model resnet18
python scripts/train.py --dataset B --feature fbank --model resnet18
```

Run the notebook in Google Colab or Jupyter:

`notebooks/ships_ear_cnn_resnet18.ipynb`

## Important reproducibility notes

- Dataset A is a random split of 5-second clips using 70:15:15.
- Dataset B must split by original recording chronology before/while assigning clips. Clips from one source recording must never cross Dataset B subsets.
- The exact ordering of source recordings must come from the dataset metadata/readme. A filename sort is only a fallback and must be checked.
- Audio is resampled to 16 kHz for consistent feature extraction. The source recordings are not modified.
- STFT uses a 254-point FFT and a Hann window. FBank uses 128 mel filters and a 10 ms hop. MFCC is computed from the mel-filterbank representation.
- CNN follows the paper's two convolutional layers (1→64 and 64→256 channels, kernel 5, stride 1, padding 2), Tanh, max-pooling, dropout 0.1, and two fully connected layers.
- ResNet18 uses the standard architecture with its first convolution changed to accept one input channel.
- Training defaults follow the paper where applicable: batch size 24, maximum 100 epochs, cross-entropy loss, Adam, learning rate 1e-4. The best validation-accuracy checkpoint is evaluated on the test set.
- Small implementation details not fully specified in the paper (such as feature image normalization and CNN dense-layer dimensions) are explicitly configurable.

## Outputs

Generated data, checkpoints, metrics, plots, and confusion matrices are written under `outputs/` and are intentionally not committed. See `.gitignore`.

## Citation

Li, P., Wu, J., Wang, Y., Lan, Q., & Xiao, W. (2022). STM: Spectrogram Transformer Model for Underwater Acoustic Target Recognition. *Journal of Marine Science and Engineering*, 10(10), 1428. https://doi.org/10.3390/jmse10101428
