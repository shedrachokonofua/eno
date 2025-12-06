# Data Documentation

## Overview

eno uses two primary data sources:

1. **RYM Metadata** — Album/track descriptors and genres scraped from Rate Your Music
2. **Audio Files** — Corresponding audio for training

## Data Location

All data lives on Smith NFS, mounted on gpu-workstation:

```
/mnt/nfs/nvme/data/eno/
├── raw/
│   ├── audio/              # Original audio files
│   │   └── {album_id}/
│   │       └── {track}.flac
│   └── metadata/
│       └── rym_dump.json   # RYM scrape
│
├── processed/
│   ├── features/           # Extracted mel spectrograms, embeddings
│   └── labels/             # Cleaned, encoded labels
│
└── cache/                  # Temporary processing artifacts
```

## Label Taxonomy

### Descriptors (multi-label)
Subjective audio characteristics:
- `melancholic`, `atmospheric`, `aggressive`, `ethereal`, `warm`, `cold`, `lush`, `sparse`, ...

### Genres (multi-label)
Musical style/genre tags:
- `shoegaze`, `post-rock`, `dream pop`, `ambient`, `noise rock`, `post-punk`, ...

## Dataset Splits

| Split | Purpose | Size |
|-------|---------|------|
| train | Training | 80% |
| val | Validation / Early stopping | 10% |
| test | Final evaluation | 10% |

Splits are stratified by genre to maintain label distribution.

## ClearML Dataset Versioning

```python
from clearml import Dataset

# Create new version
dataset = Dataset.create(
    dataset_project="eno",
    dataset_name="rym-features",
)
dataset.add_files("/mnt/nvme/data/eno/processed/")
dataset.upload()
dataset.finalize()

# Use in training
dataset = Dataset.get(dataset_project="eno", dataset_name="rym-features")
local = dataset.get_local_copy()
```

## Data Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Raw Audio  │────▶│   Resample  │────▶│   Extract   │────▶│   Store     │
│  (various)  │     │   to 16kHz  │     │   Features  │     │   .npy/.h5  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  RYM JSON   │────▶│   Parse &   │────▶│   Encode    │
│  Dump       │     │   Clean     │     │   Labels    │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Feature Extraction Options

| Feature | Dimensions | Notes |
|---------|------------|-------|
| Mel Spectrogram | (128, T) | Classic, interpretable |
| MFCC | (40, T) | Compact, lossy |
| CLAP Embedding | (512,) | Pre-trained, semantic |
| wav2vec2 | (768, T) | Pre-trained, temporal |

Start with mel spectrograms for simplicity, consider CLAP for transfer learning.




