# eno

> Predicts music tags from audio. Trained on 300k albums of RYM metadata, it generates descriptors (e.g., "melancholic", "atmospheric") and genres (e.g., "shoegaze", "post-rock") for any track. Named after Brian Eno.

## Quick Start

### 1. Extract Data from Lute

```bash
# Build and extract
task build
task extract

# Or directly with podman
podman run --rm \
  -v /mnt/nfs/nvme/data/ml/eno/raw/metadata:/output:z \
  localhost/eno:latest
```

Configuration is hardcoded in `scripts/extract_from_lute.py` - edit and rebuild to change.

### 2. Explore Data

Open JupyterLab at `https://jupyter.home.shdr.ch` and run:

```python
import pandas as pd
df = pd.read_parquet("/mnt/nfs/nvme/data/ml/eno/raw/metadata/albums.parquet")
print(f"Loaded {len(df)} albums")
df.head()
```

### 3. Next Steps

- See `docs/EXTRACTION.md` for data extraction details
- See `docs/STEPS.md` for the full ML pipeline walkthrough
- See `docs/PROJECT.md` for architecture overview

## Project Structure

```
eno/
├── Containerfile           # Lute extraction container
├── Taskfile.yml           # Task automation (podman commands)
├── pyproject.toml         # Python dependencies
├── docs/                  # Documentation
├── notebooks/             # Jupyter exploration
├── scripts/               # Extraction & training scripts
│   └── extract_from_lute.py
├── proto/                 # Lute protobuf definitions
│   └── lute.proto
└── src/eno/              # Main package (TBD)
```

## Infrastructure

- **Development**: dev-workstation (Fedora, Cursor IDE)
- **Training**: gpu-workstation (RTX Pro 6000, 64GB RAM)
- **Storage**: Smith NFS (10Gbps, /mnt/nfs/nvme/data/ml/eno/)
- **Services**: ClearML, JupyterLab (see `docs/AETHER.md`)

## Tech Stack

| Component | Tool |
|-----------|------|
| Data Extraction | gRPC → Parquet |
| Audio Processing | torchaudio |
| Model Framework | PyTorch |
| Experiment Tracking | ClearML |
| Containerization | Podman |
| Task Automation | Task |

## Development Workflow

1. **Extract** → Get album metadata from lute
2. **Explore** → Analyze data in Jupyter
3. **Prototype** → Test models in notebooks
4. **Train** → Queue GPU jobs via ClearML
5. **Serve** → Deploy via ClearML Serving

See `docs/PROJECT.md` for detailed workflow.

## References

- [Lute](https://github.com/shedrachokonofua/lute) - RYM metadata source
- [RateYourMusic](https://rateyourmusic.com) - Original data source
- [ClearML](https://clear.ml) - Experiment tracking
