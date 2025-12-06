# eno

> Predicts music tags from audio. Trained on 300k albums of RYM metadata, it generates descriptors (e.g., "melancholic", "atmospheric") and genres (e.g., "shoegaze", "post-rock") for any track. Named after Brian Eno.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              dev-workstation                                │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Cursor IDE                                                           │  │
│  │  - Code editing, git, remote Jupyter kernel                           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ SSH / Tailscale
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    gpu-workstation VM (on Neo)                              │
│                  RTX Pro 6000 · 64GB RAM · 32 vCPUs                         │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │       JupyterLab            │  │           ClearML                   │  │
│  │  - Interactive exploration  │  │  - Server (Web/API/Files)           │  │
│  │  - Prototyping              │  │  - GPU Agent (training)             │  │
│  │  jupyter.home.shdr.ch       │  │  - Serving (inference)              │  │
│  └─────────────────────────────┘  │  clearml.home.shdr.ch               │  │
│                                   └─────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  NFS Mount: /mnt/nfs/nvme/data/eno/                                   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ NFS (10Gbps)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Smith (Storage)                                │
│                     ZFS: nvme (8TB) + hdd (28TB)                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
eno/
├── README.md                   # Project overview
├── pyproject.toml              # Dependencies (use uv/pip)
├── .gitignore
│
├── docs/                       # Documentation
│   ├── PROJECT.md              # This file
│   └── DATA.md                 # Dataset documentation
│
├── notebooks/                  # Jupyter notebooks (exploration only)
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_prototyping.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   └── eno/                    # Main package
│       ├── __init__.py
│       ├── data/               # Data loading & processing
│       │   ├── __init__.py
│       │   ├── dataset.py      # PyTorch Dataset classes
│       │   ├── features.py     # Audio feature extraction
│       │   └── transforms.py   # Audio augmentations
│       │
│       ├── models/             # Model architectures
│       │   ├── __init__.py
│       │   ├── encoder.py      # Audio encoder (e.g., CLAP, wav2vec2)
│       │   └── classifier.py   # Multi-label classifier head
│       │
│       ├── training/           # Training logic
│       │   ├── __init__.py
│       │   ├── trainer.py      # Training loop
│       │   └── losses.py       # Loss functions
│       │
│       └── inference/          # Inference & serving
│           ├── __init__.py
│           └── predictor.py    # Prediction interface
│
├── scripts/                    # Executable scripts
│   ├── prepare_data.py         # Data preprocessing pipeline
│   ├── train.py                # ClearML training entry point
│   └── serve.py                # ClearML serving setup
│
├── configs/                    # Experiment configs (YAML)
│   ├── base.yaml               # Base configuration
│   ├── small.yaml              # Quick iteration config
│   └── full.yaml               # Full training config
│
└── tests/                      # Unit tests
    └── test_data.py
```

## Development Workflow

### Phase 1: Data Preparation

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Raw RYM    │───▶│   Process    │───▶│   ClearML    │
│   Metadata   │    │   & Clean    │    │   Dataset    │
└──────────────┘    └──────────────┘    └──────────────┘
       +
┌──────────────┐    ┌──────────────┐
│  Raw Audio   │───▶│   Extract    │───▶ (stored on NFS)
│   Files      │    │   Features   │
└──────────────┘    └──────────────┘
```

1. **Collect data** → Store raw files on Smith NFS (`/mnt/nvme/data/eno/raw/`)
2. **Explore in Jupyter** → Understand data distribution, label frequency
3. **Write preprocessing scripts** → `scripts/prepare_data.py`
4. **Register dataset in ClearML** → Version control your data

### Phase 2: Model Development

```
┌─────────────────────────────────────────────────────────────────┐
│                    Iterative Development Loop                   │
│                                                                 │
│   ┌───────────────┐     ┌───────────────┐     ┌─────────────┐  │
│   │   Prototype   │────▶│   Refactor    │────▶│   Submit    │  │
│   │  in Jupyter   │     │   to src/     │     │  to ClearML │  │
│   └───────────────┘     └───────────────┘     └─────────────┘  │
│          ▲                                           │         │
│          └───────────────────────────────────────────┘         │
│                        Review Results                          │
└─────────────────────────────────────────────────────────────────┘
```

1. **Prototype in Jupyter** → Fast iteration, visualize results
2. **Refactor to `src/`** → Once logic is stable, move to proper modules
3. **Submit to ClearML** → Queue training job from `scripts/train.py`
4. **Review in ClearML UI** → Compare experiments, analyze metrics
5. **Iterate** → Adjust hyperparameters, try new architectures

### Phase 3: Training at Scale

```
dev-workstation                    gpu-workstation
┌─────────────┐                    ┌─────────────────────────┐
│             │   clearml-task     │                         │
│  train.py   │──────────────────▶│   ClearML Agent         │
│             │   enqueue job      │     │                   │
└─────────────┘                    │     ▼                   │
                                   │   GPU Training          │
                                   │     │                   │
                                   │     ▼                   │
                                   │   Log to Server         │
                                   │     │                   │
                                   │     ▼                   │
                                   │   Save Model Artifact   │
                                   └─────────────────────────┘
```

### Phase 4: Serving

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Model in       │───▶│  ClearML        │───▶│  API Endpoint   │
│  Registry       │    │  Serving        │    │  /predict       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## ClearML Integration

### Initial Setup (one-time)

```bash
# On dev-workstation, configure ClearML credentials
pip install clearml
clearml-init

# Enter when prompted:
# Web: https://clearml.home.shdr.ch
# API: https://api.clearml.home.shdr.ch
# Files: https://files.clearml.home.shdr.ch
# Credentials: (get from ClearML web UI → Settings → Workspace → Create credentials)
```

### Project Organization in ClearML

```
ClearML Projects
└── eno/
    ├── data-prep           # Data preprocessing experiments
    ├── training            # Model training experiments
    └── serving             # Deployed models
```

### Training Script Pattern

```python
# scripts/train.py
from clearml import Task, Dataset

def main():
    # Initialize ClearML task
    task = Task.init(
        project_name="eno/training",
        task_name="experiment-001",
        task_type=Task.TaskTypes.training,
    )

    # Log hyperparameters (auto-captured from argparse/hydra)
    task.connect(config)

    # Get versioned dataset
    dataset = Dataset.get(dataset_project="eno", dataset_name="rym-features-v1")
    local_path = dataset.get_local_copy()

    # Training loop
    for epoch in range(epochs):
        train_loss = train_one_epoch(...)
        val_loss = validate(...)

        # Log metrics (appears in ClearML UI)
        task.get_logger().report_scalar("loss", "train", train_loss, epoch)
        task.get_logger().report_scalar("loss", "val", val_loss, epoch)

    # Save model (auto-uploaded to ClearML)
    task.upload_artifact("model", model_path)

if __name__ == "__main__":
    main()
```

### Remote Execution Pattern

```bash
# Option 1: Run locally for debugging
python scripts/train.py --config configs/small.yaml

# Option 2: Queue to ClearML agent (actual GPU training)
clearml-task --project eno/training --name full-train \
    --script scripts/train.py \
    --args config=configs/full.yaml \
    --queue default
```

## Jupyter Workflow

### Connecting from dev-workstation

**Option A: Browser**
- Navigate to `https://jupyter.home.shdr.ch`
- Notebooks saved in `/home/jovyan/work/` (persisted)

**Option B: Cursor Remote Kernel**
```bash
# SSH tunnel to Jupyter
ssh -L 8888:localhost:8890 aether@gpu-workstation

# In Cursor: Connect to kernel at localhost:8888
```

### Notebook Best Practices

1. **Exploration only** → Don't train full models in notebooks
2. **Clear outputs before commit** → Keep repo clean
3. **Export to scripts** → Once logic works, move to `src/`
4. **Use ClearML in notebooks too** → Track even exploratory work

```python
# At the top of every notebook
from clearml import Task
task = Task.init(project_name="eno/exploration", task_name="notebook-exploration")
```

## Data Storage Strategy

```
gpu-workstation:/mnt/nfs/nvme/data/eno/   (NFS from Smith)
├── raw/                        # Original files (immutable)
│   ├── audio/                  # Audio files by album/track
│   └── metadata/               # RYM scrape dumps
│
├── processed/                  # Preprocessed data
│   ├── features/               # Extracted audio features
│   └── labels/                 # Processed label files
│
└── cache/                      # Temporary processing cache
```

**In code:**
```python
DATA_ROOT = "/mnt/nfs/nvme/data/eno"
```

## Recommended Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Audio Processing | `torchaudio`, `librosa` | Standard, good GPU support |
| Audio Encoder | `CLAP` or `wav2vec2` | Pre-trained, fine-tunable |
| Model Framework | PyTorch | ClearML integration, community |
| Config Management | `hydra` or `omegaconf` | Clean config composition |
| Experiment Tracking | ClearML | Already deployed in aether |
| Dataset Versioning | ClearML Data | Unified with experiments |
| Serving | ClearML Serving | Already deployed in aether |

## Getting Started Checklist

- [ ] Create GitLab repo for eno
- [ ] Set up ClearML project (`eno/`)
- [ ] Configure ClearML credentials on dev-workstation
- [ ] Mount NFS share for data access
- [ ] Create initial `pyproject.toml` with dependencies
- [ ] Set up first Jupyter notebook for data exploration
- [ ] Write data loading code in `src/eno/data/`
- [ ] Register first dataset version in ClearML
- [ ] Run first training experiment locally
- [ ] Queue first remote training job to ClearML agent

## Quick Reference

| Service | URL |
|---------|-----|
| ClearML Web | `https://clearml.home.shdr.ch` |
| ClearML API | `https://api.clearml.home.shdr.ch` |
| ClearML Files | `https://files.clearml.home.shdr.ch` |
| JupyterLab | `https://jupyter.home.shdr.ch` |
| GitLab | `https://gitlab.home.shdr.ch` |

| Queue | Purpose |
|-------|---------|
| `default` | GPU training jobs |
| `services` | Automation/pipelines |




