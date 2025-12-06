# eno

Predicts music tags from audio.

Trained on 300k albums of RYM metadata, it generates descriptors (e.g., "melancholic", "atmospheric") and genres (e.g., "shoegaze", "post-rock") for any track.

Named after Brian Eno.

## Documentation

- [Project Structure & Workflow](docs/PROJECT.md) — Architecture, development workflow, ClearML integration
- [Aether Infrastructure](docs/AETHER.md) — Relevant homelab specs and endpoints
- [Data](docs/DATA.md) — Dataset organization and pipeline

## Quick Start

```bash
# 1. Clone and setup
git clone git@gitlab.home.shdr.ch:shdrch/eno.git
cd eno
uv sync  # or pip install -e .

# 2. Configure ClearML
clearml-init
# Web: https://clearml.home.shdr.ch
# API: https://api.clearml.home.shdr.ch
# Files: https://files.clearml.home.shdr.ch

# 3. Explore in Jupyter
# Open https://jupyter.home.shdr.ch

# 4. Train
python scripts/train.py --config configs/small.yaml

# 5. Queue to GPU
clearml-task --project eno/training --name experiment \
    --script scripts/train.py \
    --queue default
```

## Infrastructure

Built on [aether](https://gitlab.home.shdr.ch/aether/aether) home lab:

- **gpu-workstation** — RTX Pro 6000, hosts ClearML + JupyterLab
- **dev-workstation** — Code editing with Cursor
- **smith** — NFS storage for datasets




