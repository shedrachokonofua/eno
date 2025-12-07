# Aether Infrastructure

Relevant infrastructure from the [aether](https://gitlab.home.shdr.ch/aether/aether) homelab for eno development.

## Compute

### gpu-workstation

Primary ML compute VM. Runs on **Neo** (physical host) with GPU passthrough.

| Spec  | Value                                        |
| ----- | -------------------------------------------- |
| vCPUs | 32                                           |
| RAM   | 64GB                                         |
| Disk  | 1TB                                          |
| GPU   | NVIDIA RTX Pro 6000 (96GB VRAM, passthrough) |
| IP    | 10.0.3.2                                     |
| OS    | Fedora                                       |
| Host  | Neo (Ryzen 9 9950X, 128GB RAM, 10Gbps)       |

**Services hosted:**

- ClearML Server (web, api, files)
- ClearML GPU Agent
- ClearML Serving
- JupyterLab
- Ollama, ComfyUI, SwarmUI, Docling

### dev-workstation

Development VM for coding. Runs on **Trinity**.

| Spec    | Value                         |
| ------- | ----------------------------- |
| vCPUs   | 16                            |
| RAM     | 16GB                          |
| Disk    | 256GB                         |
| IP      | 10.0.3.10                     |
| OS      | Fedora                        |
| Purpose | Cursor IDE, git, code editing |

## Services

### ClearML

| Endpoint | URL                                  |
| -------- | ------------------------------------ |
| Web UI   | `https://clearml.home.shdr.ch`       |
| API      | `https://api.clearml.home.shdr.ch`   |
| Files    | `https://files.clearml.home.shdr.ch` |

**Queues:**

- `default` — GPU training jobs (RTX Pro 6000)
- `services` — Automation, pipelines

**Agent Config:**

- Docker runtime with NVIDIA CDI
- Base image: `nvidia/cuda:12.9.0-runtime-ubuntu22.04`

### JupyterLab

| Endpoint | URL                            |
| -------- | ------------------------------ |
| Web UI   | `https://jupyter.home.shdr.ch` |

**Pre-installed:**

- PyTorch + CUDA 12
- transformers, datasets, tokenizers
- torchaudio, torchvision
- pandas, numpy, scikit-learn
- matplotlib, seaborn, plotly

**Volumes:**

- `/home/jovyan/work` — Notebooks (persisted)
- `/home/jovyan/data` — Data mount

## Storage

### Smith (NFS)

Dedicated storage node with ZFS. NFS mounted on gpu-workstation.

| Pool | Type       | Usable |
| ---- | ---------- | ------ |
| nvme | NVMe RAID0 | 8TB    |
| hdd  | HDD RAID10 | 28TB   |

**Mount on gpu-workstation:**

```
/mnt/nfs/nvme/    → smith:/mnt/nvme/
/mnt/nfs/hdd/     → smith:/mnt/hdd/
```

**eno data location:**

```
/mnt/nfs/nvme/data/eno/
├── raw/           # Original audio + metadata
├── processed/     # Features, encoded labels
└── cache/         # Temp processing
```

10Gbps network between Neo and Smith — NFS is fast.

## Network

| VLAN | Name     | Subnet      | Purpose                          |
| ---- | -------- | ----------- | -------------------------------- |
| 3    | Services | 10.0.3.0/24 | gpu-workstation, dev-workstation |

All ML infrastructure on 10Gbps backbone. NFS mounts are fast.

## Quick Reference

```bash
# SSH to gpu-workstation (user: aether)
ssh aether@10.0.3.2

# SSH to dev-workstation (user: shdrch)
ssh shdrch@10.0.3.10

# Tunnel Jupyter to local (port 8890 on VM)
ssh -L 8888:localhost:8890 aether@10.0.3.2
```

## ClearML Setup

```bash
# Install
pip install clearml

# Configure (~/.clearml.conf)
clearml-init

# Credentials from: ClearML Web → Settings → Workspace → Create credentials
```

Minimal `clearml.conf`:

```
api {
    web_server: https://clearml.home.shdr.ch
    api_server: https://api.clearml.home.shdr.ch
    files_server: https://files.clearml.home.shdr.ch
    credentials {
        "access_key" = "..."
        "secret_key" = "..."
    }
}
```
