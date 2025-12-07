# Infrastructure

## Compute

**gpu-workstation** (on Neo)
- 32 vCPU, 64GB RAM, RTX Pro 6000 (96GB VRAM)
- ClearML Server + Agent, JupyterLab
- aether@10.0.3.2

**dev-workstation** (on Trinity)
- 16 vCPU, 16GB RAM
- Cursor IDE
- shdrch@10.0.3.10

## Services

| Service | URL |
|---------|-----|
| ClearML | `https://clearml.home.shdr.ch` |
| ClearML API | `https://api.clearml.home.shdr.ch` |
| JupyterLab | `https://jupyter.home.shdr.ch` |

**Queues:** `default` (GPU), `services` (automation)

## Storage

**Smith** (NFS, 10Gbps to Neo)
- nvme: 8TB
- hdd: 28TB

```
gpu-workstation:/mnt/nfs/nvme/ → smith:/mnt/nvme/
gpu-workstation:/mnt/nfs/hdd/  → smith:/mnt/hdd/
```

eno data: `/mnt/nfs/nvme/data/ml/eno/`

## ClearML Setup

```bash
pip install clearml
clearml-init
# Web: https://clearml.home.shdr.ch
# API: https://api.clearml.home.shdr.ch
# Files: https://files.clearml.home.shdr.ch
# Credentials: ClearML Web → Settings → Create credentials
```
