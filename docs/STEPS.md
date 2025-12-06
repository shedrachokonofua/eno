# eno: Step-by-Step Guide for ML Beginners

> This guide breaks down the eno project into bite-sized steps. Each section builds on the previous one. Take your time—understanding beats speed.

## Table of Contents

1. [Phase 0: Understanding What We're Building](#phase-0-understanding-what-were-building)
2. [Phase 1: Environment Setup](#phase-1-environment-setup)
3. [Phase 2: Data Exploration](#phase-2-data-exploration)
4. [Phase 3: Feature Engineering](#phase-3-feature-engineering)
5. [Phase 4: Building the Model](#phase-4-building-the-model)
6. [Phase 5: Training](#phase-5-training)
7. [Phase 6: Evaluation & Iteration](#phase-6-evaluation--iteration)
8. [Phase 7: Serving](#phase-7-serving)

---

## Phase 0: Understanding What We're Building

### What is eno?

eno is an **audio tagging system**. You give it a song, it tells you things like:

- **Descriptors**: "melancholic", "atmospheric", "aggressive"
- **Genres**: "shoegaze", "post-rock", "dream pop"

### Where Does Our Training Data Come From?

We use [Lute](https://github.com/shedrachokonofua/lute) — a RateYourMusic scraper that has already collected **300k+ albums** with rich metadata. The labels come directly from RYM's community-generated tags:

```
┌─────────────────────────────────────────────────────────────┐
│  Lute (RYM Scraper)                                         │
│  └── Album                                                  │
│       ├── primary_genres: ["shoegaze", "dream pop"]         │  ← Labels
│       ├── secondary_genres: ["post-punk", "noise pop"]      │  ← Labels
│       ├── descriptors: ["melancholic", "atmospheric", ...]  │  ← Labels
│       ├── rating: 3.72                                      │
│       ├── tracks: [...]                                     │
│       └── ...                                               │
└─────────────────────────────────────────────────────────────┘
```

**Our labels = `primary_genres` + `secondary_genres` + `descriptors`**

### What is Multi-Label Classification?

Most beginner ML tutorials do **single-label classification**: "Is this a cat or a dog?"

eno does **multi-label classification**: a song can be _both_ "melancholic" _and_ "atmospheric" _and_ "shoegaze". Each tag is an independent yes/no decision.

```
Single-label: Pick ONE from {cat, dog, bird}
Multi-label:  Pick ANY from {melancholic, atmospheric, shoegaze, ...}
```

### The Basic ML Pipeline

Every ML project follows this pattern:

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  DATA   │───▶│ FEATURE │───▶│  MODEL  │───▶│ PREDICT │
│         │    │  PREP   │    │ TRAINING│    │         │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
  raw audio      mel specs      learns          "shoegaze,
  + RYM labels   + encoded      patterns        melancholic"
                 labels
```

**✓ Checkpoint**: Can you explain what eno does to a friend in one sentence?

---

## Phase 1: Environment Setup

### Step 1.1: Clone the Repository

```bash
git clone git@gitlab.home.shdr.ch:shdrch/eno.git
cd eno
```

### Step 1.2: Install Dependencies

We use `uv` (fast Python package manager). If you don't have it:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install deps
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Step 1.3: Configure ClearML

ClearML tracks our experiments. Set it up once:

```bash
clearml-init
```

When prompted, enter:

- **Web**: `https://clearml.home.shdr.ch`
- **API**: `https://api.clearml.home.shdr.ch`
- **Files**: `https://files.clearml.home.shdr.ch`
- **Credentials**: Get from ClearML web UI → Settings → Create credentials

### Step 1.4: Verify Data Access

Check you can reach the data on Smith NFS:

```bash
ls /mnt/nvme/data/eno/raw/
```

You should see `audio/` and `metadata/` directories.

**✓ Checkpoint**:

- [ ] Can run `python -c "import torch; print(torch.cuda.is_available())"` and see `True`
- [ ] ClearML credentials configured
- [ ] Can access `/mnt/nvme/data/eno/`

---

## Phase 2: Data Exploration

### Why Explore First?

Never train a model without understanding your data. You'll catch problems early:

- Missing labels
- Class imbalance (way more "rock" than "lowercase")
- Corrupted audio files
- Outliers

### Step 2.1: Open JupyterLab

Navigate to `https://jupyter.home.shdr.ch` in your browser.

### Step 2.2: Explore the Lute/RYM Metadata

Create notebook `01_data_exploration.ipynb`:

```python
import pandas as pd
from collections import Counter

# Load Lute-scraped RYM metadata (Parquet is fast even with 300k albums!)
df = pd.read_parquet("/mnt/nvme/data/eno/raw/metadata/albums.parquet")
print(f"Total albums: {len(df)}")  # Should be ~300k!

# Inspect the structure (matches Lute's Album proto message)
print(df.columns.tolist())
# ['name', 'file_name', 'rating', 'rating_count', 'artists',
#  'primary_genres', 'secondary_genres', 'descriptors', 'tracks', ...]

df.head()
```

> **Why Parquet?** A 300k album JSON would be ~2-5GB and take minutes to load.
> Parquet compresses to ~100-200MB and loads in seconds. See `DATA.md` for details.

### Step 2.3: Analyze Label Distribution

Our labels come from three Lute fields: `primary_genres`, `secondary_genres`, and `descriptors`.

```python
# Combine all genre labels
all_genres = []
for _, row in df.iterrows():
    all_genres.extend(row.get('primary_genres', []))
    all_genres.extend(row.get('secondary_genres', []))

genre_counts = Counter(all_genres)
print(f"Unique genres: {len(genre_counts)}")
print("\nTop 20 genres:")
for genre, count in genre_counts.most_common(20):
    print(f"  {genre}: {count}")

# Same for descriptors
all_descriptors = []
for _, row in df.iterrows():
    all_descriptors.extend(row.get('descriptors', []))

descriptor_counts = Counter(all_descriptors)
print(f"\nUnique descriptors: {len(descriptor_counts)}")
print("\nTop 20 descriptors:")
for desc, count in descriptor_counts.most_common(20):
    print(f"  {desc}: {count}")
```

**What to look for:**

- **Head-heavy distribution**: A few labels dominate (normal, but important to know)
- **Rare labels**: Some tags appear only a few times (may need to filter these out)
- **Label co-occurrence**: Which tags appear together?

### Step 2.4: Explore Audio Files

```python
import torchaudio
import matplotlib.pyplot as plt

# Pick a random audio file
audio_path = "/mnt/nvme/data/eno/raw/audio/some_album/track01.flac"
waveform, sample_rate = torchaudio.load(audio_path)

print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {waveform.shape[1] / sample_rate:.1f} seconds")
print(f"Channels: {waveform.shape[0]}")

# Visualize
plt.figure(figsize=(12, 4))
plt.plot(waveform[0].numpy()[:sample_rate*5])  # First 5 seconds
plt.title("Waveform (first 5 seconds)")
plt.show()
```

### Step 2.5: Document Your Findings

Update `DATA.md` or create notes with:

- Total number of tracks
- Label distribution stats
- Any data quality issues found
- Decisions about filtering (e.g., "Remove labels with < 100 examples")

**✓ Checkpoint**:

- [ ] Know how many tracks you have
- [ ] Know the most/least common labels
- [ ] Identified any data quality issues

---

## Phase 3: Feature Engineering

### What Are Features?

Raw audio is just numbers (samples over time). Models struggle with this directly. We convert audio into **features** that capture meaningful patterns.

### Step 3.1: Understanding Mel Spectrograms

A **mel spectrogram** shows:

- **X-axis**: Time
- **Y-axis**: Frequency (scaled to how humans hear)
- **Color**: Intensity

```python
import torchaudio.transforms as T

# Load audio
waveform, sr = torchaudio.load(audio_path)

# Convert to mel spectrogram
mel_transform = T.MelSpectrogram(
    sample_rate=sr,
    n_mels=128,           # 128 frequency bins
    n_fft=2048,           # FFT window size
    hop_length=512,       # Step between windows
)

mel_spec = mel_transform(waveform)
print(f"Shape: {mel_spec.shape}")  # (channels, n_mels, time_frames)

# Visualize
plt.figure(figsize=(12, 4))
plt.imshow(mel_spec[0].log2().numpy(), aspect='auto', origin='lower')
plt.colorbar(label='Log Power')
plt.title("Mel Spectrogram")
plt.xlabel("Time Frame")
plt.ylabel("Mel Frequency Bin")
plt.show()
```

### Step 3.2: Create Feature Extraction Script

Move your notebook code to `src/eno/data/features.py`:

```python
# src/eno/data/features.py
import torch
import torchaudio
import torchaudio.transforms as T

class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mels=128):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512,
        )
        self.resampler_cache = {}

    def extract(self, audio_path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            if sr not in self.resampler_cache:
                self.resampler_cache[sr] = T.Resample(sr, self.sample_rate)
            waveform = self.resampler_cache[sr](waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Extract mel spectrogram
        mel_spec = self.mel_transform(waveform)

        # Convert to log scale (better for neural networks)
        mel_spec = torch.log(mel_spec + 1e-9)

        return mel_spec
```

### Step 3.3: Run Batch Feature Extraction

Create `scripts/prepare_data.py`:

```python
# scripts/prepare_data.py
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from eno.data.features import AudioFeatureExtractor

def main():
    data_root = Path("/mnt/nvme/data/eno")
    extractor = AudioFeatureExtractor()

    # Load metadata
    with open(data_root / "raw/metadata/rym_dump.json") as f:
        albums = json.load(f)

    output_dir = data_root / "processed/features"
    output_dir.mkdir(parents=True, exist_ok=True)

    for album in tqdm(albums, desc="Processing albums"):
        album_id = album["id"]
        for track in album["tracks"]:
            audio_path = data_root / f"raw/audio/{album_id}/{track['filename']}"
            if not audio_path.exists():
                continue

            features = extractor.extract(str(audio_path))

            # Save as numpy (smaller than torch)
            out_path = output_dir / f"{album_id}_{track['id']}.npy"
            np.save(out_path, features.numpy())

if __name__ == "__main__":
    main()
```

### Step 3.4: Encode Labels

Labels need to be numbers, not strings. We combine Lute's three label fields into one vector:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Multi-Hot Label Vector                           │
├──────────────────────┬───────────────────────┬──────────────────────────┤
│   Primary Genres     │   Secondary Genres    │      Descriptors         │
│  [1, 0, 0, 1, 0...]  │  [0, 1, 0, 0, 1...]   │  [1, 1, 0, 0, 1, 0...]   │
└──────────────────────┴───────────────────────┴──────────────────────────┘
         ↑ shoegaze=1           ↑ noise pop=1           ↑ melancholic=1
```

```python
# src/eno/data/labels.py
import json
import numpy as np
from typing import List, Dict
from pathlib import Path

class LabelEncoder:
    """Encodes Lute's primary_genres, secondary_genres, and descriptors into vectors."""

    def __init__(self, genres: List[str], descriptors: List[str]):
        # Genres (primary + secondary share the same vocabulary)
        self.genres = sorted(set(genres))
        self.descriptors = sorted(set(descriptors))

        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}
        self.desc_to_idx = {d: i for i, d in enumerate(self.descriptors)}

        self.num_genres = len(self.genres)
        self.num_descriptors = len(self.descriptors)
        self.num_labels = self.num_genres + self.num_descriptors

    def encode(
        self,
        primary_genres: List[str],
        secondary_genres: List[str],
        descriptors: List[str]
    ) -> np.ndarray:
        """Convert Lute album labels to multi-hot vector."""
        vector = np.zeros(self.num_labels, dtype=np.float32)

        # Both primary and secondary genres map to the same genre indices
        for g in primary_genres + secondary_genres:
            if g in self.genre_to_idx:
                vector[self.genre_to_idx[g]] = 1.0

        for d in descriptors:
            if d in self.desc_to_idx:
                vector[self.num_genres + self.desc_to_idx[d]] = 1.0

        return vector

    def decode(self, vector: np.ndarray, threshold: float = 0.5) -> Dict[str, List[str]]:
        """Convert prediction vector back to labels."""
        genres = [self.genres[i] for i in range(self.num_genres)
                  if vector[i] > threshold]
        descriptors = [self.descriptors[i] for i in range(self.num_descriptors)
                       if vector[self.num_genres + i] > threshold]
        return {"genres": genres, "descriptors": descriptors}

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({"genres": self.genres, "descriptors": self.descriptors}, f)

    @classmethod
    def load(cls, path: str) -> "LabelEncoder":
        with open(path) as f:
            data = json.load(f)
        return cls(data["genres"], data["descriptors"])

    def __len__(self):
        return self.num_labels
```

**✓ Checkpoint**:

- [ ] Can extract mel spectrogram from an audio file
- [ ] Processed features saved to `/mnt/nvme/data/eno/processed/features/`
- [ ] Have a label encoder that converts strings ↔ vectors

---

## Phase 4: Building the Model

### What's a Neural Network?

Think of it as a series of transformations:

```
Input (mel spec) → [Layer 1] → [Layer 2] → ... → Output (predictions)
```

Each layer learns to extract increasingly abstract features.

### Step 4.1: Create the Dataset Class

PyTorch needs a `Dataset` that loads your data:

```python
# src/eno/data/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path

class EnoDataset(Dataset):
    """PyTorch Dataset for eno training data."""

    def __init__(self, split: str, data_root: str, label_encoder):
        self.data_root = Path(data_root)
        self.label_encoder = label_encoder

        # Load split metadata from Parquet (fast even with 100k samples)
        self.df = pd.read_parquet(
            self.data_root / f"processed/splits/{split}.parquet",
            columns=["file_name", "primary_genres", "secondary_genres", "descriptors"]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load pre-extracted features (mel spectrogram)
        features = np.load(self.data_root / f"processed/features/{row['file_name']}.npy")
        features = torch.from_numpy(features)

        # Encode Lute labels to multi-hot vector
        labels = self.label_encoder.encode(
            primary_genres=row['primary_genres'] or [],
            secondary_genres=row['secondary_genres'] or [],
            descriptors=row['descriptors'] or []
        )
        labels = torch.from_numpy(labels)

        return features, labels
```

### Step 4.2: Choose a Model Architecture

Start simple! A **CNN (Convolutional Neural Network)** works well for spectrograms:

```python
# src/eno/models/classifier.py
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """A simple CNN for audio classification. Start here!"""

    def __init__(self, n_mels: int, num_labels: int):
        super().__init__()

        # Convolutional layers (extract patterns from spectrogram)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Collapse to fixed size
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_labels),
            # No softmax! We use BCEWithLogitsLoss which includes it
        )

    def forward(self, x):
        # x shape: (batch, 1, n_mels, time)
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x
```

### Step 4.3: Understand the Loss Function

For multi-label, we use **Binary Cross-Entropy (BCE)**:

```python
# Each label is an independent binary classification
loss_fn = nn.BCEWithLogitsLoss()

# Example:
# predictions: [0.8, 0.2, 0.9]  (model outputs, before sigmoid)
# labels:      [1.0, 0.0, 1.0]  (ground truth)
# Loss penalizes: pred[0] should be high ✓, pred[1] should be low ✓, pred[2] should be high ✓
```

**✓ Checkpoint**:

- [ ] Understand what a Dataset class does
- [ ] Can explain what CNN layers do (extract local patterns)
- [ ] Know why we use BCE loss for multi-label

---

## Phase 5: Training

### Step 5.1: Write the Training Loop

```python
# src/eno/training/trainer.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for features, labels in tqdm(dataloader, desc="Training"):
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        predictions = model(features)
        loss = loss_fn(predictions, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            predictions = model(features)
            loss = loss_fn(predictions, labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)
```

### Step 5.2: Create Training Script

```python
# scripts/train.py
import torch
from torch.utils.data import DataLoader
from clearml import Task

from eno.data.dataset import EnoDataset
from eno.data.labels import LabelEncoder
from eno.models.classifier import SimpleCNN
from eno.training.trainer import train_one_epoch, validate

def main():
    # Initialize ClearML tracking
    task = Task.init(project_name="eno/training", task_name="first-experiment")

    # Config (ClearML auto-tracks these!)
    config = {
        "batch_size": 32,
        "learning_rate": 1e-3,
        "epochs": 50,
        "data_root": "/mnt/nvme/data/eno",
    }
    task.connect(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup data
    label_encoder = LabelEncoder.load(config["data_root"] + "/processed/label_encoder.json")
    train_dataset = EnoDataset("train", config["data_root"], label_encoder)
    val_dataset = EnoDataset("val", config["data_root"], label_encoder)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    # Setup model
    model = SimpleCNN(n_mels=128, num_labels=len(label_encoder))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device)

        # Log to ClearML
        logger = task.get_logger()
        logger.report_scalar("loss", "train", train_loss, epoch)
        logger.report_scalar("loss", "validation", val_loss, epoch)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            task.upload_artifact("best_model", "best_model.pt")

if __name__ == "__main__":
    main()
```

### Step 5.3: Run Training Locally First

Always test locally with a tiny subset before using GPU:

```bash
# Quick sanity check (small data, few epochs)
python scripts/train.py --epochs 2 --batch_size 4
```

### Step 5.4: Queue Training on GPU

Once local test passes, run the real thing:

```bash
clearml-task \
    --project eno/training \
    --name "full-train-v1" \
    --script scripts/train.py \
    --queue default
```

Monitor progress at `https://clearml.home.shdr.ch`

**✓ Checkpoint**:

- [ ] Training loop runs without errors
- [ ] Can see loss decreasing in ClearML UI
- [ ] Saved model checkpoint

---

## Phase 6: Evaluation & Iteration

### Step 6.1: Understand Metrics

For multi-label classification, accuracy isn't enough. Track:

| Metric        | What It Measures                                              |
| ------------- | ------------------------------------------------------------- |
| **Precision** | Of all predicted "shoegaze", how many were actually shoegaze? |
| **Recall**    | Of all actual shoegaze songs, how many did we catch?          |
| **F1 Score**  | Balance of precision and recall                               |
| **mAP**       | Mean Average Precision across all labels                      |

```python
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

def evaluate(model, dataloader, device, label_encoder):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            preds = torch.sigmoid(model(features))  # Convert to probabilities
            all_preds.append(preds.cpu())
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Binary predictions at 0.5 threshold
    binary_preds = (all_preds > 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, binary_preds, average='macro'
    )

    mAP = average_precision_score(all_labels, all_preds, average='macro')

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP": mAP,
    }
```

### Step 6.2: Analyze Errors

Create a notebook `04_evaluation.ipynb`:

```python
# Which labels does the model struggle with?
per_label_metrics = precision_recall_fscore_support(all_labels, binary_preds, average=None)

for i, label in enumerate(label_encoder.all_labels):
    print(f"{label}: P={per_label_metrics[0][i]:.2f}, R={per_label_metrics[1][i]:.2f}")
```

**Common issues and fixes:**

| Problem         | Symptom                       | Fix                              |
| --------------- | ----------------------------- | -------------------------------- |
| Underfitting    | Train loss high               | More epochs, bigger model        |
| Overfitting     | Train loss low, val loss high | Dropout, augmentation, more data |
| Class imbalance | Rare labels never predicted   | Class weights, focal loss        |

### Step 6.3: Iterate!

The ML cycle is:

```
Train → Evaluate → Analyze → Hypothesize → Change Something → Repeat
```

Ideas to try:

- [ ] More training data
- [ ] Data augmentation (pitch shift, time stretch)
- [ ] Different architecture (try a pre-trained encoder like CLAP)
- [ ] Hyperparameter tuning (learning rate, batch size)
- [ ] Loss function changes (focal loss for imbalanced labels)

**✓ Checkpoint**:

- [ ] Know your model's F1 and mAP scores
- [ ] Identified which labels perform worst
- [ ] Have ideas for improvement

---

## Phase 7: Serving

### Step 7.1: Create Inference Code

```python
# src/eno/inference/predictor.py
import torch
from eno.data.features import AudioFeatureExtractor
from eno.data.labels import LabelEncoder
from eno.models.classifier import SimpleCNN

class EnoPredictor:
    def __init__(self, model_path: str, label_encoder_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = AudioFeatureExtractor()
        self.label_encoder = LabelEncoder.load(label_encoder_path)

        self.model = SimpleCNN(n_mels=128, num_labels=len(self.label_encoder))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

    def predict(self, audio_path: str, threshold: float = 0.5):
        # Extract features
        features = self.feature_extractor.extract(audio_path)
        features = features.unsqueeze(0).to(self.device)  # Add batch dimension

        # Get predictions
        with torch.no_grad():
            logits = self.model(features)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Decode to labels
        return self.label_encoder.decode(probs, threshold=threshold)
```

### Step 7.2: Test Locally

```python
predictor = EnoPredictor("best_model.pt", "label_encoder.json")
result = predictor.predict("/path/to/test/song.flac")
print(result)
# {'genres': ['shoegaze', 'dream pop'], 'descriptors': ['melancholic', 'atmospheric']}
```

### Step 7.3: Deploy with ClearML Serving

```python
# scripts/serve.py
from clearml import Model
from clearml.serving import ServingService

# Register model
model = Model.create(
    name="eno-v1",
    project="eno/serving",
)
model.update_weights("best_model.pt")

# Deploy (ClearML Serving handles the rest)
# Access at https://clearml.home.shdr.ch/serving/eno
```

**✓ Final Checkpoint**:

- [ ] Model makes reasonable predictions on new audio
- [ ] Deployed and accessible via API
- [ ] Ready to iterate and improve! 🎉

---

## Quick Reference

### Key Concepts Glossary

| Term                | Meaning                                              |
| ------------------- | ---------------------------------------------------- |
| **Epoch**           | One full pass through all training data              |
| **Batch**           | Group of samples processed together                  |
| **Loss**            | How wrong the model is (lower = better)              |
| **Gradient**        | Direction to adjust weights to reduce loss           |
| **Overfitting**     | Model memorizes training data, fails on new data     |
| **Validation set**  | Data held out to check for overfitting               |
| **Mel spectrogram** | Visual representation of audio frequencies over time |

### Troubleshooting

| Issue                  | Check                                   |
| ---------------------- | --------------------------------------- |
| CUDA out of memory     | Reduce batch size                       |
| Loss is NaN            | Lower learning rate, check for bad data |
| Model predicts nothing | Check label encoding, lower threshold   |
| Training too slow      | Use DataLoader with `num_workers > 0`   |

### Resources to Learn More

- **PyTorch tutorials**: https://pytorch.org/tutorials/
- **Audio ML with torchaudio**: https://pytorch.org/audio/
- **ClearML docs**: https://clear.ml/docs/
- **Multi-label classification**: Search "multi-label classification deep learning"

---

_You've got this! 🎧_
