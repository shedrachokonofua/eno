# Lute Data Extraction

Extract album metadata from lute → Parquet.

## Usage

```bash
task extract          # Full extraction
task reset-cursor     # Start fresh
task stats            # Label counts
task lute:grpcurl -- list  # Inspect API
```

## Output

`/mnt/nfs/nvme/data/ml/eno/raw/metadata/albums.parquet`

```python
{
    "file_name": str,           # RYM ID
    "name": str,
    "rating": float,
    "rating_count": int,
    "artists": list[dict],
    "primary_genres": list[str],
    "secondary_genres": list[str],
    "descriptors": list[str],
    "tracks": list[dict],
    "release_date": str,
    "spotify_id": str,
}
```

## Config

Edit `scripts/extract_from_lute.py`:

```python
LUTE_HOST = "core.lute.home.shdr.ch"
LUTE_PORT = 443
LUTE_USE_TLS = True
SUBSCRIBER_ID = "eno-extractor"
BATCH_SIZE = 100
MAX_ALBUMS = None
```

Then `task build && task extract`.

## How It Works

1. Stream `album_saved` events from lute EventService
2. Batch collect file names (100 at a time)
3. Fetch full albums via `GetManyAlbums` (100x fewer RPCs than individual fetches)
4. Save as Parquet
