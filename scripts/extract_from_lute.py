#!/usr/bin/env python3
"""
Extract album data from lute event stream and save as Parquet.

Listens to AlbumSavedEvents, fetches full album details, and writes to Parquet.
Uses functional style with modern Python features.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import grpc
import pandas as pd
from tqdm import tqdm

import lute_pb2
import lute_pb2_grpc

# Configuration
LUTE_HOST = "core.lute.home.shdr.ch"
LUTE_PORT = 443
LUTE_USE_TLS = True
SUBSCRIBER_ID = "eno-extractor"
BATCH_SIZE = 100
MAX_ALBUMS = None
OUTPUT_PATH = Path("/output/albums.parquet")


def create_channel(host: str, port: int, *, use_tls: bool = True) -> grpc.Channel:
    """Create gRPC channel to lute."""
    target = f"{host}:{port}"

    if use_tls:
        credentials = grpc.ssl_channel_credentials()
        return grpc.secure_channel(target, credentials)

    return grpc.insecure_channel(target)


def album_to_dict(album: lute_pb2.Album) -> dict[str, Any]:
    """Convert Album protobuf to dictionary."""
    return {
        "name": album.name,
        "file_name": album.file_name,
        "rating": album.rating,
        "rating_count": album.rating_count,
        "artists": [
            {"name": a.name, "file_name": a.file_name}
            for a in album.artists
        ],
        "primary_genres": list(album.primary_genres),
        "secondary_genres": list(album.secondary_genres),
        "descriptors": list(album.descriptors),
        "tracks": [
            {
                "name": t.name,
                "duration_seconds": t.duration_seconds if t.HasField("duration_seconds") else None,
                "rating": t.rating if t.HasField("rating") else None,
                "position": t.position if t.HasField("position") else None,
            }
            for t in album.tracks
        ],
        "release_date": album.release_date if album.HasField("release_date") else None,
        "languages": list(album.languages),
        "cover_image_url": album.cover_image_url if album.HasField("cover_image_url") else None,
        "duplicate_of": album.duplicate_of if album.HasField("duplicate_of") else None,
        "duplicates": list(album.duplicates),
        "spotify_id": album.spotify_id if album.HasField("spotify_id") else None,
        "credits": [
            {
                "artist": {"name": c.artist.name, "file_name": c.artist.file_name},
                "roles": list(c.roles),
            }
            for c in album.credits
        ],
    }


def fetch_albums_batch(channel: grpc.Channel, file_names: list[str]) -> list[dict[str, Any]]:
    """Fetch multiple albums at once using GetManyAlbums."""
    if not file_names:
        return []

    stub = lute_pb2_grpc.AlbumServiceStub(channel)
    request = lute_pb2.GetManyAlbumsRequest(file_names=file_names)

    try:
        response = stub.GetManyAlbums(request)
        return [album_to_dict(album) for album in response.albums]
    except grpc.RpcError as e:
        print(f"Failed to fetch batch of {len(file_names)} albums: {e.code()}", file=sys.stderr)
        return []


def get_stream_tail(channel: grpc.Channel, stream_id: str) -> str:
    """Get the tail cursor for a stream from the monitor."""
    from google.protobuf import empty_pb2
    stub = lute_pb2_grpc.EventServiceStub(channel)
    response = stub.GetMonitor(empty_pb2.Empty())
    for stream in response.monitor.streams:
        if stream.id == stream_id:
            return stream.tail
    return ""


def stream_album_events(
    channel: grpc.Channel,
    *,
    subscriber_id: str = "eno-extractor",
    batch_size: int = 100,
) -> Iterator[str]:
    """Stream album_saved events and yield file names."""
    stub = lute_pb2_grpc.EventServiceStub(channel)

    # Get the current tail of the stream so we know when to stop
    stream_tail = get_stream_tail(channel, "album")
    print(f"Stream tail: {stream_tail}")

    cursor = None

    try:
        while True:
            # Create request with current cursor
            request = lute_pb2.EventStreamRequest(
                stream_id="album",
                subscriber_id=subscriber_id,
                max_batch_size=batch_size,
            )
            if cursor:
                request.cursor = cursor

            # Get one batch
            response = stub.Stream(iter([request]))
            batch = next(response, None)

            if not batch:
                break

            # Update cursor
            cursor = batch.cursor

            # Yield file names from this batch
            for item in batch.items:
                event = item.payload.event
                if event.HasField("album_saved"):
                    yield event.album_saved.file_name

            # Check if we've reached the tail (compare as integers)
            if cursor and int(cursor) >= int(stream_tail):
                break

    except grpc.RpcError as e:
        print(f"Stream error: {e.code()}: {e.details()}", file=sys.stderr)


def extract_albums_from_stream(
    host: str,
    port: int,
    *,
    use_tls: bool = True,
    subscriber_id: str = "eno-extractor",
    batch_size: int = 100,
    max_albums: int | None = None,
) -> list[dict[str, Any]]:
    """Extract albums by listening to event stream and batch fetching details."""
    print(f"Connecting to lute at {host}:{port} (TLS: {use_tls})")
    print(f"Listening to album stream (subscriber: {subscriber_id})...")
    print(f"Batch fetching {batch_size} albums at a time")

    with create_channel(host, port, use_tls=use_tls) as channel:
        albums = []
        file_name_batch = []

        with tqdm(desc="Processing events", unit="albums") as pbar:
            for file_name in stream_album_events(channel, subscriber_id=subscriber_id):
                file_name_batch.append(file_name)

                # Fetch in batches for efficiency
                if len(file_name_batch) >= batch_size:
                    album_data = fetch_albums_batch(channel, file_name_batch)
                    albums.extend(album_data)
                    pbar.update(len(album_data))
                    file_name_batch.clear()

                if max_albums and len(albums) >= max_albums:
                    break

            # Fetch remaining albums in final batch
            if file_name_batch and (not max_albums or len(albums) < max_albums):
                album_data = fetch_albums_batch(channel, file_name_batch)
                albums.extend(album_data)
                pbar.update(len(album_data))

        print(f"\nExtracted {len(albums)} albums from event stream")
        return albums


def save_parquet(albums: list[dict[str, Any]], output_path: Path) -> None:
    """Save albums as Parquet file."""
    if not albums:
        print("No albums to save", file=sys.stderr)
        return

    print(f"Saving {len(albums)} albums to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(albums)
    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    # Print summary
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved to {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Albums: {len(df)}")

    # Label stats
    total_primary = sum(len(x) for x in df["primary_genres"])
    total_secondary = sum(len(x) for x in df["secondary_genres"])
    total_descriptors = sum(len(x) for x in df["descriptors"])

    print("\nLabel Distribution:")
    print(f"  Primary genres: {total_primary}")
    print(f"  Secondary genres: {total_secondary}")
    print(f"  Descriptors: {total_descriptors}")


def main() -> None:
    print(f"Configuration:")
    print(f"  Host: {LUTE_HOST}:{LUTE_PORT}")
    print(f"  TLS: {LUTE_USE_TLS}")
    print(f"  Subscriber: {SUBSCRIBER_ID}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max albums: {MAX_ALBUMS or 'unlimited'}")
    print(f"  Output: {OUTPUT_PATH}")
    print()

    # Extract from event stream
    albums = extract_albums_from_stream(
        LUTE_HOST,
        LUTE_PORT,
        use_tls=LUTE_USE_TLS,
        subscriber_id=SUBSCRIBER_ID,
        batch_size=BATCH_SIZE,
        max_albums=MAX_ALBUMS,
    )

    if not albums:
        print("No albums extracted", file=sys.stderr)
        sys.exit(1)

    # Save to Parquet
    save_parquet(albums, OUTPUT_PATH)


if __name__ == "__main__":
    main()
