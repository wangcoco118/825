"""Deterministic scene-level splits for optical distillation training."""

import json
import random
from pathlib import Path
from typing import Iterable


def list_video_ids(data_root) -> list[str]:
    """List complete Train video directories before any temporal windows are made."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Train data root does not exist: {root}")
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def build_video_split(
    video_ids: Iterable[str],
    num_train_videos: int,
    num_val_videos: int,
    split_seed: int,
) -> dict:
    video_ids = sorted({str(video_id) for video_id in video_ids})
    if num_train_videos < 0 or num_val_videos < 0:
        raise ValueError("num_train_videos and num_val_videos must be non-negative")
    required = num_train_videos + num_val_videos
    if len(video_ids) < required:
        raise ValueError(
            f"split requires {required} videos but only {len(video_ids)} are available"
        )

    shuffled = list(video_ids)
    random.Random(split_seed).shuffle(shuffled)
    train_video_ids = shuffled[:num_train_videos]
    val_video_ids = shuffled[num_train_videos:required]
    unused_video_ids = shuffled[required:]
    return {
        "source": "train",
        "split_seed": int(split_seed),
        "num_available_videos": len(video_ids),
        "num_train_videos": int(num_train_videos),
        "num_val_videos": int(num_val_videos),
        "train_video_ids": train_video_ids,
        "val_video_ids": val_video_ids,
        "unused_video_ids": unused_video_ids,
    }


def _validate_manifest(manifest: dict, available_video_ids: set[str]) -> dict:
    train_video_ids = [str(video_id) for video_id in manifest.get("train_video_ids", [])]
    val_video_ids = [str(video_id) for video_id in manifest.get("val_video_ids", [])]
    if set(train_video_ids).intersection(val_video_ids):
        raise ValueError("split manifest contains overlapping train and validation IDs")
    selected = set(train_video_ids) | set(val_video_ids)
    missing = sorted(selected - available_video_ids)
    if missing:
        raise ValueError(
            "split manifest refers to missing Train video IDs: " + ",".join(missing)
        )
    manifest["train_video_ids"] = train_video_ids
    manifest["val_video_ids"] = val_video_ids
    manifest.setdefault("unused_video_ids", sorted(available_video_ids - selected))
    return manifest


def require_existing_video_split(
    data_root,
    manifest_path,
    num_train_videos: int,
    num_val_videos: int,
    split_seed: int,
) -> dict:
    """Load a persisted split and reject missing manifests."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"split manifest does not exist; refusing to create a random split: "
            f"{manifest_path}"
        )
    return load_or_create_video_split(
        data_root,
        manifest_path,
        num_train_videos=num_train_videos,
        num_val_videos=num_val_videos,
        split_seed=split_seed,
    )


def load_or_create_video_split(
    data_root,
    manifest_path,
    num_train_videos: int,
    num_val_videos: int,
    split_seed: int,
) -> dict:
    """Reuse a persisted split; otherwise create it from complete video directories."""
    data_root = Path(data_root)
    manifest_path = Path(manifest_path)
    available_video_ids = set(list_video_ids(data_root))
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            return _validate_manifest(json.load(handle), available_video_ids)

    split = build_video_split(
        available_video_ids,
        num_train_videos=num_train_videos,
        num_val_videos=num_val_videos,
        split_seed=split_seed,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return split

