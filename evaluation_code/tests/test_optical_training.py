import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evals.intuitive_physics.intphys_dataset import IntPhysDataset
from evals.intuitive_physics.train_optical import (
    _apply_cli_overrides,
    _compute_jepa_loss,
    _end_to_end_checkpoint,
    _extract_jepa_clips,
    _last_checkpoint_path,
    _load_end_to_end_checkpoint,
    _save_checkpoint,
)
from evals.intuitive_physics.optical_split import (
    build_video_split,
    load_or_create_video_split,
)


class OpticalSplitTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_scene_level(self):
        video_ids = ["00001", "00002", "00003", "00004", "00005"]
        first = build_video_split(video_ids, 2, 2, split_seed=42)
        second = build_video_split(list(reversed(video_ids)), 2, 2, split_seed=42)

        self.assertEqual(first["train_video_ids"], second["train_video_ids"])
        self.assertEqual(first["val_video_ids"], second["val_video_ids"])
        self.assertTrue(
            set(first["train_video_ids"]).isdisjoint(first["val_video_ids"])
        )
        self.assertEqual(
            set(first["train_video_ids"])
            | set(first["val_video_ids"])
            | set(first["unused_video_ids"]),
            set(video_ids),
        )

    def test_manifest_is_reused_even_if_directory_order_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "train"
            root.mkdir()
            for video_id in ("00001", "00002", "00003", "00004"):
                (root / video_id).mkdir()
            manifest = Path(directory) / "split.json"

            first = load_or_create_video_split(
                root, manifest, num_train_videos=2, num_val_videos=1, split_seed=42
            )
            (root / "00005").mkdir()
            second = load_or_create_video_split(
                root, manifest, num_train_videos=2, num_val_videos=1, split_seed=999
            )

            self.assertEqual(first, second)
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(saved["train_video_ids"], first["train_video_ids"])

    def test_split_rejects_insufficient_videos(self):
        with self.assertRaisesRegex(ValueError, "requires 4 videos"):
            build_video_split(["00001", "00002", "00003"], 2, 2, split_seed=42)


class CliOverrideTests(unittest.TestCase):
    def test_batch_size_cli_override_updates_data_config(self):
        config = {"data": {"batch_size": 20}}
        updated = _apply_cli_overrides(config, batch_size=5)
        self.assertEqual(updated["data"]["batch_size"], 5)


class JepaLossTests(unittest.TestCase):
    def test_jepa_loss_matches_official_mean_absolute_formula(self):
        predictions = [torch.tensor([[1.0, 3.0]]), torch.tensor([[2.0, 8.0]])]
        targets = [torch.tensor([[0.0, 1.0]]), torch.tensor([[4.0, 4.0]])]
        expected = torch.tensor((1.5 + 3.0) / 2.0)
        actual = _compute_jepa_loss(predictions, targets, loss_exp=1.0)
        self.assertTrue(torch.allclose(actual, expected))


class EndToEndJepaTests(unittest.TestCase):
    def test_one_video_batch_is_one_16_frame_clip(self):
        clips = torch.zeros(2, 1, 3, 16, 4, 4)
        labels = torch.zeros(2, 1)
        actual = _extract_jepa_clips((clips, labels), torch.device("cpu"))
        self.assertEqual(tuple(actual.shape), (2, 3, 16, 4, 4))

    def test_full_predictor_checkpoint_is_distinct_from_legacy_checkpoint(self):
        predictor = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-3)
        config = {
            "pretrain": {"folder": "/checkpoint", "checkpoint": "official.pt"},
            "training": {"mode": "end_to_end_jepa"},
            "optical_qkv": {"qkv_backend": "fsonn_tdm"},
        }
        split = {"train_video_ids": ["a"], "val_video_ids": ["b"]}
        checkpoint = _end_to_end_checkpoint(
            predictor, optimizer, None, 1, 1, 0.5, split,
            "/tmp/split.json", config, "best"
        )
        self.assertEqual(checkpoint["mode"], "end_to_end_jepa")
        self.assertEqual(set(checkpoint["predictor"]), set(predictor.state_dict()))
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "optical_best.pt"
            torch.save({"optical_state_dict": {}}, legacy)
            with self.assertRaisesRegex(ValueError, "end_to_end_jepa"):
                _load_end_to_end_checkpoint(
                    legacy, predictor, optimizer, None
                )


class CheckpointTests(unittest.TestCase):
    def test_last_checkpoint_has_stable_suffix_and_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            best_path = Path(directory) / "optical_best.pt"
            last_path = _last_checkpoint_path(best_path)
            self.assertEqual(last_path, Path(directory) / "optical_best.last.pt")

            _save_checkpoint({"epoch": 50}, last_path)
            saved = torch.load(last_path, weights_only=False)
            self.assertEqual(saved["epoch"], 50)


class TrainDatasetTests(unittest.TestCase):
    def test_train_mode_filters_video_ids_before_window_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for video_id in ("00001", "00002"):
                video_root = root / video_id
                (video_root / "scene").mkdir(parents=True)
                (video_root / "status.json").write_text(
                    json.dumps({"header": {"is_possible": True}}),
                    encoding="utf-8",
                )
                for frame_index in range(1, 11):
                    image = np.full((4, 4, 3), frame_index, dtype=np.uint8)
                    Image.fromarray(image).save(
                        video_root / "scene" / f"scene_{frame_index:03d}.png"
                    )

            transform = lambda clip: clip.permute(3, 0, 1, 2).float()
            dataset = IntPhysDataset(
                root,
                frames_per_clip=3,
                frame_step=2,
                transform=transform,
                video_ids=["00002"],
                train_format=True,
            )

            clips, labels = dataset[0]
            self.assertEqual(dataset.scenes, ["00002"])
            self.assertEqual(tuple(clips.shape), (1, 3, 3, 4, 4))
            self.assertEqual(tuple(labels.shape), (1,))

            full_video_dataset = IntPhysDataset(
                root,
                frames_per_clip=99,
                frame_step=2,
                transform=transform,
                video_ids=["00002"],
                train_format=True,
            )
            full_video_clips, _ = full_video_dataset[0]
            self.assertEqual(tuple(full_video_clips.shape), (1, 3, 5, 4, 4))


if __name__ == "__main__":
    unittest.main()

