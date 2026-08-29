import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evals.intuitive_physics.intphys_dataset import IntPhysDataset
from evals.intuitive_physics import eval as dev_eval
from evals.intuitive_physics.train_optical import (
    _apply_cli_overrides,
    _compute_jepa_loss,
    _compute_distillation_loss,
    _end_to_end_checkpoint,
    _resolve_distillation_config,
    _resolve_experiment_mode,
    _should_run_internal_validation,
    _extract_jepa_clips,
    _last_checkpoint_path,
    _load_end_to_end_checkpoint,
    _save_checkpoint,
)
from evals.intuitive_physics.optical_split import (
    build_video_split,
    load_or_create_video_split,
)
from src.models.predictor import VisionTransformerPredictor
from src.models.utils.modules import Block
from src.models.optical_distillation import freeze_stage_one
from src.utils.transforms import VideoTransform


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

    def test_target_node_cli_override_updates_distillation_config(self):
        config = {"distillation": {"target_node": "qkv"}}
        updated = _apply_cli_overrides(
            config, batch_size=10, target_node="attention_output"
        )
        self.assertEqual(updated["data"]["batch_size"], 10)
        self.assertEqual(
            updated["distillation"]["target_node"], "attention_output"
        )


class WarningCleanupTests(unittest.TestCase):
    def test_tensor_video_transform_does_not_copy_construct_warning(self):
        transform = VideoTransform(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0),
            crop_size=4,
        )
        buffer = torch.zeros(2, 4, 4, 3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            transformed = transform(buffer)
        self.assertEqual(tuple(transformed.shape), (3, 2, 4, 4))
        self.assertFalse(
            any('copy construct from a tensor' in str(item.message) for item in caught)
        )

    def test_bfloat16_flag_uses_bfloat16_autocast(self):
        try:
            from src.utils.amp import autocast_context
        except ImportError:
            self.fail('shared autocast helper is missing')
        with autocast_context(torch.device('cpu'), True):
            self.assertEqual(torch.get_autocast_dtype('cpu'), torch.bfloat16)


class JepaLossTests(unittest.TestCase):
    def test_jepa_loss_matches_official_mean_absolute_formula(self):
        predictions = [torch.tensor([[1.0, 3.0]]), torch.tensor([[2.0, 8.0]])]
        targets = [torch.tensor([[0.0, 1.0]]), torch.tensor([[4.0, 4.0]])]
        expected = torch.tensor((1.5 + 3.0) / 2.0)
        actual = _compute_jepa_loss(predictions, targets, loss_exp=1.0)
        self.assertTrue(torch.allclose(actual, expected))


class ExperimentModeTests(unittest.TestCase):
    def test_legacy_qkv_distill_aliases_to_realtime_last_node_distillation(self):
        self.assertEqual(
            _resolve_experiment_mode(
                {"training": {"experiment_mode": "qkv_distill"}}
            ),
            "realtime_last_node_distillation",
        )

    def test_experiment_mode_selects_electronic_or_optical_control(self):
        self.assertEqual(
            _resolve_experiment_mode(
                {"training": {"experiment_mode": "electronic_control"}}
            ),
            "electronic_control",
        )
        self.assertEqual(
            _resolve_experiment_mode(
                {"training": {"experiment_mode": "optical_qkv"}}
            ),
            "optical_qkv",
        )
        self.assertEqual(
            _resolve_experiment_mode(
                {"training": {"mode": "end_to_end_jepa"}}
            ),
            "optical_qkv",
        )


        self.assertEqual(
            _resolve_experiment_mode(
                {"training": {"mode": "realtime_last_node_distillation"}}
            ),
            "realtime_last_node_distillation",
        )


class EvaluationCheckpointModeTests(unittest.TestCase):
    def test_electronic_control_checkpoint_is_accepted_as_full_predictor(self):
        is_compatible = getattr(
            dev_eval, '_is_full_predictor_checkpoint_mode', lambda mode: False
        )
        self.assertTrue(is_compatible('electronic_control'))
        self.assertTrue(is_compatible('end_to_end_jepa'))
        self.assertFalse(is_compatible('unknown_mode'))


class RealtimeDistillationNodeTests(unittest.TestCase):
    def test_block_captures_all_four_block_nodes_without_detaching_state(self):
        block = Block(
            dim=8,
            num_heads=2,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            use_sdpa=False,
        )
        inputs = torch.randn(2, 5, 8, requires_grad=True)
        output, captured = block(
            inputs,
            mask=None,
            capture_nodes={
                "qkv",
                "attention_output",
                "post_output",
                "block_output",
            },
        )
        self.assertEqual(tuple(captured["qkv"].shape), (2, 5, 24))
        self.assertEqual(tuple(captured["attention_output"].shape), (2, 5, 8))
        self.assertEqual(tuple(captured["post_output"].shape), (2, 5, 8))
        self.assertEqual(tuple(captured["block_output"].shape), (2, 5, 8))
        self.assertTrue(output.requires_grad)
        captured["block_output"].sum().backward()
        self.assertIsNotNone(inputs.grad)


class RealtimeDistillationPredictorTests(unittest.TestCase):
    def test_predictor_forward_with_nodes_keeps_serial_graph(self):
        predictor = VisionTransformerPredictor(
            img_size=4,
            patch_size=2,
            num_frames=2,
            tubelet_size=1,
            embed_dim=8,
            predictor_embed_dim=8,
            depth=2,
            num_heads=2,
            use_mask_tokens=True,
            use_sdpa=False,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
        )
        predictor.eval()
        context = torch.randn(1, 4, 8)
        target = torch.randn(1, 4, 8)
        masks_ctxt = [torch.tensor([[0, 1, 2, 3]])]
        masks_tgt = [torch.tensor([[4, 5, 6, 7]])]
        output, nodes = predictor.forward_with_nodes(
            context,
            target,
            masks_ctxt,
            masks_tgt,
            target_node="block_output",
        )
        self.assertEqual(tuple(output.shape), (1, 4, 8))
        self.assertEqual(len(nodes["block_output"]), 2)
        self.assertTrue(nodes["block_output"][-1].requires_grad)

    def test_qkv_no_bias_capture_does_not_change_teacher_block_state(self):
        block = Block(
            dim=8,
            num_heads=2,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            use_sdpa=False,
        )
        block.eval()
        inputs = torch.randn(1, 5, 8)
        expected = block(inputs)
        actual, captured = block(
            inputs,
            capture_nodes={"qkv"},
            qkv_include_bias=False,
        )
        expected_qkv = torch.nn.functional.linear(
            block.norm1(inputs),
            block.attn.qkv.weight,
            bias=None,
        )
        self.assertTrue(torch.allclose(captured["qkv"], expected_qkv))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))


    def test_all_target_nodes_have_expected_shapes_and_upstream_optical_grads(self):
        class TinyOpticalQKV(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(8, 24)

            def forward(self, value):
                return self.proj(value)

        def make_student():
            model = VisionTransformerPredictor(
                img_size=4,
                patch_size=2,
                num_frames=2,
                tubelet_size=1,
                embed_dim=8,
                predictor_embed_dim=8,
                depth=2,
                num_heads=2,
                use_mask_tokens=True,
                use_sdpa=False,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
            )
            for block in model.predictor_blocks:
                block.attn.qkv = None
                block.attn.qkv_backend = "fsonn_tdm"
                block.attn.optical_qkv = TinyOpticalQKV()
            freeze_stage_one(model)
            model.eval()
            return model

        context = torch.randn(1, 4, 8)
        target = torch.randn(1, 4, 8)
        masks_ctxt = [torch.tensor([[0, 1, 2, 3]])]
        masks_tgt = [torch.tensor([[4, 5, 6, 7]])]
        expected_shapes = {
            "qkv": (1, 8, 24),
            "attention_output": (1, 8, 8),
            "post_output": (1, 8, 8),
            "block_output": (1, 8, 8),
            "predictor_output": (1, 4, 8),
        }
        for target_node, expected_shape in expected_shapes.items():
            model = make_student()
            _, nodes = model.forward_with_nodes(
                context,
                target,
                masks_ctxt,
                masks_tgt,
                target_node=target_node,
            )
            selected = nodes[target_node]
            if isinstance(selected, list):
                self.assertEqual(len(selected), 2)
                selected = selected[-1]
            self.assertEqual(tuple(selected.shape), expected_shape)
            selected.square().mean().backward()
            optical_grads = [
                parameter.grad
                for name, parameter in model.named_parameters()
                if "optical_qkv" in name
            ]
            self.assertTrue(optical_grads)
            self.assertTrue(
                all(
                    gradient is not None
                    and torch.isfinite(gradient).all()
                    and gradient.abs().sum() > 0
                    for gradient in optical_grads
                )
            )
            self.assertTrue(
                all(
                    parameter.grad is None
                    for name, parameter in model.named_parameters()
                    if "optical_qkv" not in name
                )
            )


class RealtimeTrainingScheduleTests(unittest.TestCase):
    def test_realtime_distillation_has_no_internal_validation(self):
        realtime = {
            "training": {
                "experiment_mode": "realtime_last_node_distillation"
            }
        }
        optical = {"training": {"experiment_mode": "optical_qkv"}}
        self.assertFalse(_should_run_internal_validation(realtime))
        self.assertTrue(_should_run_internal_validation(optical))


class RealtimeDistillationConfigTests(unittest.TestCase):
    def test_distillation_config_accepts_one_target_node(self):
        config = {
            "training": {
                "experiment_mode": "realtime_last_node_distillation"
            },
            "distillation": {
                "target_node": "qkv",
                "optimization_scope": "last_layer",
                "log_all_layers": True,
                "cosine_loss_weight": 0.1,
            },
        }
        resolved = _resolve_distillation_config(config)
        self.assertEqual(resolved["target_node"], "qkv")
        self.assertEqual(resolved["optimization_scope"], "last_layer")

    def test_distillation_flag_cannot_enable_old_optical_mode(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            _resolve_experiment_mode(
                {
                    "training": {"experiment_mode": "optical_qkv"},
                    "distillation": {"enabled": True},
                }
            )

    def test_distillation_loss_returns_nmse_cosine_and_combined_loss(self):
        student = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        teacher = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        total, nmse, cosine = _compute_distillation_loss(
            student, teacher, cosine_loss_weight=0.1
        )
        self.assertTrue(torch.allclose(nmse, torch.tensor(2.0)))
        self.assertTrue(torch.allclose(cosine, torch.tensor(1.0)))
        self.assertTrue(torch.allclose(total, torch.tensor(2.1)))


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
        self.assertEqual(checkpoint["config"], config)
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

