import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from evals.intuitive_physics import train_optical
from src.models.fsonn import FeedbackFSONN, ONNConfig
from evals.intuitive_physics.train_optical import (
    _configure_logging,
    _format_jepa_batch_log,
    _format_progress,
)


class TrainLoggingTests(unittest.TestCase):
    def test_logger_writes_same_record_to_stdout_and_file(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "train.log"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                logger = _configure_logging(str(log_path))
                logger.info("step=1 loss=0.25 elapsed_s=1.5")
                for handler in logger.handlers:
                    handler.flush()

            self.assertIn("step=1 loss=0.25 elapsed_s=1.5", stdout.getvalue())
            self.assertIn("step=1 loss=0.25 elapsed_s=1.5", log_path.read_text(encoding="utf-8"))


    def test_progress_format_uses_completed_step_and_stage_total(self):
        self.assertEqual(
            _format_progress(10, 40),
            "[#####---------------] 25.0% (10/40)",
        )

    def test_batch_log_contains_only_dynamic_training_fields(self):
        message = _format_jepa_batch_log(
            epoch=0,
            stage="val",
            step=1,
            total_steps=4,
            mask_mode="unified_random",
            batch_size=20,
            n_ctxt=196,
            n_tgt=1372,
            covered_count=1568,
            missing_count=0,
            loss=0.713809,
            grad_norm=0.0,
            time_s=1.561,
        )

        self.assertEqual(
            message,
            "epoch=0 stage=val mask_mode=unified_random "
            "[#####---------------] 25.0% (1/4) batch=20 "
            "n_ctxt=196 n_tgt=1372 covered_count=1568 "
            "missing_count=0 loss=0.713809 grad_norm=0.000 time=1.561s",
        )
        for repeated_field in (
            "ctxt_shape",
            "onn_shape",
            "chunk=",
            "feedback_layer",
            "lr=",
            "batch_time_s",
            "epoch_time_s",
        ):
            self.assertNotIn(repeated_field, message)


    def test_feedback_metadata_and_checkpoint_use_resolved_independent_model(self):
        self.assertTrue(hasattr(train_optical, "_feedback_runtime_metadata"))
        config = ONNConfig.from_mapping(
            {
                "input_dim": 2,
                "output_dim": 2,
                "num_slm_layers": 5,
                "chunk_tokens": 2,
                "grid_height": 2,
                "grid_width": 2,
                "feedback_layer_mode": "multi",
                "feedback_layer_indices": [2, 3, 4],
                "feedback_gain_mode": "independent",
                "feedback_gain_init": [0.5, 1.5, 3.0],
                "feedback_phase_max_rad": 0.75,
                "slm_intervals_um": [8.0, 8.0, 8.0, 8.0],
                "input_to_first_slm_um": 8.0,
                "last_slm_to_detector_um": 8.0,
                "asm_padding_factor": 1.0,
            }
        )

        class PredictorStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.onn_core = FeedbackFSONN(config)
                self.feedback_mode = "fixed_middle_phase"

        predictor = PredictorStub()
        metadata = train_optical._feedback_runtime_metadata(predictor)

        self.assertEqual(metadata["feedback_layer_mode"], "multi")
        self.assertNotIn("feedback_layer_index", metadata)
        self.assertEqual(metadata["feedback_layer_indices"], [2, 3, 4])
        self.assertEqual(metadata["physical_feedback_layers"], [3, 4, 5])
        self.assertEqual(metadata["feedback_gain_mode"], "independent")
        self.assertEqual(metadata["feedback_gain_parameter_count"], 3)
        self.assertTrue(
            torch.allclose(
                torch.tensor(metadata["effective_feedback_gains"]),
                torch.tensor([0.5, 1.5, 3.0]),
                atol=1e-6,
            )
        )
        message = train_optical._format_feedback_metadata(metadata)
        self.assertIn("feedback_layer_mode=multi", message)
        self.assertIn("SLM3_K=0.500000", message)
        self.assertIn("SLM4_K=1.500000", message)
        self.assertIn("SLM5_K=3.000000", message)

        optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-3)
        checkpoint = train_optical._end_to_end_checkpoint(
            predictor,
            optimizer,
            None,
            1,
            1,
            0.5,
            {"train_video_ids": ["a"], "val_video_ids": ["b"]},
            "/tmp/split.json",
            {
                "pretrain": {"folder": "/checkpoint", "checkpoint": "official.pt"},
                "training": {"mode": "onn_feedback"},
            },
            "best",
            experiment_mode="onn_feedback",
        )
        for key, value in metadata.items():
            self.assertEqual(checkpoint[key], value)



if __name__ == "__main__":
    unittest.main()
