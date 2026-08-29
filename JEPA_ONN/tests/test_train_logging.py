import contextlib
import io
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
