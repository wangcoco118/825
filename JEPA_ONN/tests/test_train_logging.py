import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from evals.intuitive_physics.train_optical import _configure_logging, _format_progress


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
        self.assertEqual(_format_progress(10, 40), "step=10/40 progress=25.0%")


if __name__ == "__main__":
    unittest.main()
