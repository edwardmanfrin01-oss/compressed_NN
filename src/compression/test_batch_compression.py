"""Checks for fresh compression timing and baseline reuse isolation."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import batch_compression as batch


class BatchTimingTests(unittest.TestCase):
    def test_no_reuse_compresses_every_png_without_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "images"
            inputs.mkdir()
            for name in ("image_00001.png", "image_00002.png"):
                (inputs / name).touch()
            executable = root / "compressor.exe"
            executable.touch()
            output = root / "fresh"
            argv = ["batch_compression.py", "--input-dir", str(inputs),
                    "--output-dir", str(output), "--gdcompress-bin", str(executable),
                    "--no-reuse", "--k-min", "128"]
            captured = io.StringIO()
            with patch("sys.argv", argv), patch.object(batch.subprocess, "run") as run, \
                    patch.object(batch.subprocess, "Popen") as helper, \
                    patch.object(batch.shutil, "copy2") as copy, \
                    patch.object(batch.time, "perf_counter", side_effect=[100, 160]), \
                    contextlib.redirect_stdout(captured):
                batch.main()
            self.assertEqual(run.call_count, 2)
            helper.assert_not_called()
            copy.assert_not_called()
            for call in run.call_args_list:
                self.assertEqual(call.args[0][-2:], ["--k-min", "128"])
            self.assertIn("Compressed: 2; copied: 0; skipped: 0", captured.getvalue())
            self.assertIn("60.000 seconds (1.000 minutes)", captured.getvalue())
            # A second run must fail before invoking the compressor.
            with patch("sys.argv", argv), patch.object(batch.subprocess, "run") as run:
                with self.assertRaisesRegex(SystemExit, "new or empty"):
                    batch.main()
                run.assert_not_called()

    def test_no_reuse_rejects_resume_and_explicit_baseline(self):
        for option in (["--resume"], ["--existing-dir", "baseline", "--k-min", "32"]):
            with patch("sys.argv", ["batch_compression.py", "--no-reuse", *option]), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    batch.parse_args()

    def test_default_kmin_still_reuses_baseline(self):
        with patch("sys.argv", ["batch_compression.py", "--k-min", "128"]):
            args = batch.parse_args()
        self.assertFalse(args.no_reuse)
        self.assertEqual(args.existing_dir, batch.PROJECT_ROOT / "data/cifar-10_compressed_1x1/test")


if __name__ == "__main__":
    unittest.main()
