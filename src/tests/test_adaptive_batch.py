import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / 'src/batch_network_features.py'

@unittest.skipUnless(
    (ROOT / 'tmp/manual-test').is_dir()
    and any((ROOT / 'tmp/manual-test').glob('*.igd'))
    and (ROOT / 'src/gdcompress/target/release/igd_extract.exe').is_file(),
    'Requires local IGD smoke fixtures and the release extractor',
)
class AdaptiveBatchTests(unittest.TestCase):
    def test_export_resume_and_version_guards(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            output = Path(directory)
            command = [sys.executable, str(BATCH), '--version', 'v3', '--input-dir',
                       str(ROOT / 'tmp/manual-test'), '--output-dir', str(output), '--verify']
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            report = output / 'scales_v3.csv'
            original = report.read_bytes()
            with report.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows)
            for row in rows:
                with ZipFile(output / row['output']) as archive:
                    metadata = json.loads(archive.read('metadata.json'))
                self.assertEqual(int(row['S_i']), metadata['representation']['scale'])
                self.assertLessEqual(int(row['z_max']), 65535)
                self.assertTrue(metadata['verification']['passed'])
            resumed = subprocess.run(command + ['--resume'], capture_output=True, text=True)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(original, report.read_bytes())
            rejected = subprocess.run(command + ['--scale', '3'], capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            mismatch = command.copy()
            mismatch[mismatch.index('v3')] = 'v2'
            rejected = subprocess.run(mismatch + ['--resume'], capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('representation differs', rejected.stderr)

if __name__ == '__main__':
    unittest.main()
