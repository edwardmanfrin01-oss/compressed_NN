import struct
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from audit_compression import describe, image_id, index_npz, read_header, read_npz_sizes, summarize


def make_npz(path, shape=(2, 3), compression=ZIP_STORED):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = repr(dict(descr='<u2', fortran_order=False, shape=shape)).encode() + b'\n'
    array = b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + bytes(shape[0] * shape[1] * 2)
    with ZipFile(path, 'w', compression=compression) as archive:
        archive.writestr('z_u16.npy', array)
        archive.writestr('metadata.json', '{}')


class AuditTests(unittest.TestCase):
    def test_npz_stored_and_compressed_tensor_sizes_and_shape(self):
        with tempfile.TemporaryDirectory() as folder:
            for compression in (ZIP_STORED, ZIP_DEFLATED):
                path = Path(folder) / f'{compression}.npz'
                make_npz(path, compression=compression)
                result = read_npz_sizes(path, 3, 2)
                self.assertEqual(result['tensor_bytes'], 12)
                self.assertEqual(result['npz_bytes'], path.stat().st_size)
                with self.assertRaises(ValueError):
                    read_npz_sizes(path, 2, 3)

    def test_npz_identity_keeps_splits_separate_and_rejects_aliases(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ('train', 'test'):
                make_npz(root / split / 'image_00001.network.npz')
            self.assertEqual(len(index_npz(root)), 2)
            make_npz(root / 'train/image_00001.npz')
            with self.assertRaises(ValueError):
                index_npz(root)

    def test_cli_missing_and_unmatched_npz_and_optional_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            header = b'IGD\x01' + struct.pack('<II4BIIQ', 3, 2, 3, 0, 2, 1, 1, 1, 0)
            for split in ('train', 'test'):
                path = root / 'igd' / split / 'image_00001.igd'
                path.parent.mkdir(parents=True)
                path.write_bytes(header)
            make_npz(root / 'npz/train/image_00001.network.npz')
            make_npz(root / 'npz/test/image_00002.network.npz')
            script = Path(__file__).with_name('audit_compression.py')
            command = [sys.executable, '-B', str(script), '--igd-root', str(root / 'igd'),
                       '--original-root', str(root / 'original')]
            subprocess.run(command + ['--npz-root', str(root / 'npz'), '--output-dir', str(root / 'audit')],
                           check=True, capture_output=True, text=True)
            report = json.loads((root / 'audit/summary.json').read_text())
            group = report['groups']['all']
            self.assertEqual(group['npz_comparison_count'], 1)
            self.assertEqual(group['tensor_bytes'], 12)
            self.assertEqual(group['npz_paired_raw_bytes'], 18)
            self.assertEqual(report['missing_npz'], ['test/image_00001.network.npz'])
            self.assertEqual(report['unmatched_npz'], ['test/image_00002.network.npz'])
            self.assertIn('NPZ feature storage', (root / 'audit/report.md').read_text())
            subprocess.run(command + ['--output-dir', str(root / 'legacy')],
                           check=True, capture_output=True, text=True)
            legacy = json.loads((root / 'legacy/summary.json').read_text())
            self.assertNotIn('npz_bytes', legacy['groups']['all'])
            self.assertNotIn('missing_npz', legacy)

    def test_header_sizes_and_truncation(self):
        header = b"IGD\x01" + struct.pack("<II4BIIQ", 224, 224, 3, 0, 2, 1, 1, 1, 100)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.igd"
            path.write_bytes(header + bytes(100))
            row = read_header(path)
            self.assertEqual(row["raw_bytes"], 150528)
            self.assertEqual(row["igd_bytes"], 132)
            self.assertAlmostEqual(row["saving_raw_pct"], 100 * (1 - 132 / 150528))
            path.write_bytes(header + bytes(99))
            with self.assertRaises(ValueError):
                read_header(path)

    def test_aggregate_uses_byte_totals_and_retains_expansion(self):
        rows = [dict(raw_bytes=100, igd_bytes=200), dict(raw_bytes=900, igd_bytes=300)]
        result = summarize(rows)
        self.assertEqual(result["total_saving_raw_pct"], 50)
        self.assertEqual(result["expanded_vs_raw_count"], 1)

    def test_statistics_and_portable_identifiers(self):
        self.assertEqual(describe([1, 3])["median"], 2)
        self.assertEqual(describe([1, 3])["std_population"], 1)
        self.assertEqual(image_id(r"C:\old\image_00001.network.npz"), "image_00001")

    def test_png_aggregation_uses_only_matched_files(self):
        rows = [dict(raw_bytes=100, igd_bytes=50, png_bytes=25),
                dict(raw_bytes=100, igd_bytes=10, png_bytes=None)]
        result = summarize(rows)
        self.assertEqual(result["png_comparison_count"], 1)
        self.assertEqual(result["total_saving_png_pct"], -100)
        self.assertEqual(result["expanded_vs_png_count"], 1)
        self.assertNotIn("K", result["metrics"])


if __name__ == "__main__":
    unittest.main()
