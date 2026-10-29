import struct
import tempfile
import unittest
from pathlib import Path

from audit_compression import describe, image_id, read_header, summarize


class AuditTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
