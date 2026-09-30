"""Run: python -m unittest discover -s src -p test_delta_baseline.py"""
import unittest
import numpy as np
from tests.try_delta_baseline import deltas_from_document, summarize_deltas, forward


class DeltaBaselineTests(unittest.TestCase):
    def test_exact_gaps_not_spatial_or_sorted_again(self):
        doc = dict(image=dict(pixel_group_width=1, pixel_group_height=1, channels=3),
                   base_bits=6, bases=[dict(bits="101000"), dict(bits="001010"), dict(bits="001100")])
        gaps, width = deltas_from_document(doc)
        np.testing.assert_array_equal(gaps, [2, 28])
        self.assertEqual(width, 6)
        self.assertEqual(gaps.dtype, np.uint32)

    def test_extreme_gap_and_single_base(self):
        doc = dict(image=dict(pixel_group_width=1, pixel_group_height=1, channels=3),
                   base_bits=24, bases=[dict(bits="0" * 24), dict(bits="1" * 24)])
        gaps, _ = deltas_from_document(doc)
        self.assertEqual(int(gaps[0]), 2**24 - 1)
        np.testing.assert_array_equal(summarize_deltas(gaps), np.ones(256))
        doc["bases"] = [dict(bits="0" * 24)]
        gaps, _ = deltas_from_document(doc)
        self.assertEqual(len(gaps), 0)
        np.testing.assert_array_equal(summarize_deltas(gaps), np.zeros(256))

    def test_pooling_preserves_mean_and_does_not_zero_pad(self):
        gaps = np.array([1, 7, 3], dtype=np.uint32)
        for length in [1, 2, 256]:
            summary = summarize_deltas(gaps, length)
            self.assertEqual(summary.shape, (length,))
            self.assertAlmostEqual(float(summary.mean()), float((np.log2(1 + gaps) / 24).mean()), places=6)
            self.assertTrue((summary > 0).all())

    def test_rejects_grouped_data(self):
        doc = dict(image=dict(pixel_group_width=3, pixel_group_height=3, channels=3))
        with self.assertRaises(ValueError):
            deltas_from_document(doc)

    def test_softmax_and_output_gradient(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(4, 3))
        w1, b1 = rng.normal(size=(3, 5)), np.ones(5)
        w2, b2 = rng.normal(size=(5, 10)), np.zeros(10)
        labels = np.array([0, 2, 3, 4])
        hidden, probabilities = forward(x, w1, b1, w2, b2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        grad = probabilities.copy()
        grad[np.arange(4), labels] -= 1
        analytic = hidden.T @ (grad / 4)
        eps = 1e-6
        def loss(weight):
            _, p = forward(x, w1, b1, weight, b2)
            return -np.log(p[np.arange(4), labels]).mean()
        plus, minus = w2.copy(), w2.copy()
        plus[1, 2] += eps
        minus[1, 2] -= eps
        self.assertAlmostEqual(float(analytic[1, 2]), float((loss(plus) - loss(minus)) / (2 * eps)), places=6)


if __name__ == "__main__":
    unittest.main()
