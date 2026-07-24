import math
import unittest

from north_dflash_training.cache import estimate_feature_cache
from north_dflash_training.weights import exponential_loss_weights


class WeightAndCacheTests(unittest.TestCase):
    def test_paper_gamma_defaults(self):
        self.assertAlmostEqual(exponential_loss_weights(16)[1], math.exp(-1 / 7))
        self.assertAlmostEqual(exponential_loss_weights(10)[-1], math.exp(-8 / 5))
        self.assertEqual(len(exponential_loss_weights(8)), 7)

    def test_unknown_block_size_requires_explicit_gamma(self):
        with self.assertRaises(ValueError):
            exponential_loss_weights(12)
        self.assertEqual(len(exponential_loss_weights(12, gamma=6)), 11)
        self.assertEqual(len(exponential_loss_weights(12, gamma=6, include_anchor=True)), 12)

    def test_cache_estimate_exposes_disk_and_ring_growth(self):
        estimate = estimate_feature_cache(
            num_sequences=100,
            sequence_length=20,
            selected_layers=2,
            hidden_size=4,
            dtype_bytes=2,
            batch_size=1,
            ring_buffer_tokens=5,
        )
        self.assertEqual(estimate.feature_bytes_per_token, 16)
        self.assertEqual(estimate.disk_cache_bytes, 32_000)
        self.assertEqual(estimate.online_peak_bytes, 320)
        self.assertEqual(estimate.ring_buffer_bytes, 80)
        self.assertEqual(estimate.disk_to_ring_ratio, 400)


if __name__ == "__main__":
    unittest.main()
