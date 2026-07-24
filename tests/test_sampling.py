import unittest

from north_dflash_training import ResponseExample, sample_anchor_blocks
from north_dflash_training.schema import IGNORE_INDEX


class AnchorSamplingTests(unittest.TestCase):
    def setUp(self):
        self.example = ResponseExample((9, 8), tuple(range(20)))

    def test_seed_is_deterministic_and_bounded(self):
        first = sample_anchor_blocks(self.example, block_size=5, max_anchors=3, mask_token_id=99, seed=11)
        second = sample_anchor_blocks(self.example, block_size=5, max_anchors=3, mask_token_id=99, seed=11)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first.blocks), 3)
        self.assertEqual(len(first.blocks), len(set(first.anchor_positions)))
        self.assertTrue(all(0 <= p <= 15 for p in first.anchor_positions))

    def test_anchor_is_clean_and_future_is_masked(self):
        result = sample_anchor_blocks(self.example, block_size=4, max_anchors=100, mask_token_id=77, seed=0)
        self.assertEqual(len(result.blocks), 17)  # 20 - 4 + 1 eligible anchors
        block = result.blocks[0]
        self.assertEqual(block.input_tokens[0], self.example.response_tokens[block.anchor_position])
        self.assertEqual(block.input_tokens[1:], (77, 77, 77))
        self.assertEqual(block.labels[0], IGNORE_INDEX)
        self.assertEqual(block.labels[1:], self.example.response_tokens[block.anchor_position + 1 : block.anchor_position + 4])
        self.assertEqual(block.loss_mask, (False, True, True, True))
        self.assertEqual(block.absolute_anchor_position, 2 + block.anchor_position)

    def test_sampling_includes_the_last_full_block_and_excludes_the_partial_tail(self):
        result = sample_anchor_blocks(self.example, block_size=4, max_anchors=100, mask_token_id=9)
        self.assertEqual(result.eligible_anchor_positions, tuple(range(17)))
        self.assertEqual(result.anchor_positions[-1], 16)
        self.assertEqual(result.blocks[-1].labels[1:], (17, 18, 19))

        exact = sample_anchor_blocks(ResponseExample((), (1, 2, 3, 4)), block_size=4, max_anchors=5, mask_token_id=9)
        self.assertEqual(exact.anchor_positions, (0,))

    def test_short_response_has_no_partial_tail_block(self):
        example = ResponseExample((), (1, 2, 3))
        result = sample_anchor_blocks(example, block_size=4, max_anchors=5, mask_token_id=9)
        self.assertEqual(result.blocks, ())


if __name__ == "__main__":
    unittest.main()
