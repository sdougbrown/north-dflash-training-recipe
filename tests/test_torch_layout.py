"""CPU coverage for the optional PyTorch layout adapter."""

import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from north_dflash_training import ResponseExample, build_training_batch_layout, sample_anchor_blocks
    from north_dflash_training.torch_layout import (
        build_flex_attention_block_mask,
        build_torch_training_batch,
        flex_attention_block_mask_supported,
        make_flex_attention_visibility_predicate,
    )


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class TorchSparseLayoutTests(unittest.TestCase):
    @staticmethod
    def _layout():
        sampled = sample_anchor_blocks(
            ResponseExample((90, 91, 92), tuple(range(100, 130))),
            block_size=4,
            max_anchors=2,
            mask_token_id=1,
            seed=4,
        )
        return build_training_batch_layout(sampled, gamma=2.0)

    def test_tensor_inputs_preserve_values_dtypes_and_shapes(self):
        layout = self._layout()
        batch = build_torch_training_batch(layout)
        context_length = max(layout.block_anchor_positions) + 1

        self.assertEqual(batch.input_ids.shape, (1, layout.num_queries))
        self.assertEqual(batch.labels.shape, (1, layout.num_queries))
        self.assertEqual(batch.loss_mask.shape, (1, layout.num_queries))
        self.assertEqual(batch.loss_weights.shape, (1, layout.num_queries))
        self.assertEqual(batch.context_positions.shape, (context_length,))
        self.assertEqual(batch.dense_visibility.shape, (layout.num_queries, context_length + layout.num_queries))
        self.assertEqual(batch.input_ids.dtype, torch.int64)
        self.assertEqual(batch.labels.dtype, torch.int64)
        self.assertEqual(batch.loss_mask.dtype, torch.bool)
        self.assertEqual(batch.loss_weights.dtype, torch.float32)
        self.assertEqual(batch.dense_visibility.dtype, torch.bool)
        self.assertEqual(batch.input_ids.tolist()[0], list(layout.query_tokens))
        self.assertEqual(batch.labels.tolist()[0], list(layout.labels))
        self.assertEqual(batch.loss_mask.tolist()[0], list(layout.loss_mask))
        self.assertEqual(batch.labels[0, 0].item(), -100)
        self.assertEqual(batch.loss_weights[0, 1].item(), 1.0)
        self.assertGreater(batch.loss_weights[0, 1].item(), batch.loss_weights[0, 2].item())
        batch.validate()

    def test_dense_oracle_has_block_isolation_and_context_boundary(self):
        layout = self._layout()
        batch = build_torch_training_batch(layout)
        context_length = batch.context_length
        dense = batch.dense_visibility

        self.assertEqual(dense[:, context_length:].tolist(), [list(row) for row in layout.query_visibility])
        self.assertTrue(dense[0, context_length + 3].item())
        self.assertFalse(dense[0, context_length + 4].item())
        self.assertFalse(dense[4, context_length].item())
        for query_index, anchor in enumerate(layout.anchor_positions):
            self.assertEqual(
                dense[query_index, :context_length].tolist(),
                [position <= anchor for position in range(context_length)],
            )
            self.assertTrue(dense[query_index, anchor].item())
            if anchor + 1 < context_length:
                self.assertFalse(dense[query_index, anchor + 1].item())

    def test_flex_predicate_matches_dense_oracle_on_cpu(self):
        batch = build_torch_training_batch(self._layout())
        predicate = make_flex_attention_visibility_predicate(batch)
        query_indices = torch.arange(batch.num_queries).unsqueeze(1)
        key_indices = torch.arange(batch.key_length).unsqueeze(0)
        actual = predicate(torch.zeros_like(query_indices), torch.zeros_like(query_indices), query_indices, key_indices)
        self.assertTrue(torch.equal(actual, batch.dense_visibility))

    def test_flex_block_mask_matches_dense_oracle_when_exposed(self):
        if not flex_attention_block_mask_supported():
            self.skipTest("this PyTorch build does not expose FlexAttention create_block_mask")
        batch = build_torch_training_batch(self._layout())
        block_mask = build_flex_attention_block_mask(batch, block_size=4)
        query_indices = torch.arange(batch.num_queries).unsqueeze(1)
        key_indices = torch.arange(batch.key_length).unsqueeze(0)
        actual_rule = block_mask.mask_mod(
            torch.zeros_like(query_indices),
            torch.zeros_like(query_indices),
            query_indices,
            key_indices,
        )
        self.assertTrue(torch.equal(actual_rule, batch.dense_visibility))

        # BlockMask.to_dense() reports block occupancy, not token occupancy.
        expected_blocks = torch.zeros(
            ((batch.num_queries + 3) // 4, (batch.key_length + 3) // 4), dtype=torch.bool
        )
        for query_block in range(expected_blocks.shape[0]):
            for key_block in range(expected_blocks.shape[1]):
                expected_blocks[query_block, key_block] = batch.dense_visibility[
                    query_block * 4 : (query_block + 1) * 4,
                    key_block * 4 : (key_block + 1) * 4,
                ].any()
        actual_blocks = block_mask.to_dense()[0, 0].to(dtype=torch.bool)
        self.assertTrue(torch.equal(actual_blocks, expected_blocks))


if __name__ == "__main__":
    unittest.main()
