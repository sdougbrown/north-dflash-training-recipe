"""CPU tests for the optional-PyTorch training-step contract."""

import math
import unittest

try:
    import torch
    from torch import nn
except ImportError:
    torch = None

if torch is not None:
    from north_dflash_training import ResponseExample, build_training_batch_layout, sample_anchor_blocks
    from north_dflash_training.torch_layout import build_torch_training_batch
    from north_dflash_training.training import (
        DFlashTrainingStep,
        FrozenSharedWeights,
        SyntheticDraftAdapter,
        TeacherFeatureBundle,
        concatenate_target_features,
        weighted_masked_cross_entropy,
    )


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class TrainingContractTests(unittest.TestCase):
    @staticmethod
    def _batch():
        sampled = sample_anchor_blocks(
            ResponseExample((), tuple(range(1, 9))),
            block_size=4,
            max_anchors=1,
            mask_token_id=0,
            seed=3,
        )
        return build_torch_training_batch(build_training_batch_layout(sampled, gamma=2.0))

    def _bundle(self, batch, *, layer_ids=(7, 2), hidden_size=3):
        context = batch.context_length
        first = torch.full((1, context, hidden_size), 1.0)
        second = torch.full((1, context, hidden_size), 2.0)
        return TeacherFeatureBundle(
            selected_layer_ids=layer_ids,
            hidden_states=(first, second),
            clean_positions=torch.arange(context, dtype=torch.int64),
        )

    def test_selected_feature_order_and_context_shape_are_preserved(self):
        batch = self._batch()
        bundle = self._bundle(batch)
        features = concatenate_target_features(bundle, expected_layer_ids=(7, 2))
        self.assertEqual(features.shape, (1, batch.context_length, 6))
        self.assertTrue(torch.equal(features[..., :3], torch.ones_like(features[..., :3])))
        self.assertTrue(torch.equal(features[..., 3:], torch.full_like(features[..., 3:], 2.0)))
        with self.assertRaisesRegex(ValueError, "layer order"):
            concatenate_target_features(bundle, expected_layer_ids=(2, 7))
        bad_positions = TeacherFeatureBundle(
            selected_layer_ids=(7, 2),
            hidden_states=bundle.hidden_states,
            clean_positions=torch.arange(1, batch.context_length + 1, dtype=torch.int64),
        )
        with self.assertRaisesRegex(ValueError, "clean prefix"):
            concatenate_target_features(bad_positions)
        mismatched_shape = TeacherFeatureBundle(
            selected_layer_ids=(7, 2),
            hidden_states=(bundle.hidden_states[0], bundle.hidden_states[1][..., :2]),
            clean_positions=bundle.clean_positions,
        )
        with self.assertRaisesRegex(ValueError, "share shape"):
            concatenate_target_features(mismatched_shape)

    def test_shared_weights_are_direct_references_frozen_and_draft_only_gets_gradients(self):
        torch.manual_seed(0)
        batch = self._batch()
        embedding = nn.Embedding(16, 6)
        lm_head = nn.Linear(6, 16, bias=False)
        lm_head.weight = embedding.weight  # Simulate a teacher with tied input/output weights.
        shared = FrozenSharedWeights.handoff(embedding, lm_head)
        self.assertIs(shared.embedding, embedding)
        self.assertIs(shared.lm_head, lm_head)
        self.assertIs(shared.embedding.weight, shared.lm_head.weight)
        self.assertTrue(all(not parameter.requires_grad for parameter in embedding.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in lm_head.parameters()))
        adapter = SyntheticDraftAdapter(target_feature_width=6, hidden_size=6)
        step = DFlashTrainingStep(adapter=adapter, shared_weights=shared, selected_layer_ids=(7, 2))
        output = step(batch, self._bundle(batch))
        output.loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in adapter.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in embedding.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in lm_head.parameters()))
        self.assertEqual(list(step.parameters()), list(adapter.parameters()))

    def test_north_tied_embedding_handoff_uses_mask_row_and_output_weight(self):
        embedding = nn.Embedding(12, 4)
        with torch.no_grad():
            embedding.weight.copy_(torch.arange(48).reshape(12, 4))
        shared = FrozenSharedWeights.handoff_tied_embedding(
            embedding,
            mask_token_id=1,
        )
        self.assertIs(shared.embedding, embedding)
        self.assertIsNone(shared.lm_head)
        self.assertTrue(shared.tied_output_embedding)
        self.assertEqual(shared.mask_token_id, 1)
        mask_ids = torch.tensor([[1, 1]], dtype=torch.int64)
        self.assertTrue(
            torch.equal(shared.embed(mask_ids), embedding.weight[1].reshape(1, 1, 4).expand(1, 2, 4))
        )
        hidden = torch.randn(1, 2, 4)
        self.assertTrue(torch.equal(shared.logits(hidden), torch.nn.functional.linear(hidden, embedding.weight)))
        self.assertTrue(all(not parameter.requires_grad for parameter in embedding.parameters()))
        with self.assertRaisesRegex(ValueError, "mask_token_id"):
            FrozenSharedWeights.handoff_tied_embedding(nn.Embedding(3, 4), mask_token_id=3)

    def test_weighted_ce_ignores_anchor_and_uses_exponential_weights(self):
        logits = torch.tensor([[[100.0, -100.0], [0.0, 2.0], [1.0, 0.0]]])
        labels = torch.tensor([[-100, 1, 0]], dtype=torch.int64)
        mask = torch.tensor([[False, True, True]])
        weights = torch.tensor([[123.0, 1.0, math.exp(-0.5)]])
        actual = weighted_masked_cross_entropy(logits, labels, mask, weights)
        nll_one = -torch.log_softmax(logits[0, 1], dim=-1)[1]
        nll_two = -torch.log_softmax(logits[0, 2], dim=-1)[0]
        expected = (nll_one + math.exp(-0.5) * nll_two) / (1 + math.exp(-0.5))
        self.assertTrue(torch.allclose(actual, expected))
        with self.assertRaisesRegex(ValueError, "exactly"):
            weighted_masked_cross_entropy(logits, labels, ~mask, weights)

    def test_synthetic_adapter_decreases_cpu_loss_without_training_shared_weights(self):
        torch.manual_seed(4)
        batch = self._batch()
        hidden_size = vocab_size = 16
        embedding = nn.Embedding(vocab_size, hidden_size)
        lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        with torch.no_grad():
            embedding.weight.zero_()
            lm_head.weight.copy_(torch.eye(vocab_size))
        shared = FrozenSharedWeights.handoff(embedding, lm_head)
        bundle = TeacherFeatureBundle(
            selected_layer_ids=(0,),
            hidden_states=(torch.zeros(1, batch.context_length, 2),),
            clean_positions=torch.arange(batch.context_length, dtype=torch.int64),
        )
        adapter = SyntheticDraftAdapter(target_feature_width=2, hidden_size=hidden_size)
        step = DFlashTrainingStep(adapter=adapter, shared_weights=shared, selected_layer_ids=(0,))
        optimizer = torch.optim.Adam(step.parameters(), lr=0.08)
        initial = step(batch, bundle).loss.item()
        for _ in range(80):
            optimizer.zero_grad()
            loss = step(batch, bundle).loss
            loss.backward()
            optimizer.step()
        final = step(batch, bundle).loss.item()
        self.assertLess(final, initial)
        self.assertTrue(all(parameter.grad is None for parameter in embedding.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in lm_head.parameters()))


if __name__ == "__main__":
    unittest.main()
