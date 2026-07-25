"""CPU evidence for the local eager ``DFlashDraftModel`` adapter."""

import importlib.util
import math
import unittest

try:
    import torch
    from torch import nn
except ImportError:
    torch = None

HAS_REFERENCE = torch is not None and importlib.util.find_spec("transformers") is not None and importlib.util.find_spec("dflash") is not None

if HAS_REFERENCE:
    from dflash.model import DFlashDraftModel
    from transformers import Qwen3Config

    from north_dflash_training.training import DFlashTrainingStep, FrozenSharedWeights, TeacherFeatureBundle
    from north_dflash_training.transformers_draft_adapter import (
        TransformersDFlashDraftAdapter,
        dense_visibility_to_eager_attention_mask,
        full_dflash_position_ids,
    )


@unittest.skipUnless(HAS_REFERENCE, "requires optional torch, transformers, and local dflash reference")
class TransformersDFlashDraftAdapterTests(unittest.TestCase):
    hidden_size = 16
    vocab_size = 32

    @classmethod
    def _model(cls):
        config = Qwen3Config(
            vocab_size=cls.vocab_size,
            hidden_size=cls.hidden_size,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            attention_dropout=0.0,
            num_target_layers=4,
            block_size=2,
            dflash_config={"target_layer_ids": [0, 2]},
            attn_implementation="eager",
        )
        model = DFlashDraftModel(config)
        model.eval()
        return model

    @staticmethod
    def _visibility():
        # Rows 0/1 belong to the first block anchored at 1; rows 2/3 are a
        # separate block anchored at 3. Target context is strictly before the
        # anchor, and the other query block is always forbidden.
        return torch.tensor(
            [
                [True, False, False, True, True, False, False],
                [True, False, False, True, True, False, False],
                [True, True, True, False, False, True, True],
                [True, True, True, False, False, True, True],
            ],
            dtype=torch.bool,
        )

    @classmethod
    def _inputs(cls, batch_size=1):
        torch.manual_seed(13)
        context_length, query_length = 3, 4
        target_features = torch.randn(batch_size, context_length, 2 * cls.hidden_size)
        noise_embeddings = torch.randn(batch_size, query_length, cls.hidden_size)
        query_positions = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64).expand(batch_size, -1).clone()
        return target_features, noise_embeddings, query_positions, cls._visibility()

    def test_reference_forward_shape_and_full_rope_positions(self):
        adapter = TransformersDFlashDraftAdapter.from_reference_model(self._model())
        target, noise, query_positions, visibility = self._inputs(batch_size=2)
        full_positions = full_dflash_position_ids(query_positions, context_length=target.shape[1])
        self.assertEqual(full_positions.tolist(), [[0, 1, 2, 1, 2, 3, 4]] * 2)
        output = adapter(
            target_features=target,
            noise_embeddings=noise,
            position_ids=query_positions,
            dense_visibility=visibility,
        )
        self.assertEqual(output.shape, noise.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_additive_mask_exactly_preserves_visibility_and_blocks_leaks(self):
        adapter = TransformersDFlashDraftAdapter.from_reference_model(self._model())
        target, noise, query_positions, visibility = self._inputs()
        mask = dense_visibility_to_eager_attention_mask(
            visibility, batch_size=1, dtype=noise.dtype, device=noise.device
        )
        self.assertEqual(mask.shape, (1, 1, 4, 7))
        self.assertTrue(torch.equal(mask[0, 0].eq(0), visibility))
        self.assertTrue(torch.equal(mask[0, 0].eq(torch.finfo(noise.dtype).min), ~visibility))

        baseline = adapter(
            target_features=target,
            noise_embeddings=noise,
            position_ids=query_positions,
            dense_visibility=visibility,
        )
        altered_target = target.clone()
        altered_target[:, 2, :] += 1000.0  # Context 2 is invisible to rows 0/1.
        altered_noise = noise.clone()
        altered_noise[:, 2:, :] -= 1000.0  # The second query block is invisible to rows 0/1.
        altered = adapter(
            target_features=altered_target,
            noise_embeddings=altered_noise,
            position_ids=query_positions,
            dense_visibility=visibility,
        )
        self.assertTrue(torch.allclose(baseline[:, :2], altered[:, :2], atol=1e-6, rtol=1e-6))

    def test_real_model_gets_all_gradients_while_shared_embedding_and_head_stay_frozen(self):
        torch.manual_seed(7)
        adapter = TransformersDFlashDraftAdapter.from_reference_model(self._model())
        embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        lm_head.weight = embedding.weight
        shared = FrozenSharedWeights.handoff(embedding, lm_head)
        step = DFlashTrainingStep(adapter=adapter, shared_weights=shared, selected_layer_ids=(0, 2))
        target, _noise, positions, visibility = self._inputs()
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        labels = torch.tensor([[-100, 2, 3, 4]], dtype=torch.int64)
        from north_dflash_training.torch_layout import TorchSparseTrainingBatch

        batch = TorchSparseTrainingBatch(
            input_ids=input_ids,
            labels=labels,
            loss_mask=labels.ne(-100),
            loss_weights=torch.ones_like(input_ids, dtype=torch.float32),
            block_ids=torch.tensor([[0, 0, 1, 1]], dtype=torch.int64),
            anchor_positions=torch.tensor([[1, 1, 3, 3]], dtype=torch.int64),
            absolute_query_positions=positions,
            context_positions=torch.arange(3, dtype=torch.int64),
            dense_visibility=visibility,
            mask_token_id=0,
        )
        batch.validate()
        bundle = TeacherFeatureBundle(
            selected_layer_ids=(0, 2),
            hidden_states=(target[..., : self.hidden_size], target[..., self.hidden_size :]),
            clean_positions=torch.arange(3, dtype=torch.int64),
        )
        output = step(batch, bundle)
        output.loss.backward()
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(all(parameter.grad is not None for parameter in adapter.draft_model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in embedding.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in lm_head.parameters()))
        self.assertEqual(list(step.parameters()), list(adapter.draft_model.parameters()))

    def test_bounded_real_model_optimization_smoke(self):
        torch.manual_seed(11)
        adapter = TransformersDFlashDraftAdapter.from_reference_model(self._model())
        embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        shared = FrozenSharedWeights.handoff(embedding, lm_head)
        step = DFlashTrainingStep(adapter=adapter, shared_weights=shared, selected_layer_ids=(0, 2))
        target, _noise, positions, visibility = self._inputs()
        from north_dflash_training.torch_layout import TorchSparseTrainingBatch

        labels = torch.tensor([[-100, 2, 3, 4]], dtype=torch.int64)
        batch = TorchSparseTrainingBatch(
            input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
            labels=labels,
            loss_mask=labels.ne(-100),
            loss_weights=torch.ones((1, 4)),
            block_ids=torch.tensor([[0, 0, 1, 1]], dtype=torch.int64),
            anchor_positions=torch.tensor([[1, 1, 3, 3]], dtype=torch.int64),
            absolute_query_positions=positions,
            context_positions=torch.arange(3, dtype=torch.int64),
            dense_visibility=visibility,
            mask_token_id=0,
        )
        bundle = TeacherFeatureBundle(
            selected_layer_ids=(0, 2),
            hidden_states=(target[..., : self.hidden_size], target[..., self.hidden_size :]),
            clean_positions=torch.arange(3, dtype=torch.int64),
        )
        optimizer = torch.optim.AdamW(step.parameters(), lr=0.01)
        initial = step(batch, bundle).loss.detach().item()
        losses = []
        for _ in range(12):
            optimizer.zero_grad()
            loss = step(batch, bundle).loss
            self.assertTrue(math.isfinite(loss.item()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(step.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach().item())
        self.assertTrue(all(math.isfinite(loss) for loss in losses))
        self.assertLess(losses[-1], initial)
        self.assertTrue(all(parameter.grad is None for parameter in embedding.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in lm_head.parameters()))


if __name__ == "__main__":
    unittest.main()
