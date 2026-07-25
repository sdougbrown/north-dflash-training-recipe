import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from north_dflash_training.feature_stream import (
    BoundedFeatureRing,
    TeacherRuntimeIdentity,
    load_connector_feature_batch,
)
from north_dflash_training.online_step import run_one_bounded_optimizer_step
from north_dflash_training.training import (
    DFlashTrainingStep,
    FrozenSharedWeights,
    SyntheticDraftAdapter,
)


class OnlineStepTests(unittest.TestCase):
    def _ring_and_stream(self):
        identity = TeacherRuntimeIdentity(
            target_name="NorthW4A16Target",
            checkpoint_manifest_sha256="a" * 64,
            runtime_image_id="sha256:" + "b" * 64,
            backend="MARLIN_DETERMINISTIC_PR48032",
            selected_layer_ids=(2, 8),
            hidden_size=3,
            prefix_caching_enabled=False,
        )
        tokens = torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.int64)
        hidden = (
            torch.arange(tokens.numel() * 2 * 3, dtype=torch.float32)
            .add(1)
            .to(torch.bfloat16)
            .reshape(tokens.numel(), 2, 3)
        )
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "trace.safetensors"
        save_file({"hidden_states": hidden, "token_ids": tokens}, path)
        streamed = load_connector_feature_batch(
            path,
            request_id="request-48032",
            runtime_identity=identity,
            expected_token_ids=tokens,
        )
        ring = BoundedFeatureRing(
            runtime_identity=identity,
            max_items=1,
            max_tokens=tokens.numel(),
            max_bytes=streamed.feature_bytes,
        )
        ring.put(streamed)
        return temporary, ring, streamed

    @staticmethod
    def _training_step():
        torch.manual_seed(5)
        embedding = nn.Embedding(32, 4)
        shared = FrozenSharedWeights.handoff_tied_embedding(
            embedding,
            mask_token_id=1,
        )
        adapter = SyntheticDraftAdapter(target_feature_width=6, hidden_size=4)
        step = DFlashTrainingStep(
            adapter=adapter,
            shared_weights=shared,
            selected_layer_ids=(2, 8),
        )
        return embedding, step

    def test_successful_optimizer_step_acknowledges_one_ring_item(self):
        temporary, ring, streamed = self._ring_and_stream()
        self.addCleanup(temporary.cleanup)
        embedding, step = self._training_step()
        frozen_before = embedding.weight.detach().clone()
        versions_before = [parameter._version for parameter in step.parameters()]
        optimizer = torch.optim.AdamW(step.parameters(), lr=0.01)

        result = run_one_bounded_optimizer_step(
            ring=ring,
            training_step=step,
            optimizer=optimizer,
            prompt_length=1,
            block_size=2,
            max_anchors=2,
            mask_token_id=1,
            seed=11,
            loss_gamma=2.0,
        )

        self.assertEqual(result.request_id, streamed.request_id)
        self.assertEqual(result.source_token_count, 6)
        self.assertEqual(result.query_count, 4)
        self.assertEqual(result.active_label_count, 2)
        self.assertGreater(result.gradient_norm, 0)
        self.assertGreater(result.gradients_with_nonzero_values, 0)
        self.assertGreater(result.updated_parameter_tensors, 0)
        self.assertEqual(result.feature_bytes_released, streamed.feature_bytes)
        self.assertEqual((len(ring), ring.token_count, ring.feature_bytes), (0, 0, 0))
        self.assertTrue(torch.equal(embedding.weight, frozen_before))
        self.assertTrue(all(parameter.grad is None for parameter in embedding.parameters()))
        self.assertTrue(
            any(
                parameter._version > version
                for parameter, version in zip(step.parameters(), versions_before, strict=True)
            )
        )

    def test_pre_step_failure_leaves_ring_head_unacknowledged(self):
        temporary, ring, streamed = self._ring_and_stream()
        self.addCleanup(temporary.cleanup)
        _embedding, step = self._training_step()
        optimizer = torch.optim.AdamW(step.parameters(), lr=0.01)
        with self.assertRaisesRegex(ValueError, "mask token"):
            run_one_bounded_optimizer_step(
                ring=ring,
                training_step=step,
                optimizer=optimizer,
                prompt_length=1,
                block_size=2,
                max_anchors=2,
                mask_token_id=9,
                seed=11,
                loss_gamma=2.0,
            )
        self.assertIs(ring.peekleft(), streamed)
        self.assertEqual(len(ring), 1)
        with self.assertRaisesRegex(ValueError, "acknowledgement"):
            ring.ackleft("wrong-request")
        self.assertIs(ring.peekleft(), streamed)


if __name__ == "__main__":
    unittest.main()
