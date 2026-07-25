import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from north_dflash_training.connector_lifecycle import (
    ingest_connector_handoff,
    release_connector_after_optimizer_step,
)
from north_dflash_training.feature_stream import (
    BoundedFeatureRing,
    TeacherRuntimeIdentity,
)
from north_dflash_training.online_step import (
    BoundedOptimizerStepResult,
    run_one_bounded_optimizer_step,
)
from north_dflash_training.training import (
    DFlashTrainingStep,
    FrozenSharedWeights,
    SyntheticDraftAdapter,
)


class ConnectorLifecycleTests(unittest.TestCase):
    def _fixture(self, *, max_bytes=10_000):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "trace.safetensors"
        lock_path = Path(f"{path}.lock")
        tokens = torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.int64)
        hidden = (
            torch.arange(tokens.numel() * 2 * 3, dtype=torch.float32)
            .add(1)
            .to(torch.bfloat16)
            .reshape(tokens.numel(), 2, 3)
        )
        save_file({"hidden_states": hidden, "token_ids": tokens}, path)
        lock_path.write_bytes(b"")
        identity = TeacherRuntimeIdentity(
            target_name="NorthFP8Target",
            checkpoint_manifest_sha256="a" * 64,
            runtime_image_id="sha256:" + "b" * 64,
            backend="TRITON_FP8_MOE",
            selected_layer_ids=(2, 8),
            hidden_size=3,
            prefix_caching_enabled=False,
        )
        ring = BoundedFeatureRing(
            runtime_identity=identity,
            max_items=1,
            max_tokens=tokens.numel(),
            max_bytes=max_bytes,
        )
        return path, lock_path, tokens, identity, ring

    @staticmethod
    def _training_step():
        torch.manual_seed(5)
        embedding = nn.Embedding(32, 4)
        shared = FrozenSharedWeights.handoff_tied_embedding(embedding, mask_token_id=1)
        step = DFlashTrainingStep(
            adapter=SyntheticDraftAdapter(target_feature_width=6, hidden_size=4),
            shared_weights=shared,
            selected_layer_ids=(2, 8),
        )
        return step

    @staticmethod
    def _result(request_id="request-fp8", feature_bytes=72):
        return BoundedOptimizerStepResult(
            request_id=request_id,
            source_token_count=6,
            context_length=4,
            query_count=2,
            active_label_count=1,
            loss=1.0,
            gradient_norm=1.0,
            gradients_with_nonzero_values=1,
            updated_parameter_tensors=1,
            feature_bytes_released=feature_bytes,
        )

    def test_successful_optimizer_commit_releases_exact_feature_and_lock(self):
        path, lock_path, tokens, identity, ring = self._fixture()
        handoff = ingest_connector_handoff(
            path,
            request_id="request-fp8",
            runtime_identity=identity,
            expected_token_ids=tokens,
            ring=ring,
        )
        step = self._training_step()
        result = run_one_bounded_optimizer_step(
            ring=ring,
            training_step=step,
            optimizer=torch.optim.AdamW(step.parameters(), lr=0.01),
            prompt_length=1,
            block_size=2,
            max_anchors=2,
            mask_token_id=1,
            seed=11,
            loss_gamma=2.0,
        )

        released = release_connector_after_optimizer_step(
            handoff,
            optimizer_result=result,
            ring=ring,
        )

        self.assertFalse(path.exists())
        self.assertFalse(lock_path.exists())
        self.assertEqual(released.request_id, "request-fp8")
        self.assertEqual(released.feature_tensor_bytes_consumed, 72)
        self.assertGreater(released.feature_file_bytes_released, 72)
        self.assertEqual(len(released.feature_sha256), 64)
        self.assertTrue(released.lock_file_released)

    def test_release_before_ring_acknowledgement_fails_closed(self):
        path, lock_path, tokens, identity, ring = self._fixture()
        handoff = ingest_connector_handoff(
            path,
            request_id="request-fp8",
            runtime_identity=identity,
            expected_token_ids=tokens,
            ring=ring,
        )

        with self.assertRaisesRegex(RuntimeError, "still active"):
            release_connector_after_optimizer_step(
                handoff,
                optimizer_result=self._result(),
                ring=ring,
            )

        self.assertTrue(path.is_file())
        self.assertTrue(lock_path.is_file())
        self.assertTrue(ring.contains_request("request-fp8"))

    def test_changed_artifact_is_not_deleted_after_acknowledgement(self):
        path, lock_path, tokens, identity, ring = self._fixture()
        handoff = ingest_connector_handoff(
            path,
            request_id="request-fp8",
            runtime_identity=identity,
            expected_token_ids=tokens,
            ring=ring,
        )
        ring.ackleft("request-fp8")
        with path.open("ab") as handle:
            handle.write(b"changed")

        with self.assertRaisesRegex(RuntimeError, "changed"):
            release_connector_after_optimizer_step(
                handoff,
                optimizer_result=self._result(),
                ring=ring,
            )

        self.assertTrue(path.is_file())
        self.assertTrue(lock_path.is_file())

    def test_ring_overflow_leaves_connector_artifacts_untouched(self):
        path, lock_path, tokens, identity, ring = self._fixture(max_bytes=1)

        with self.assertRaisesRegex(BufferError, "ring is full"):
            ingest_connector_handoff(
                path,
                request_id="request-fp8",
                runtime_identity=identity,
                expected_token_ids=tokens,
                ring=ring,
            )

        self.assertTrue(path.is_file())
        self.assertTrue(lock_path.is_file())
        self.assertEqual(len(ring), 0)

    def test_symbolic_lock_file_is_rejected_without_deleting_feature(self):
        path, lock_path, tokens, identity, ring = self._fixture()
        lock_path.unlink()
        lock_path.symlink_to(Path(path.parent) / "missing-lock-target")

        with self.assertRaisesRegex(ValueError, "symbolic links"):
            ingest_connector_handoff(
                path,
                request_id="request-fp8",
                runtime_identity=identity,
                expected_token_ids=tokens,
                ring=ring,
            )

        self.assertTrue(path.is_file())
        self.assertTrue(lock_path.is_symlink())
        self.assertEqual(len(ring), 0)

    def test_request_drift_never_releases_artifact(self):
        path, lock_path, tokens, identity, ring = self._fixture()
        handoff = ingest_connector_handoff(
            path,
            request_id="request-fp8",
            runtime_identity=identity,
            expected_token_ids=tokens,
            ring=ring,
        )
        ring.ackleft("request-fp8")

        with self.assertRaisesRegex(ValueError, "request_id"):
            release_connector_after_optimizer_step(
                handoff,
                optimizer_result=self._result(request_id="wrong-request"),
                ring=ring,
            )

        self.assertTrue(path.is_file())
        self.assertTrue(lock_path.is_file())


if __name__ == "__main__":
    unittest.main()
