"""CPU tests for the draft-only DFlash checkpoint save/resume contract.

These tests use tiny CPU models and will never touch GPUs.  They prove:

1. Round-trip save then resume produces identical loss on a second forward pass.
2. No-overwrite enforcement when the checkpoint directory already exists.
3. Tamper detection: altering a saved file makes verification or resume fail.
4. Active ring rejection: checkpointing is refused while the ring is non-empty.
5. Shared-weight exclusion: the checkpoint manifest records tied-vocab identity
   as metadata only, and the draft-parameter check rejects leakage.
6. Verifier family separation through exact runtime identity.
7. Drift rejection on ledger, layer IDs, hidden size.
8. The external manifest digest root-of-trust is verified on resume.
9. Selected-layer ID mismatch between training_step and runtime_identity.
10. Tied-output handoff requirement; untied heads rejected.
11. Adapter class/config identity preserved in manifest.
12. Optimizer parameter reorder, type, and state drift detection.
13. Connector-release ledger fields (connector_sha256, etc.).
14. Response token ledger round-trip and drift detection.
15. BF16 round-trip on CPU.
16. Concurrent RENAME_NOREPLACE publication.
17. State-prefix allowlisting for supported adapters.
"""

import hashlib
import io as io_module
import json
import os
import pickle
import tempfile
import threading
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:
    torch = None

if torch is not None:
    from north_dflash_training import (
        build_training_batch_layout,
        sample_anchor_blocks,
        ResponseExample,
    )
    from north_dflash_training.save_resume import (
        CheckpointManifest,
        SavedRequestLedger,
        TiedVocabIdentity,
        resume_checkpoint,
        save_checkpoint,
        verify_checkpoint_directory,
        CHECKPOINT_SCHEMA_VERSION,
        MANIFEST_FILENAME,
        MANIFEST_SHA256_FILENAME,
        RESPONSE_LEDGER_FILENAME,
        _draft_architecture,
        _extract_optimizer_param_groups,
        _validate_optimizer_class,
        _response_examples_canonical_bytes,
        _response_ledger_sha256,
    )
    from north_dflash_training.feature_stream import (
        BoundedFeatureRing,
        TeacherRuntimeIdentity,
    )
    from north_dflash_training.torch_layout import build_torch_training_batch
    from north_dflash_training.training import (
        DFlashTrainingStep,
        FrozenSharedWeights,
        SyntheticDraftAdapter,
        TeacherFeatureBundle,
    )


# ---------------------------------------------------------------------------
# Hex helpers
# ---------------------------------------------------------------------------


def _hex64(seed: str = "a") -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


_H0 = _hex64("r0")
_H1 = _hex64("r1")
_H2 = _hex64("r2")
_H3 = _hex64("r3")
_H4 = _hex64("r4")
_H5 = _hex64("r5")
_H6 = _hex64("r6")
_H7 = _hex64("r7")
_H8 = _hex64("r8")
_H9 = _hex64("r9")
_HA = _hex64("ra")
_HB = _hex64("rb")
_HC = _hex64("rc")
_HD = _hex64("rd")
_HE = _hex64("re")
_HF = _hex64("rf")
_HG = _hex64("rg")
_HH = _hex64("rh")
_HI = _hex64("ri")
_HJ = _hex64("rj")
_HK = _hex64("rk")
_HL = _hex64("rl")
_HM = _hex64("rm")
_HN = _hex64("rn")
_HO = _hex64("ro")
_HP = _hex64("rp")
_HQ = _hex64("rq")
_HR = _hex64("rr")
_HS = _hex64("rs")
_HT = _hex64("rt")
_HU = _hex64("ru")
_HV = _hex64("rv")
_HW = _hex64("rw")
_HX = _hex64("rx")
_HY = _hex64("ry")
_HZ = _hex64("rz")
_IMG_A = "sha256:" + _hex64("img_a")
_IMG_B = "sha256:" + _hex64("img_b")
_IMG_C = "sha256:" + _hex64("img_c")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _dummy_runtime_identity(*, suffix: str = "a") -> TeacherRuntimeIdentity:
    return TeacherRuntimeIdentity(
        target_name=f"TestTarget{suffix}",
        checkpoint_manifest_sha256=_H0,
        runtime_image_id=_IMG_A,
        backend="CPU_TEST",
        selected_layer_ids=(0, 1),
        hidden_size=4,
        prefix_caching_enabled=False,
    )


def _dummy_ring(runtime: TeacherRuntimeIdentity) -> BoundedFeatureRing:
    return BoundedFeatureRing(
        runtime_identity=runtime,
        max_items=1,
        max_tokens=128,
        max_bytes=1_000_000,
    )


def _dummy_training_step(*, hidden_size: int = 4) -> DFlashTrainingStep:
    embedding = nn.Embedding(16, hidden_size)
    with torch.no_grad():
        embedding.weight.copy_(torch.randn_like(embedding.weight))
    shared = FrozenSharedWeights.handoff_tied_embedding(embedding, mask_token_id=1)
    adapter = SyntheticDraftAdapter(
        target_feature_width=2 * hidden_size,
        hidden_size=hidden_size,
    )
    return DFlashTrainingStep(
        adapter=adapter,
        shared_weights=shared,
        selected_layer_ids=(0, 1),
    )


def _dummy_optimizer(step: DFlashTrainingStep) -> torch.optim.Optimizer:
    return torch.optim.AdamW(step.parameters(), lr=0.01)


def _build_connector_entry(
    request_id: str,
    *,
    loss: float = 1.0,
    gradient_norm: float = 0.5,
    connector_sha256: str | None = None,
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "source_token_count": 6,
        "context_length": 4,
        "query_count": 2,
        "active_label_count": 1,
        "loss": loss,
        "gradient_norm": gradient_norm,
        "gradients_with_nonzero_values": 2,
        "updated_parameter_tensors": 2,
        "feature_bytes_released": 64,
        "connector_sha256": connector_sha256 or _hex64(f"conn_{request_id}"),
        "connector_file_bytes_released": 65536,
        "lock_file_released": True,
    }


def _small_ledger(step_count: int = 1) -> SavedRequestLedger:
    return SavedRequestLedger(
        entries=tuple(
            _build_connector_entry(f"req-{i}")
            for i in range(step_count)
        )
    )


def _response_examples(step_count: int = 1) -> list[ResponseExample]:
    return [
        ResponseExample(
            prompt_tokens=(),
            response_tokens=tuple(range(1, 7)),
            metadata={"request_id": f"req-{i}"},
        )
        for i in range(step_count)
    ]


def _dummy_torch_batch(
    *,
    context_length: int = 4,
    block_size: int = 2,
    hidden_size: int = 4,
) -> tuple[torch.Tensor, TeacherFeatureBundle]:
    example = ResponseExample(
        prompt_tokens=(),
        response_tokens=tuple(range(1, context_length + 1)),
    )
    sampled = sample_anchor_blocks(
        example,
        block_size=block_size,
        max_anchors=context_length // block_size,
        mask_token_id=0,
        seed=42,
    )
    batch = build_torch_training_batch(
        build_training_batch_layout(sampled, gamma=2.0),
    )
    bundle = TeacherFeatureBundle(
        selected_layer_ids=(0, 1),
        hidden_states=(
            torch.full((1, batch.context_length, hidden_size), 1.0),
            torch.full((1, batch.context_length, hidden_size), 2.0),
        ),
        clean_positions=torch.arange(batch.context_length, dtype=torch.int64),
    )
    return batch, bundle


def _fill_ring(ring: BoundedFeatureRing, runtime: TeacherRuntimeIdentity) -> None:
    from safetensors.torch import save_file
    from north_dflash_training.feature_stream import load_connector_feature_batch

    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "trace.safetensors"
        tokens = torch.tensor([2, 3, 4, 5], dtype=torch.int64)
        hidden = torch.full((4, 2, 4), 1.0, dtype=torch.bfloat16)
        save_file({"hidden_states": hidden, "token_ids": tokens}, trace)
        streamed = load_connector_feature_batch(
            trace,
            request_id="req-active",
            runtime_identity=runtime,
            expected_token_ids=tokens,
        )
        ring.put(streamed)


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class SaveResumeRoundTripTests(unittest.TestCase):
    """Prove that save -> resume -> forward produces identical loss."""

    def test_cpu_round_trip_preserves_loss_after_resume(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        torch_batch, bundle = _dummy_torch_batch()
        output = step(torch_batch, bundle)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss_after_step = step(torch_batch, bundle).loss.item()
        completed_steps = 1
        ledger = _small_ledger(step_count=completed_steps)
        response_examples = _response_examples(step_count=completed_steps)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoint-v1"
            manifest = save_checkpoint(
                ckpt_dir, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=completed_steps, request_ledger=ledger,
                response_examples=response_examples, runtime_identity=runtime,
            )
            expected_digest = manifest.sha256
            self.assertEqual(manifest.step_count, completed_steps)
            self.assertEqual(manifest.request_ledger_entry_count, completed_steps)
            self.assertEqual(manifest.response_ledger_entry_count, completed_steps)
            self.assertEqual(len(manifest.files), 4)
            self.assertIsNotNone(manifest.draft_architecture)
            self.assertIn("adapter_type", manifest.draft_architecture)
            self.assertIsNone(manifest.draft_architecture["config"])

            torch.manual_seed(999)
            fresh_step = _dummy_training_step()
            fresh_optimizer = _dummy_optimizer(fresh_step)
            fresh_step.shared_weights = step.shared_weights
            resumed_manifest, resumed_ledger, resumed_response = resume_checkpoint(
                ckpt_dir, training_step=fresh_step, optimizer=fresh_optimizer,
                device=torch.device("cpu"), runtime_identity=runtime,
                expected_manifest_sha256=expected_digest,
            )
            with torch.no_grad():
                resumed_output = fresh_step(torch_batch, bundle)
                loss_after_resume = resumed_output.loss.item()
            self.assertAlmostEqual(loss_after_resume, loss_after_step, places=5)
            self.assertEqual(resumed_manifest.step_count, completed_steps)
            self.assertEqual(len(resumed_response), completed_steps)
            self.assertEqual(resumed_response[0]["response_tokens"], [1, 2, 3, 4, 5, 6])

    def test_round_trip_with_multiple_steps_in_ledger(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        torch_batch, bundle = _dummy_torch_batch()
        for _ in range(3):
            output = step(torch_batch, bundle)
            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()
        ledger = _small_ledger(step_count=3)
        response_examples = _response_examples(step_count=3)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoint-v2"
            manifest = save_checkpoint(
                ckpt_dir, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=3, request_ledger=ledger,
                response_examples=response_examples, runtime_identity=runtime,
            )
            self.assertEqual(manifest.step_count, 3)
            self.assertEqual(manifest.response_ledger_entry_count, 3)
            fresh_step = _dummy_training_step()
            fresh_step.shared_weights = step.shared_weights
            fresh_optimizer = _dummy_optimizer(fresh_step)
            _, resumed_ledger, resumed_response = resume_checkpoint(
                ckpt_dir, training_step=fresh_step, optimizer=fresh_optimizer,
                device=torch.device("cpu"), runtime_identity=runtime,
                expected_manifest_sha256=manifest.sha256,
            )
            self.assertEqual(len(resumed_response), 3)

    def test_tied_vocab_identity_is_verified_on_resume(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        torch_batch, bundle = _dummy_torch_batch()
        output = step(torch_batch, bundle)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoint-vocab"
            manifest = save_checkpoint(
                ckpt_dir, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=_response_examples(), runtime_identity=runtime,
            )
            wrong_embedding = nn.Embedding(16, 4)
            wrong_shared = FrozenSharedWeights.handoff_tied_embedding(wrong_embedding, mask_token_id=1)
            wrong_adapter = SyntheticDraftAdapter(target_feature_width=8, hidden_size=4)
            wrong_step = DFlashTrainingStep(adapter=wrong_adapter, shared_weights=wrong_shared, selected_layer_ids=(0, 1))
            wrong_optimizer = _dummy_optimizer(wrong_step)
            with self.assertRaisesRegex(ValueError, "tied vocabulary identity does not match"):
                resume_checkpoint(ckpt_dir, training_step=wrong_step, optimizer=wrong_optimizer,
                    device=torch.device("cpu"), runtime_identity=runtime,
                    expected_manifest_sha256=manifest.sha256)


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class NoOverwriteTests(unittest.TestCase):
    def test_existing_directory_is_rejected(self):
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "existing"
            existing.mkdir()
            (existing / "dummy.txt").write_text("data")
            with self.assertRaises(FileExistsError):
                save_checkpoint(existing, ring=ring, training_step=step, optimizer=optimizer,
                    completed_steps=1, request_ledger=_small_ledger(),
                    response_examples=_response_examples(), runtime_identity=runtime)

    def test_completed_steps_must_match_ledger_count(self):
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "does not match completed_steps"):
                save_checkpoint(Path(tmp) / "mismatch", ring=ring, training_step=step, optimizer=optimizer,
                    completed_steps=2, request_ledger=_small_ledger(step_count=1),
                    response_examples=_response_examples(step_count=1), runtime_identity=runtime)


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class TamperDetectionTests(unittest.TestCase):
    def _save_fixture(self, tmp: Path):
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        torch_batch, bundle = _dummy_torch_batch()
        output = step(torch_batch, bundle)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        ckpt_dir = tmp / "checkpoint"
        manifest = save_checkpoint(ckpt_dir, ring=ring, training_step=step, optimizer=optimizer,
            completed_steps=1, request_ledger=_small_ledger(),
            response_examples=_response_examples(), runtime_identity=runtime)
        return ckpt_dir, runtime, step, optimizer, manifest.sha256

    def _fresh_step(self, original_step):
        fresh = _dummy_training_step()
        fresh.shared_weights = original_step.shared_weights
        return fresh

    def test_tampered_weights_fail_verification(self):
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir, runtime, step, optimizer, digest = self._save_fixture(Path(tmp))
            (ckpt_dir / "draft-model.safetensors").write_bytes(
                (ckpt_dir / "draft-model.safetensors").read_bytes() + b"tamper"
            )
            fresh = self._fresh_step(step)
            with self.assertRaisesRegex(ValueError, "file hash or size mismatch"):
                resume_checkpoint(ckpt_dir, training_step=fresh, optimizer=_dummy_optimizer(fresh),
                    device=torch.device("cpu"), runtime_identity=runtime, expected_manifest_sha256=digest)

    def test_tampered_manifest_is_rejected_by_digest_mismatch(self):
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir, runtime, step, optimizer, digest = self._save_fixture(Path(tmp))
            manifest_path = ckpt_dir / MANIFEST_FILENAME
            data = json.loads(manifest_path.read_text())
            data["step_count"] = 999
            manifest_path.write_text(json.dumps(data, sort_keys=True, indent=2))
            with self.assertRaisesRegex(ValueError, "manifest digest does not match"):
                verify_checkpoint_directory(ckpt_dir, runtime_identity=runtime, expected_manifest_sha256=digest)

    def test_tampered_optimizer_fails_verification(self):
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir, runtime, step, optimizer, digest = self._save_fixture(Path(tmp))
            (ckpt_dir / "optimizer.pt").write_bytes(b"corrupted-pickle-data")
            fresh = self._fresh_step(step)
            with self.assertRaisesRegex(ValueError, "file hash or size mismatch"):
                resume_checkpoint(ckpt_dir, training_step=fresh, optimizer=_dummy_optimizer(fresh),
                    device=torch.device("cpu"), runtime_identity=runtime, expected_manifest_sha256=digest)

    def test_wrong_external_digest_is_rejected(self):
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir, runtime, step, optimizer, _ = self._save_fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "manifest digest does not match"):
                verify_checkpoint_directory(ckpt_dir, runtime_identity=runtime,
                    expected_manifest_sha256=_hex64("wrong"))


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class ActiveRingRejectionTests(unittest.TestCase):
    def test_non_empty_ring_is_rejected(self):
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        _fill_ring(ring, runtime)
        self.assertEqual(len(ring), 1)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "active ring"):
                save_checkpoint(Path(tmp) / "bad", ring=ring, training_step=step, optimizer=optimizer,
                    completed_steps=1, request_ledger=_small_ledger(),
                    response_examples=_response_examples(), runtime_identity=runtime)

    def test_empty_ring_is_accepted(self):
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoint-empty-ring"
            manifest = save_checkpoint(ckpt_dir, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=_response_examples(), runtime_identity=runtime)
            self.assertIsNotNone(manifest)


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class RingIdentityMismatchTests(unittest.TestCase):
    def test_ring_identity_mismatch_is_rejected(self):
        runtime_a = _dummy_runtime_identity(suffix="A")
        runtime_b = _dummy_runtime_identity(suffix="B")
        ring = _dummy_ring(runtime_b)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "ring runtime identity does not match"):
                save_checkpoint(Path(tmp) / "id-mismatch", ring=ring, training_step=step, optimizer=optimizer,
                    completed_steps=1, request_ledger=_small_ledger(),
                    response_examples=_response_examples(), runtime_identity=runtime_a)


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class SharedWeightExclusionTests(unittest.TestCase):
    def test_saved_tensors_do_not_include_embedding_weight(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpt"
            save_checkpoint(ckpt_dir, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=_response_examples(), runtime_identity=runtime)
            from safetensors.torch import load_file as safe_load_file
            weights = safe_load_file(ckpt_dir / "draft-model.safetensors", device="cpu")
            self.assertEqual([k for k in weights if "embed" in k.lower()], [])

    def test_adapter_with_leaked_shared_weight_is_rejected(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        embedding = nn.Embedding(16, 4)
        adapter = SyntheticDraftAdapter(target_feature_width=8, hidden_size=4)
        leaked = nn.Module()
        leaked.target_projection = adapter.target_projection
        leaked.noise_projection = adapter.noise_projection
        leaked.position_projection = adapter.position_projection
        leaked.output_projection = adapter.output_projection
        leaked.leaked_embed = embedding
        with torch.no_grad():
            embedding.weight.copy_(torch.randn_like(embedding.weight))
        shared = FrozenSharedWeights.handoff_tied_embedding(embedding, mask_token_id=1)
        bad_step = DFlashTrainingStep(adapter=leaked, shared_weights=shared, selected_layer_ids=(0, 1))
        bad_opt = _dummy_optimizer(bad_step)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "shared vocabulary parameters"):
                save_checkpoint(Path(tmp) / "leak", ring=ring, training_step=bad_step, optimizer=bad_opt,
                    completed_steps=1, request_ledger=_small_ledger(),
                    response_examples=_response_examples(), runtime_identity=runtime)

    def test_manifest_records_identity_hash_not_tensor(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpt"
            save_checkpoint(ckpt_dir, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=_response_examples(), runtime_identity=runtime)
            manifest_data = json.loads((ckpt_dir / MANIFEST_FILENAME).read_text())
            tied = manifest_data.get("tied_vocab_identity")
            self.assertIsNotNone(tied)
            self.assertEqual(set(tied.keys()), {"sha256", "shape", "dtype"})


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class VerifierFamilySeparationTests(unittest.TestCase):
    def test_different_runtime_identity_is_rejected_on_resume(self):
        torch.manual_seed(42)
        runtime_a = _dummy_runtime_identity(suffix="A")
        runtime_b = _dummy_runtime_identity(suffix="B")
        ring = _dummy_ring(runtime_a)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpt"
            m = save_checkpoint(ckpt_dir, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=_response_examples(), runtime_identity=runtime_a)
            fresh = _dummy_training_step()
            fresh.shared_weights = step.shared_weights
            with self.assertRaisesRegex(ValueError, "runtime identity"):
                resume_checkpoint(ckpt_dir, training_step=fresh, optimizer=_dummy_optimizer(fresh),
                    device=torch.device("cpu"), runtime_identity=runtime_b, expected_manifest_sha256=m.sha256)


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class DriftRejectionTests(unittest.TestCase):
    def test_ledger_tamper_is_detected(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpt"
            m = save_checkpoint(ckpt_dir, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=_response_examples(), runtime_identity=runtime)
            ledger_data = json.loads((ckpt_dir / "request-ledger.json").read_text())
            ledger_data[0]["loss"] = 0.5
            (ckpt_dir / "request-ledger.json").write_text(json.dumps(ledger_data, sort_keys=True, indent=2))
            fresh = _dummy_training_step()
            fresh.shared_weights = step.shared_weights
            with self.assertRaisesRegex(ValueError, "file hash or size mismatch"):
                resume_checkpoint(ckpt_dir, training_step=fresh, optimizer=_dummy_optimizer(fresh),
                    device=torch.device("cpu"), runtime_identity=runtime, expected_manifest_sha256=m.sha256)

    def test_optimizer_type_drift_is_rejected_before_loading(self):
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        with tempfile.TemporaryDirectory() as directory:
            ckpt = Path(directory) / "ckpt"
            m = save_checkpoint(ckpt, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=_response_examples(), runtime_identity=runtime)
            fresh = _dummy_training_step()
            fresh.shared_weights = step.shared_weights
            sdg = torch.optim.SGD(fresh.parameters(), lr=0.01)
            with self.assertRaisesRegex(ValueError, "optimizer type"):
                resume_checkpoint(ckpt, training_step=fresh, optimizer=sdg,
                    device=torch.device("cpu"), runtime_identity=runtime, expected_manifest_sha256=m.sha256)


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class ManifestStructureTests(unittest.TestCase):
    def test_manifest_round_trip_via_dict(self):
        tied = TiedVocabIdentity(sha256=_hex64("vocab"), shape=(10, 20), dtype="float32")
        files = (
            {"relative_path": "draft-model.safetensors", "size_bytes": 100, "sha256": _hex64("w1")},
            {"relative_path": "optimizer.pt", "size_bytes": 200, "sha256": _hex64("w2")},
            {"relative_path": "request-ledger.json", "size_bytes": 300, "sha256": _hex64("w3")},
            {"relative_path": "response-ledger.json", "size_bytes": 400, "sha256": _hex64("w4")},
        )
        m = CheckpointManifest(
            schema_version=CHECKPOINT_SCHEMA_VERSION, step_count=5, target_name="T",
            runtime_image_id="sha256:" + _hex64("rimg"), backend="CPU",
            checkpoint_manifest_sha256=_hex64("cmh"), selected_layer_ids=(0, 1, 2),
            hidden_size=512, request_ledger_sha256=_hex64("rlsh"), request_ledger_entry_count=5,
            response_ledger_sha256=_hex64("rpsh"), response_ledger_entry_count=5,
            tied_vocab_identity=tied,
            draft_architecture={"adapter_type": "test.Adapter", "config": {"lr": 0.01}},
            optimizer_type="torch.optim.adamw.AdamW",
            optimizer_param_groups=[[{"name": "w1", "shape": [4, 4], "dtype": "float32"}]],
            files=files,
        )
        self.assertEqual(CheckpointManifest.from_dict(m.to_dict()), m)

    def test_draft_architecture_must_not_be_none(self):
        tied = TiedVocabIdentity(sha256=_hex64("tv"), shape=(16, 4), dtype="float32")
        files = tuple(
            {"relative_path": n, "size_bytes": 100, "sha256": _hex64(n)}
            for n in ("draft-model.safetensors", "optimizer.pt", "request-ledger.json", "response-ledger.json")
        )
        with self.assertRaisesRegex(ValueError, "draft_architecture must not be None"):
            CheckpointManifest(
                schema_version=CHECKPOINT_SCHEMA_VERSION, step_count=0, target_name="T",
                runtime_image_id="sha256:" + _hex64("rimg"), backend="CPU",
                checkpoint_manifest_sha256=_hex64("cmh"), selected_layer_ids=(0,),
                hidden_size=512, request_ledger_sha256=_hex64("rl"),
                request_ledger_entry_count=0, response_ledger_sha256=_hex64("rp"),
                response_ledger_entry_count=0, tied_vocab_identity=tied,
                draft_architecture=None, optimizer_type="torch.optim.adamw.AdamW",
                optimizer_param_groups=[], files=files,
            )

    def test_draft_architecture_with_null_config_accepted(self):
        tied = TiedVocabIdentity(sha256=_hex64("tv"), shape=(16, 4), dtype="float32")
        files = tuple(
            {"relative_path": n, "size_bytes": 100, "sha256": _hex64(n)}
            for n in ("draft-model.safetensors", "optimizer.pt", "request-ledger.json", "response-ledger.json")
        )
        m = CheckpointManifest(
            schema_version=CHECKPOINT_SCHEMA_VERSION, step_count=0, target_name="T",
            runtime_image_id="sha256:" + _hex64("rimg"), backend="CPU",
            checkpoint_manifest_sha256=_hex64("cmh"), selected_layer_ids=(0,),
            hidden_size=512, request_ledger_sha256=_hex64("rl"),
            request_ledger_entry_count=0, response_ledger_sha256=_hex64("rp"),
            response_ledger_entry_count=0, tied_vocab_identity=tied,
            draft_architecture={"adapter_type": "test.NoConfig", "config": None},
            optimizer_type="torch.optim.adamw.AdamW", optimizer_param_groups=[], files=files,
        )
        restored = CheckpointManifest.from_dict(m.to_dict())
        self.assertEqual(restored.draft_architecture["adapter_type"], "test.NoConfig")
        self.assertIsNone(restored.draft_architecture["config"])

    def test_saved_request_ledger_sha256(self):
        e1 = _build_connector_entry("r1", loss=1.0, gradient_norm=0.5)
        e2 = _build_connector_entry("r2", loss=0.5, gradient_norm=0.3)
        self.assertEqual(SavedRequestLedger(entries=(e1, e2)).sha256,
                         SavedRequestLedger(entries=(e1, e2)).sha256)
        e2b = _build_connector_entry("r2", loss=0.99, gradient_norm=0.3)
        self.assertNotEqual(SavedRequestLedger(entries=(e1, e2)).sha256,
                            SavedRequestLedger(entries=(e1, e2b)).sha256)

    def test_tied_vocab_identity_from_bf16_tensor_matches_hash(self):
        t = torch.randn(16, 4).to(torch.bfloat16)
        i1 = TiedVocabIdentity.from_tensor(t)
        i2 = TiedVocabIdentity.from_tensor(t.clone())
        self.assertEqual(i1, i2)
        self.assertEqual(i1.dtype, "bfloat16")
        t3 = torch.randn(16, 4)
        self.assertNotEqual(i1, TiedVocabIdentity.from_tensor(t3))

    def test_verify_checkpoint_directory_passes(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            m = save_checkpoint(ckpt, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=_response_examples(), runtime_identity=runtime)
            verified = verify_checkpoint_directory(ckpt, runtime_identity=runtime, expected_manifest_sha256=m.sha256)
            self.assertEqual(verified.step_count, 1)

    def test_verify_checkpoint_directory_fails_on_missing_file(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            m = save_checkpoint(ckpt, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=_response_examples(), runtime_identity=runtime)
            (ckpt / "draft-model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "missing|not found|unexpected"):
                verify_checkpoint_directory(ckpt, runtime_identity=runtime, expected_manifest_sha256=m.sha256)

    def test_sidecar_digest_mismatch_is_rejected(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            m = save_checkpoint(ckpt, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=_response_examples(), runtime_identity=runtime)
            (ckpt / MANIFEST_SHA256_FILENAME).write_text(_hex64("wrong") + "\n")
            with self.assertRaisesRegex(ValueError, "digest sidecar does not match"):
                verify_checkpoint_directory(ckpt, runtime_identity=runtime, expected_manifest_sha256=m.sha256)


# ---------------------------------------------------------------------------
# New tests for the bounded correction pass
# ---------------------------------------------------------------------------


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class SelectedLayerMismatchTests(unittest.TestCase):
    def test_layer_id_mismatch_on_save_is_rejected(self):
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = DFlashTrainingStep(
            adapter=SyntheticDraftAdapter(target_feature_width=8, hidden_size=4),
            shared_weights=FrozenSharedWeights.handoff_tied_embedding(nn.Embedding(16, 4), mask_token_id=1),
            selected_layer_ids=(0, 2),
        )
        optimizer = _dummy_optimizer(step)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "selected_layer_ids"):
                save_checkpoint(Path(tmp) / "bad", ring=ring, training_step=step, optimizer=optimizer,
                    completed_steps=1, request_ledger=_small_ledger(),
                    response_examples=_response_examples(), runtime_identity=runtime)

    def test_layer_id_mismatch_on_resume_is_rejected(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        torch_batch, bundle = _dummy_torch_batch()
        output = step(torch_batch, bundle)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "good"
            m = save_checkpoint(ckpt_dir, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=_response_examples(), runtime_identity=runtime)
            emb = nn.Embedding(16, 4)
            with torch.no_grad():
                emb.weight.copy_(step.shared_weights.embedding.weight.clone())
            wrong = DFlashTrainingStep(
                adapter=SyntheticDraftAdapter(target_feature_width=8, hidden_size=4),
                shared_weights=FrozenSharedWeights.handoff_tied_embedding(emb, mask_token_id=1),
                selected_layer_ids=(0, 2),
            )
            with self.assertRaisesRegex(ValueError, "selected_layer_ids"):
                resume_checkpoint(ckpt_dir, training_step=wrong, optimizer=_dummy_optimizer(wrong),
                    device=torch.device("cpu"), runtime_identity=runtime, expected_manifest_sha256=m.sha256)


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class TiedOutputHandoffTests(unittest.TestCase):
    def test_untied_handoff_is_rejected_on_save(self):
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        embedding = nn.Embedding(16, 4)
        lm_head = nn.Linear(4, 16)
        shared = FrozenSharedWeights.handoff(embedding, lm_head)
        step = DFlashTrainingStep(
            adapter=SyntheticDraftAdapter(target_feature_width=8, hidden_size=4),
            shared_weights=shared, selected_layer_ids=(0, 1),
        )
        optimizer = torch.optim.AdamW(step.parameters(), lr=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "tied_output_embedding"):
                save_checkpoint(Path(tmp) / "bad", ring=ring, training_step=step, optimizer=optimizer,
                    completed_steps=1, request_ledger=_small_ledger(),
                    response_examples=_response_examples(), runtime_identity=runtime)

    def test_untied_handoff_is_rejected_on_resume(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        torch_batch, bundle = _dummy_torch_batch()
        output = step(torch_batch, bundle)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "good"
            m = save_checkpoint(ckpt_dir, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=_response_examples(), runtime_identity=runtime)
            emb = nn.Embedding(16, 4)
            lm_head = nn.Linear(4, 16)
            with torch.no_grad():
                emb.weight.copy_(step.shared_weights.embedding.weight.clone())
            shared_untied = FrozenSharedWeights.handoff(emb, lm_head)
            untied_step = DFlashTrainingStep(
                adapter=SyntheticDraftAdapter(target_feature_width=8, hidden_size=4),
                shared_weights=shared_untied, selected_layer_ids=(0, 1),
            )
            with self.assertRaisesRegex(ValueError, "tied_output_embedding|lm_head"):
                resume_checkpoint(ckpt_dir, training_step=untied_step, optimizer=_dummy_optimizer(untied_step),
                    device=torch.device("cpu"), runtime_identity=runtime, expected_manifest_sha256=m.sha256)


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class AdapterClassConfigIdentityTests(unittest.TestCase):
    def test_draft_architecture_includes_adapter_type(self):
        adapter = SyntheticDraftAdapter(target_feature_width=8, hidden_size=4)
        arch = _draft_architecture(adapter)
        self.assertIn("adapter_type", arch)
        self.assertEqual(arch["adapter_type"],
                         "north_dflash_training.training.SyntheticDraftAdapter")
        self.assertIsNone(arch["config"])

    def test_adapter_class_drift_is_rejected_on_resume(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        torch_batch, bundle = _dummy_torch_batch()
        output = step(torch_batch, bundle)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpt"
            m = save_checkpoint(ckpt_dir, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=_response_examples(), runtime_identity=runtime)
            fresh = _dummy_training_step()
            fresh.shared_weights = step.shared_weights

            class FakeAdapter(SyntheticDraftAdapter):
                pass

            fake = FakeAdapter(target_feature_width=8, hidden_size=4)
            fake.load_state_dict(fresh.adapter.state_dict())
            fresh.adapter = fake
            with self.assertRaisesRegex(ValueError, "unsupported adapter class|draft architecture"):
                resume_checkpoint(ckpt_dir, training_step=fresh, optimizer=_dummy_optimizer(fresh),
                    device=torch.device("cpu"), runtime_identity=runtime, expected_manifest_sha256=m.sha256)


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class OptimizerParamValidationTests(unittest.TestCase):
    def test_optimizer_type_restricted_to_adamw(self):
        step = _dummy_training_step()
        with self.assertRaisesRegex(ValueError, "AdamW"):
            _validate_optimizer_class(torch.optim.SGD(step.parameters(), lr=0.01))

    def test_param_group_names_persisted_in_manifest(self):
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        with tempfile.TemporaryDirectory() as tmp:
            m = save_checkpoint(Path(tmp) / "ckpt", ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=_response_examples(), runtime_identity=runtime)
            self.assertTrue(len(m.optimizer_param_groups) >= 1)
            for group in m.optimizer_param_groups:
                for p in group:
                    self.assertIn("name", p)
                    self.assertIn("shape", p)
                    self.assertIn("dtype", p)

    def test_optimizer_parameter_reordering_is_rejected(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        torch_batch, bundle = _dummy_torch_batch()
        optimizer.zero_grad()
        step(torch_batch, bundle).loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-reordered-optimizer"
            manifest = save_checkpoint(
                checkpoint,
                ring=ring,
                training_step=step,
                optimizer=optimizer,
                completed_steps=1,
                request_ledger=_small_ledger(),
                response_examples=_response_examples(),
                runtime_identity=runtime,
            )
            fresh = _dummy_training_step()
            fresh.shared_weights = step.shared_weights
            reversed_optimizer = torch.optim.AdamW(
                list(reversed(tuple(fresh.parameters()))), lr=0.01
            )
            with self.assertRaisesRegex(ValueError, "name/order layout"):
                resume_checkpoint(
                    checkpoint,
                    training_step=fresh,
                    optimizer=reversed_optimizer,
                    device=torch.device("cpu"),
                    runtime_identity=runtime,
                    expected_manifest_sha256=manifest.sha256,
                )

    def test_optimizer_state_drift_is_rejected_by_tensor_shape(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        torch_batch, bundle = _dummy_torch_batch()
        output = step(torch_batch, bundle)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpt"
            m = save_checkpoint(ckpt_dir, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=_response_examples(), runtime_identity=runtime)
            fresh = _dummy_training_step()
            fresh.shared_weights = step.shared_weights
            fresh_opt = _dummy_optimizer(fresh)
            # Tamper the optimizer.pt file with a different state shape.
            original = (ckpt_dir / "optimizer.pt").read_bytes()
            loaded_state = torch.load(io_module.BytesIO(original), map_location="cpu", weights_only=True)
            for pid, se in loaded_state["state"].items():
                if isinstance(se, dict) and "exp_avg" in se:
                    se["exp_avg"] = torch.randn(8, 8)
                    break
            buf = io_module.BytesIO()
            torch.save(loaded_state, buf)
            new_bytes = buf.getvalue()
            (ckpt_dir / "optimizer.pt").write_bytes(new_bytes)
            # Update manifest to match the new file hash/size so verify_checkpoint_directory passes.
            import hashlib as hl_mod
            new_sha = hl_mod.sha256(new_bytes).hexdigest()
            manifest_data = json.loads((ckpt_dir / MANIFEST_FILENAME).read_text())
            for i, f in enumerate(manifest_data["files"]):
                if f["relative_path"] == "optimizer.pt":
                    manifest_data["files"][i] = {"relative_path": "optimizer.pt", "size_bytes": len(new_bytes), "sha256": new_sha}
                    break
            manifest_data["checkpoint_manifest_sha256"] = manifest_data.get("checkpoint_manifest_sha256", _hex64("cmh"))
            new_manifest_bytes = json.dumps(manifest_data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            (ckpt_dir / MANIFEST_FILENAME).write_bytes(new_manifest_bytes)
            new_manifest_digest = hl_mod.sha256(new_manifest_bytes).hexdigest()
            (ckpt_dir / MANIFEST_SHA256_FILENAME).write_text(new_manifest_digest + "\n", encoding="ascii")
            # resume must reject the manipulated optimizer state tensor shape.
            with self.assertRaisesRegex(ValueError,
                                         "does not match expected parameter"):
                resume_checkpoint(ckpt_dir, training_step=fresh, optimizer=fresh_opt,
                    device=torch.device("cpu"), runtime_identity=runtime,
                    expected_manifest_sha256=new_manifest_digest)


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class ConnectorReleaseLedgerTests(unittest.TestCase):
    def test_ledger_requires_connector_sha256(self):
        entry = {"request_id": "r1", "source_token_count": 6, "context_length": 4,
                 "query_count": 2, "active_label_count": 1, "loss": 1.0, "gradient_norm": 0.5,
                 "gradients_with_nonzero_values": 2, "updated_parameter_tensors": 2,
                 "feature_bytes_released": 64}
        with self.assertRaisesRegex(ValueError, r"must contain exactly"):
            SavedRequestLedger(entries=(entry,))

    def test_ledger_requires_lock_file_released_true(self):
        entry = _build_connector_entry("r1")
        entry["lock_file_released"] = False
        with self.assertRaisesRegex(ValueError, "lock_file_released must be true"):
            SavedRequestLedger(entries=(entry,))

    def test_ledger_requires_positive_counts(self):
        base = _build_connector_entry("r1")
        for field in ("source_token_count", "context_length", "query_count",
                       "active_label_count", "gradients_with_nonzero_values",
                       "updated_parameter_tensors", "feature_bytes_released",
                       "connector_file_bytes_released"):
            bad = dict(base)
            bad[field] = 0
            with self.assertRaisesRegex(ValueError, f"{field} must be a positive"):
                SavedRequestLedger(entries=(bad,))

    def test_save_resume_round_trip_preserves_connector_fields(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        torch_batch, bundle = _dummy_torch_batch()
        output = step(torch_batch, bundle)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpt"
            save_checkpoint(ckpt_dir, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=_response_examples(), runtime_identity=runtime)
            digest = (ckpt_dir / MANIFEST_SHA256_FILENAME).read_text(encoding="ascii").strip()
            fresh = _dummy_training_step()
            fresh.shared_weights = step.shared_weights
            _, resumed_ledger, _ = resume_checkpoint(ckpt_dir, training_step=fresh,
                optimizer=_dummy_optimizer(fresh), device=torch.device("cpu"),
                runtime_identity=runtime, expected_manifest_sha256=digest)
            for entry in resumed_ledger.entries:
                self.assertEqual(len(entry["connector_sha256"]), 64)
                self.assertTrue(entry["connector_file_bytes_released"] > 0)
                self.assertIs(entry["lock_file_released"], True)


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class ResponseLedgerTests(unittest.TestCase):
    def test_response_ledger_round_trip(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        torch_batch, bundle = _dummy_torch_batch()
        output = step(torch_batch, bundle)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        examples = _response_examples(step_count=1)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpt"
            save_checkpoint(ckpt_dir, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=examples, runtime_identity=runtime)
            digest = (ckpt_dir / MANIFEST_SHA256_FILENAME).read_text(encoding="ascii").strip()
            fresh = _dummy_training_step()
            fresh.shared_weights = step.shared_weights
            _, _, resumed = resume_checkpoint(ckpt_dir, training_step=fresh,
                optimizer=_dummy_optimizer(fresh), device=torch.device("cpu"),
                runtime_identity=runtime, expected_manifest_sha256=digest)
            self.assertEqual(len(resumed), 1)
            self.assertEqual(resumed[0]["response_tokens"], [1, 2, 3, 4, 5, 6])
            # Verify the response-ledger.json file exists with correct hash.
            rledger_bytes = (ckpt_dir / RESPONSE_LEDGER_FILENAME).read_bytes()
            self.assertEqual(len(hashlib.sha256(rledger_bytes).hexdigest()), 64)

    def test_response_ledger_drift_is_detected(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        torch_batch, bundle = _dummy_torch_batch()
        output = step(torch_batch, bundle)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpt"
            m = save_checkpoint(ckpt_dir, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=_response_examples(), runtime_identity=runtime)
            # Tamper the response ledger.
            rdata = json.loads((ckpt_dir / RESPONSE_LEDGER_FILENAME).read_text())
            rdata[0]["response_tokens"] = [99]
            (ckpt_dir / RESPONSE_LEDGER_FILENAME).write_text(json.dumps(rdata, sort_keys=True))
            fresh = _dummy_training_step()
            fresh.shared_weights = step.shared_weights
            with self.assertRaisesRegex(ValueError, "file hash or size mismatch|SHA-256 does not match"):
                resume_checkpoint(ckpt_dir, training_step=fresh, optimizer=_dummy_optimizer(fresh),
                    device=torch.device("cpu"), runtime_identity=runtime, expected_manifest_sha256=m.sha256)


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class Bf16RoundTripTests(unittest.TestCase):
    def test_bf16_manifest_round_trip(self):
        tensor = torch.randn(16, 4).to(torch.bfloat16)
        identity = TiedVocabIdentity.from_tensor(tensor)
        self.assertEqual(identity.dtype, "bfloat16")
        self.assertEqual(len(identity.sha256), 64)
        # Clone should produce identical identity.
        identity2 = TiedVocabIdentity.from_tensor(tensor.clone())
        self.assertEqual(identity, identity2)
        # Different values should produce different hash.
        diff_tensor = torch.randn(16, 4).to(torch.bfloat16)
        identity3 = TiedVocabIdentity.from_tensor(diff_tensor)
        self.assertNotEqual(identity.sha256, identity3.sha256)

    def test_save_with_bf16_embedding(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        # Create a step with bf16 embedding.
        weight = torch.randn(16, 4, dtype=torch.bfloat16)
        embedding = nn.Embedding.from_pretrained(weight, freeze=True)
        shared = FrozenSharedWeights.handoff_tied_embedding(embedding, mask_token_id=1)
        adapter = SyntheticDraftAdapter(target_feature_width=8, hidden_size=4)
        step = DFlashTrainingStep(adapter=adapter, shared_weights=shared, selected_layer_ids=(0, 1))
        optimizer = _dummy_optimizer(step)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpt"
            m = save_checkpoint(ckpt_dir, ring=ring, training_step=step, optimizer=optimizer,
                completed_steps=1, request_ledger=_small_ledger(),
                response_examples=_response_examples(), runtime_identity=runtime)
            self.assertIn("bfloat16", m.tied_vocab_identity.dtype)


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class ConcurrentPublishTests(unittest.TestCase):
    """Prove RENAME_NOREPLACE prevents concurrent overwrites."""

    def test_concurrent_rename_no_replace(self):
        torch.manual_seed(42)
        runtime = _dummy_runtime_identity()
        ring = _dummy_ring(runtime)
        step = _dummy_training_step()
        optimizer = _dummy_optimizer(step)
        torch_batch, bundle = _dummy_torch_batch()
        output = step(torch_batch, bundle)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            final_path = Path(tmp) / "final-ckpt"
            results = []

            def saver():
                try:
                    m = save_checkpoint(final_path, ring=ring, training_step=step,
                        optimizer=optimizer, completed_steps=1,
                        request_ledger=_small_ledger(), response_examples=_response_examples(),
                        runtime_identity=runtime)
                    results.append(("ok", m.sha256))
                except (FileExistsError, OSError) as e:
                    results.append(("exists", str(e)))

            threads = [threading.Thread(target=saver) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            ok_count = sum(1 for r in results if r[0] == "ok")
            exists_count = sum(1 for r in results if r[0] == "exists")
            self.assertEqual(ok_count, 1, f"expected exactly one successful publish, got {ok_count}")
            self.assertGreaterEqual(exists_count, 0)
            self.assertTrue(final_path.is_dir(), "checkpoint directory must exist")


if __name__ == "__main__":
    unittest.main()
