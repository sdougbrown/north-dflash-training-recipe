import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import save_file

from north_dflash_training.feature_stream import (
    BoundedFeatureRing,
    TeacherRuntimeIdentity,
    load_connector_feature_batch,
)


DIGEST = "a" * 64


class FeatureStreamTests(unittest.TestCase):
    def setUp(self):
        self.identity = TeacherRuntimeIdentity(
            target_name="NorthINT4Target",
            checkpoint_manifest_sha256=DIGEST,
            runtime_image_id="sha256:" + "b" * 64,
            backend="TRITON_WNA16_TP2",
            selected_layer_ids=(1, 7),
            hidden_size=3,
            prefix_caching_enabled=False,
        )
        self.tokens = torch.tensor([2, 11, 12, 13], dtype=torch.int64)
        self.hidden = torch.arange(4 * 2 * 3, dtype=torch.bfloat16).reshape(4, 2, 3)

    def _write_trace(self, root: Path, **tensors: torch.Tensor) -> Path:
        path = root / "trace.safetensors"
        save_file(tensors or {"hidden_states": self.hidden, "token_ids": self.tokens}, path)
        return path

    def test_loads_owned_ordered_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_trace(Path(directory))
            batch = load_connector_feature_batch(
                path,
                request_id="request-1",
                runtime_identity=self.identity,
                expected_token_ids=self.tokens,
            )
        self.assertEqual(batch.features.selected_layer_ids, (1, 7))
        self.assertEqual(tuple(batch.features.hidden_states[0].shape), (1, 4, 3))
        self.assertTrue(torch.equal(batch.features.hidden_states[0][0], self.hidden[:, 0]))
        self.assertTrue(torch.equal(batch.features.hidden_states[1][0], self.hidden[:, 1]))
        self.assertEqual(batch.feature_bytes, self.hidden.numel() * self.hidden.element_size())
        batch.validate()

    def test_rejects_prefix_cache_and_token_drift(self):
        with self.assertRaisesRegex(ValueError, "prefix caching disabled"):
            replace(self.identity, prefix_caching_enabled=True)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_trace(Path(directory))
            with self.assertRaisesRegex(ValueError, "request ledger"):
                load_connector_feature_batch(
                    path,
                    request_id="request-1",
                    runtime_identity=self.identity,
                    expected_token_ids=torch.tensor([2, 11, 99, 13]),
                )

    def test_rejects_shape_dtype_and_extra_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_trace(
                root,
                hidden_states=self.hidden.float(),
                token_ids=self.tokens,
            )
            with self.assertRaisesRegex(ValueError, "remain BF16"):
                load_connector_feature_batch(
                    path,
                    request_id="request-1",
                    runtime_identity=self.identity,
                    expected_token_ids=self.tokens,
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_trace(
                root,
                hidden_states=self.hidden,
                token_ids=self.tokens,
                unexpected=torch.ones(1),
            )
            with self.assertRaisesRegex(ValueError, "only hidden_states"):
                load_connector_feature_batch(
                    path,
                    request_id="request-1",
                    runtime_identity=self.identity,
                    expected_token_ids=self.tokens,
                )

    def test_ring_backpressure_is_atomic_and_forbids_mixed_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_trace(Path(directory))
            first = load_connector_feature_batch(
                path,
                request_id="request-1",
                runtime_identity=self.identity,
                expected_token_ids=self.tokens,
            )
            second = replace(first, request_id="request-2")
        ring = BoundedFeatureRing(
            runtime_identity=self.identity,
            max_items=1,
            max_tokens=4,
            max_bytes=first.feature_bytes,
        )
        ring.put(first)
        with self.assertRaises(BufferError):
            ring.put(second)
        self.assertEqual(len(ring), 1)
        self.assertEqual(ring.token_count, 4)
        self.assertEqual(ring.feature_bytes, first.feature_bytes)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ring.put(first)
        other = replace(
            second,
            runtime_identity=replace(self.identity, target_name="NorthFP8Target"),
        )
        with self.assertRaisesRegex(ValueError, "mixed target"):
            ring.put(other)
        self.assertIs(ring.popleft(), first)
        self.assertEqual((len(ring), ring.token_count, ring.feature_bytes), (0, 0, 0))
        ring.put(second)


if __name__ == "__main__":
    unittest.main()
