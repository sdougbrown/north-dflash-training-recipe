import json
from pathlib import Path
import tempfile
import unittest

from north_dflash_training.checkpoint_identity import (
    CheckpointIdentityManifest,
    build_checkpoint_identity_manifest,
    sha256_file,
    verify_checkpoint_identity,
)


class CheckpointIdentityTests(unittest.TestCase):
    def _fixture(self, root: Path) -> None:
        (root / "config.json").write_text('{"model_type":"fixture"}')
        (root / "model-00001-of-00002.safetensors").write_bytes(b"first-small-shard")
        (root / "model-00002-of-00002.safetensors").write_bytes(b"second-small-shard")
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": 35},
            "weight_map": {
                "layer.a": "model-00001-of-00002.safetensors",
                "layer.b": "model-00002-of-00002.safetensors",
                "layer.c": "model-00001-of-00002.safetensors",
            },
        }))

    def test_hashes_config_index_and_unique_small_fixture_shards_incrementally(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            manifest = build_checkpoint_identity_manifest(root, chunk_bytes=3)
            self.assertEqual(manifest.config.relative_path, "config.json")
            self.assertEqual(manifest.index.relative_path, "model.safetensors.index.json")
            self.assertEqual([item.relative_path for item in manifest.shards], [
                "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
            ])
            self.assertEqual(manifest.shards[0].sha256, sha256_file(root / manifest.shards[0].relative_path, chunk_bytes=1))
            self.assertEqual(CheckpointIdentityManifest.from_dict(manifest.to_dict()), manifest)
            verify_checkpoint_identity(root, manifest, chunk_bytes=2)

    def test_verification_detects_a_shard_hash_mismatch_without_tensor_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            manifest = build_checkpoint_identity_manifest(root, chunk_bytes=4)
            (root / "model-00002-of-00002.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                verify_checkpoint_identity(root, manifest, chunk_bytes=2)

    def test_rejects_index_paths_outside_checkpoint_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"bad": "../outside.safetensors"},
            }))
            with self.assertRaisesRegex(ValueError, "escapes root"):
                build_checkpoint_identity_manifest(root)


if __name__ == "__main__":
    unittest.main()
