"""Tiny CPU fixtures for the guarded runtime-probe artifact."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from north_dflash_training.runtime_probe import (
    FULL_GEOMETRY,
    ProbeDimensions,
    ProbeGeometry,
    SMOKE_GEOMETRY,
    _assert_reference_state_is_vllm_loadable,
    build_probe_config,
    estimate_weight_bytes,
    verify_vllm_loader_contract,
)


class RuntimeProbeConfigTests(unittest.TestCase):
    def test_smoke_keeps_north_runtime_boundary_and_explicit_layer_convention(self):
        config = build_probe_config()
        self.assertEqual(config["architectures"], ["DFlashDraftModel"])
        self.assertEqual(config["model_type"], "qwen3")
        self.assertEqual(config["hidden_size"], 2048)
        self.assertEqual(config["intermediate_size"], 6144)
        self.assertEqual(config["num_attention_heads"], 32)
        self.assertEqual(config["num_key_value_heads"], 4)
        self.assertEqual(config["head_dim"], 128)
        self.assertEqual(config["vocab_size"], 262144)
        self.assertEqual(config["num_target_layers"], 49)
        self.assertEqual(config["dflash_config"]["mask_token_id"], 1)
        self.assertEqual(config["dflash_config"]["target_layer_ids"], [24])
        self.assertEqual(config["eagle_aux_hidden_state_layer_ids"], [25])
        self.assertEqual(config["block_size"], 2)
        self.assertEqual(config["runtime_probe"]["not_for_acceptance"], True)

    def test_full_is_explicit_geometry_and_larger(self):
        config = build_probe_config("full")
        self.assertEqual(config["num_hidden_layers"], FULL_GEOMETRY.draft_layers)
        self.assertEqual(config["dflash_config"]["target_layer_ids"], list(FULL_GEOMETRY.target_layer_ids))
        self.assertEqual(config["eagle_aux_hidden_state_layer_ids"], [2, 13, 25, 36, 47])
        self.assertEqual(config["block_size"], 16)
        self.assertGreater(estimate_weight_bytes(config), estimate_weight_bytes(build_probe_config("smoke")))

    def test_tiny_fixture_config_preserves_dflash_shape_rules(self):
        dimensions = ProbeDimensions(
            hidden_size=16,
            intermediate_size=24,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
            num_target_layers=4,
            max_position_embeddings=32,
            rope_theta=10000,
        )
        # The production smoke uses layer 24; this tiny fixture keeps the
        # same one-layer/one-feature/block-size relation within four blocks.
        fixture = ProbeGeometry("fixture", 1, (1,), 2)
        config = build_probe_config(fixture, dimensions=dimensions)
        self.assertEqual(config["num_hidden_layers"], SMOKE_GEOMETRY.draft_layers)
        self.assertEqual(config["dflash_config"]["target_layer_ids"], [1])

    def test_static_vllm_contract_is_provable_from_local_checkout(self):
        source = Path("/home/douglasbrown/Code/_worktrees/vllm-north-dflash")
        if not source.is_dir():
            self.skipTest("local vLLM evidence checkout is unavailable")
        evidence = verify_vllm_loader_contract(source)
        self.assertIn("vllm_revision", evidence)


HAS_REFERENCE = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors", "dflash")
)


@unittest.skipUnless(HAS_REFERENCE, "requires existing torch/transformers/safetensors/dflash reference environment")
class RuntimeProbeReferenceFixtureTests(unittest.TestCase):
    def test_save_pretrained_tiny_fixture_has_only_proved_vllm_names(self):
        from dflash.model import DFlashDraftModel
        from safetensors import safe_open
        from transformers import Qwen3Config

        config = build_probe_config()
        # Do not materialize North-shaped tensors in tests. Replace only this
        # fixture's model geometry while retaining the runtime config contract.
        config.update(
            hidden_size=16,
            intermediate_size=24,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
            draft_vocab_size=32,
            target_hidden_size=16,
            num_target_layers=4,
            max_position_embeddings=32,
            rope_theta=10000,
            rope_parameters={"rope_type": "default", "rope_theta": 10000},
        )
        config["dflash_config"]["target_layer_ids"] = [1]
        config["eagle_aux_hidden_state_layer_ids"] = [2]
        model = DFlashDraftModel(Qwen3Config(**config))
        state_keys = _assert_reference_state_is_vllm_loadable(model)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            with safe_open(str(Path(directory) / "model.safetensors"), framework="pt", device="cpu") as saved:
                self.assertEqual(sorted(saved.keys()), state_keys)
            loaded = json.loads((Path(directory) / "config.json").read_text())
        self.assertEqual(loaded["architectures"], ["DFlashDraftModel"])
        self.assertNotIn("embed_tokens", " ".join(state_keys))
        self.assertNotIn("lm_head", " ".join(state_keys))


if __name__ == "__main__":
    unittest.main()
