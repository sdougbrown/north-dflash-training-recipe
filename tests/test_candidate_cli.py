import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

from north_dflash_training.candidate import build_target_layer_ids, derive_north_candidate
from north_dflash_training.cli import main


class CandidateAndCliTests(unittest.TestCase):
    def test_layer_spread_matches_reference_formula(self):
        self.assertEqual(build_target_layer_ids(49, 5), [1, 12, 24, 35, 46])
        self.assertEqual(build_target_layer_ids(49, 1), [24])
        with self.assertRaisesRegex(ValueError, "more distinct target layers"):
            build_target_layer_ids(4, 2)

    def test_candidate_reads_config_only_and_marks_unknowns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({
                "architectures": ["Cohere2MoeForCausalLM"],
                "model_type": "cohere2_moe",
                "num_hidden_layers": 9,
                "hidden_size": 16,
                "head_dim": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "layer_norm_eps": 1e-5,
                "quantization_config": {
                    "bits": 4,
                    "group_size": 32,
                    "data_type": "int",
                    "quant_method": "gptq",
                    "provider": "auto-round",
                    "autoround_version": "0.13.0",
                },
                "num_experts": 128,
                "num_experts_per_tok": 8,
            }))
            (root / "tokenizer.json").write_text("{}")
            (root / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "TokenizersBackend"}))
            candidate = derive_north_candidate(root / "config.json", root / "tokenizer_config.json")
        self.assertTrue(candidate["deployment_target"]["requires_exact_expert_only_autogptq"])
        self.assertIsNone(candidate["derived_draft_candidate"]["mask_token_id"])
        self.assertEqual(candidate["derived_draft_candidate"]["rms_norm_eps"], 1e-6)
        self.assertEqual(candidate["derived_draft_candidate"]["reference_hidden_state_indices_candidate"], [2, 3, 5, 6, 7])
        self.assertIn("AutoGPTQ", candidate["unresolved_choices"][2])

    def test_cli_dry_run_is_cpu_only(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["dry-run", "--response-length", "8", "--block-size", "8", "--max-anchors", "1"]), 0)


if __name__ == "__main__":
    unittest.main()
