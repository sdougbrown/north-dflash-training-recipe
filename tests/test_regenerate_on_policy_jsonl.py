import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "regenerate_on_policy_jsonl.py"
SPEC = importlib.util.spec_from_file_location("regenerate_on_policy_jsonl", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RegenerateOnPolicyJsonlTests(unittest.TestCase):
    def test_build_sample_preserves_exact_boundary(self):
        sample = MODULE.build_sample(
            source_index=3,
            prompt="Fix it",
            response={
                "prompt_token_ids": [10, 11],
                "choices": [
                    {
                        "token_ids": [12, 13, 14],
                        "finish_reason": "stop",
                        "message": {"content": "Done"},
                    }
                ],
                "usage": {"completion_tokens": 3},
            },
            endpoint="http://target/v1/chat/completions",
            sampling={"temperature": 0.6, "top_p": 0.95, "seed": 0},
        )
        self.assertEqual(sample["input_ids"], [10, 11, 12, 13, 14])
        self.assertEqual(sample["loss_mask"], [0, 0, 1, 1, 1])
        self.assertEqual(sample["metadata"]["source_index"], 3)
        self.assertEqual(sample["conversations"][-1]["content"], "Done")

    def test_build_sample_rejects_missing_token_ids(self):
        with self.assertRaisesRegex(ValueError, "completion token_ids"):
            MODULE.build_sample(
                source_index=0,
                prompt="x",
                response={
                    "prompt_token_ids": [1],
                    "choices": [{"message": {"content": "y"}}],
                },
                endpoint="http://target",
                sampling={},
            )

    def test_load_prompts_rejects_empty_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            path.write_text(json.dumps({"instruction": ""}) + "\n")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                MODULE.load_prompts(path, "instruction")

    def test_stable_id_includes_source_index(self):
        self.assertNotEqual(MODULE.stable_id(1, "same"), MODULE.stable_id(2, "same"))


if __name__ == "__main__":
    unittest.main()
