import importlib.util
import json
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location(
        "measure_jsonl_generation", SCRIPTS / "measure_jsonl_generation.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.remove(str(SCRIPTS))


class MeasureJsonlGenerationTests(unittest.TestCase):
    def test_token_root_is_canonical_compact_json(self):
        token_ids = [[1, 2], [3]]
        self.assertEqual(
            module.token_id_root(token_ids),
            __import__("hashlib").sha256(
                json.dumps(token_ids, separators=(",", ":")).encode()
            ).hexdigest(),
        )

    def test_request_is_deterministic_and_ignores_eos(self):
        payload = module.request_payload("north", "prompt", 128, 7)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["max_completion_tokens"], 128)
        self.assertTrue(payload["ignore_eos"])
        self.assertTrue(payload["return_token_ids"])


if __name__ == "__main__":
    unittest.main()
