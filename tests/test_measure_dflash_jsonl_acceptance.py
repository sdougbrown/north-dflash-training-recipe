import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location(
        "measure_dflash_jsonl_acceptance", SCRIPTS / "measure_dflash_jsonl_acceptance.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.remove(str(SCRIPTS))


class MeasureDflashJsonlAcceptanceTests(unittest.TestCase):
    def test_load_prompt_slice_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text(
                "".join(
                    json.dumps({"instruction": f"prompt-{index}"}) + "\n"
                    for index in range(5)
                )
            )
            self.assertEqual(
                module.load_prompt_slice(path, "instruction", 2, 2),
                [(2, "prompt-2"), (3, "prompt-3")],
            )

    def test_load_prompt_slice_refuses_short_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text(json.dumps({"instruction": "only"}) + "\n")
            with self.assertRaisesRegex(ValueError, "requested 2"):
                module.load_prompt_slice(path, "instruction", 0, 2)


if __name__ == "__main__":
    unittest.main()
