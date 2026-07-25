import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RetainedPilotConfigTests(unittest.TestCase):
    def test_pilot40_partitions_all_fifty_cases_without_overlap(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "north-fp8-retained-pilot40.json").read_text()
        )
        completed = config["completed_initial_cases"]
        continuation = [case for chunk in config["continuation_chunks"] for case in chunk]
        holdout = config["acceptance_holdout_cases"]
        self.assertEqual([len(chunk) for chunk in config["continuation_chunks"]], [8] * 4)
        self.assertEqual((len(completed), len(continuation), len(holdout)), (8, 32, 10))
        all_cases = completed + continuation + holdout
        self.assertEqual(len(all_cases), 50)
        self.assertEqual(len(set(all_cases)), 50)
        self.assertEqual(
            config["optimization"]["total_cumulative_updates"],
            len(completed) + len(continuation),
        )


if __name__ == "__main__":
    unittest.main()
