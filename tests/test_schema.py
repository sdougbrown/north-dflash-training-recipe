import unittest

from north_dflash_training.schema import ResponseExample


class ResponseExampleTests(unittest.TestCase):
    def test_round_trip(self):
        example = ResponseExample((1, 2), (3, 4), {"id": "x"})
        self.assertEqual(ResponseExample.from_json(example.to_json()), example)

    def test_required_and_token_validation(self):
        with self.assertRaises(ValueError):
            ResponseExample.from_mapping({"schema_version": 1, "prompt_tokens": [], "response_tokens": []})
        with self.assertRaises(ValueError):
            ResponseExample((1,), (True,))
        with self.assertRaises(ValueError):
            ResponseExample.from_mapping({"schema_version": 2, "prompt_tokens": [], "response_tokens": [1]})


if __name__ == "__main__":
    unittest.main()
