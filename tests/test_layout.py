from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from north_dflash_training import (
    ResponseExample,
    build_training_batch_layout,
    sample_anchor_blocks,
)
from north_dflash_training.candidate import audit_mask_token
from north_dflash_training.teacher import teacher_feature_manifest_from_config, validate_teacher_feature_manifest


class SparseLayoutTests(unittest.TestCase):
    def test_concatenation_carries_fields_and_is_bidirectional_per_block(self):
        example = ResponseExample((50,), tuple(range(20)))
        sampled = sample_anchor_blocks(example, block_size=4, max_anchors=2, mask_token_id=1, seed=4)
        layout = build_training_batch_layout(sampled, gamma=2.0)

        self.assertEqual(layout.num_queries, 8)
        self.assertEqual(layout.block_ids, (0, 0, 0, 0, 1, 1, 1, 1))
        self.assertEqual(layout.anchor_positions[:4], (sampled.blocks[0].absolute_anchor_position,) * 4)
        self.assertEqual(layout.anchor_positions[4:], (sampled.blocks[1].absolute_anchor_position,) * 4)
        self.assertEqual(
            layout.absolute_query_positions[:4],
            tuple(sampled.blocks[0].absolute_anchor_position + offset for offset in range(4)),
        )
        self.assertEqual(layout.labels[1:4], sampled.blocks[0].labels[1:])
        self.assertEqual(layout.loss_mask, (False, True, True, True) * 2)
        self.assertEqual(layout.position_weights[0], 1.0)  # ignored anchor bookkeeping
        self.assertEqual(layout.position_weights[1], 1.0)  # first prediction
        self.assertGreater(layout.position_weights[1], layout.position_weights[2])
        self.assertEqual(layout.labels[0], -100)
        self.assertNotIn(-100, layout.labels[1:4])
        self.assertTrue(layout.can_query_see_query(0, 3))
        self.assertTrue(layout.can_query_see_query(3, 0))
        self.assertFalse(layout.can_query_see_query(0, 4))
        self.assertFalse(layout.can_query_see_query(7, 0))

    def test_target_context_includes_clean_prefix_through_anchor_without_future(self):
        example = ResponseExample((10, 11), tuple(range(30)))
        sampled = sample_anchor_blocks(example, block_size=5, max_anchors=3, mask_token_id=1, seed=0)
        layout = build_training_batch_layout(sampled, gamma=2.0)
        for query_index, anchor in enumerate(layout.anchor_positions):
            context = layout.target_context_for_query(query_index)
            self.assertEqual(context, tuple(range(anchor + 1)))
            self.assertEqual(context[-1], anchor)
            self.assertNotIn(anchor + 1, context)
        layout.validate()

    def test_layout_rejects_shifted_loss_weights_and_non_ignored_anchor_label(self):
        sampled = sample_anchor_blocks(ResponseExample((), tuple(range(10))), block_size=4, max_anchors=1, mask_token_id=1)
        layout = build_training_batch_layout(sampled, gamma=2.0)
        shifted = replace(layout, position_weights=(1.0, 0.5, 0.25, 0.125))
        with self.assertRaisesRegex(ValueError, "loss-decay"):
            shifted.validate()
        bad_labels = replace(layout, labels=(0,) + layout.labels[1:])
        with self.assertRaisesRegex(ValueError, "anchors must be ignored"):
            bad_labels.validate()

    def test_ordering_is_deterministic(self):
        example = ResponseExample((), tuple(range(10)))
        first = build_training_batch_layout(sample_anchor_blocks(example, block_size=3, max_anchors=4, mask_token_id=1, seed=9), gamma=2.0)
        second = build_training_batch_layout(sample_anchor_blocks(example, block_size=3, max_anchors=4, mask_token_id=1, seed=9), gamma=2.0)
        self.assertEqual(first, second)


class MaskAuditAndTeacherManifestTests(unittest.TestCase):
    def test_mask_audit_derives_id_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            path.write_text(json.dumps({
                "model": {"vocab": {"<MASK_TOKEN>": 1, "x": 2}},
                "added_tokens": [{"id": 1, "content": "<MASK_TOKEN>", "special": True}],
            }))
            audit = audit_mask_token(path, expected_vocab_size=262144)
            self.assertTrue(audit["valid"])
            self.assertEqual(audit["id"], 1)
            path.write_text(json.dumps({
                "model": {"vocab": {"<MASK_TOKEN>": 1}},
                "added_tokens": [{"id": 7, "content": "<MASK_TOKEN>", "special": True}],
            }))
            self.assertFalse(audit_mask_token(path)["valid"])

    def test_teacher_manifest_is_identity_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "architectures": ["Cohere2MoeForCausalLM"],
                "model_type": "cohere2_moe",
                "quantization_config": {
                    "bits": 4,
                    "group_size": 32,
                    "data_type": "int",
                    "quant_method": "gptq",
                    "provider": "auto-round",
                    "autoround_version": "0.13.0",
                },
                "num_hidden_layers": 49,
                "num_experts": 128,
                "num_experts_per_tok": 8,
            }))
            manifest = teacher_feature_manifest_from_config(path, [1, 12, 24])
        self.assertEqual(manifest.selected_layer_ids, (1, 12, 24))
        data = manifest.to_dict()
        self.assertEqual(data["quantization_identity"]["bits"], 4)
        self.assertEqual(data["selected_layer_id_convention"], "zero_based_transformer_block_index")
        self.assertEqual(len(data["teacher_config_sha256"]), 64)
        self.assertEqual(data["checkpoint_identity_status"], "not_verified")
        self.assertEqual(data["extraction_status"], "not_implemented")
        validate_teacher_feature_manifest(data)


if __name__ == "__main__":
    unittest.main()
