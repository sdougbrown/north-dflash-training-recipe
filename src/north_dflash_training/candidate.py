"""Derive a reviewable North candidate without loading model weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .teacher import teacher_feature_manifest_from_config


DEFAULT_TARGET_CONFIG = Path("/home/douglasbrown/Models/North-Mini-Code-1.0-int4-autoround-gptq-g32/config.json")
DEFAULT_TOKENIZER_CONFIG = Path("/home/douglasbrown/Models/North-Mini-Code-1.0-int4-autoround-gptq-g32/tokenizer_config.json")
DEFAULT_TOKENIZER_JSON = Path("/home/douglasbrown/Models/North-Mini-Code-1.0-int4-autoround-gptq-g32/tokenizer.json")
MASK_TOKEN_CONTENT = "<MASK_TOKEN>"


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int = 5) -> list[int]:
    """Match the reference dflash/model.py spread of distinct block indices."""
    if num_target_layers < 4 or num_draft_layers < 1:
        raise ValueError("target must have at least 4 layers and draft at least 1")
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    if num_draft_layers > end - start + 1:
        raise ValueError("draft requests more distinct target layers than the reference range contains")
    layer_ids = [int(round(start + i * (end - start) / (num_draft_layers - 1))) for i in range(num_draft_layers)]
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError("reference layer spread produced duplicate target layers")
    return layer_ids


def audit_mask_token(
    tokenizer_json_path: str | Path,
    *,
    expected_vocab_size: int | None = None,
) -> dict[str, Any]:
    """Derive and audit the mask ID from ``tokenizer.json`` only.

    The token spelling is the semantic contract; its numeric ID is never
    guessed.  Both the tokenizer model vocabulary and its special-token table
    must agree when present.  A missing or inconsistent tokenizer produces an
    invalid audit rather than a fallback ID.
    """
    path = Path(tokenizer_json_path)
    tokenizer = json.loads(path.read_text())
    vocab = tokenizer.get("model", {}).get("vocab", {})
    added_tokens = tokenizer.get("added_tokens", [])
    vocab_id = vocab.get(MASK_TOKEN_CONTENT) if isinstance(vocab, dict) else None
    special_entries = [
        item for item in added_tokens
        if isinstance(item, dict) and item.get("content") == MASK_TOKEN_CONTENT
    ]
    special_ids = {item.get("id") for item in special_entries}
    reasons: list[str] = []
    if not isinstance(vocab_id, int) or isinstance(vocab_id, bool) or vocab_id < 0:
        reasons.append("mask token is absent from tokenizer model vocabulary")
    if len(special_entries) != 1:
        reasons.append("mask token must have exactly one added-token entry")
    elif special_entries[0].get("special") is not True:
        reasons.append("mask token entry is not marked special")
    special_id = next(iter(special_ids), None)
    if isinstance(special_id, bool) or not isinstance(special_id, int) or special_id < 0:
        reasons.append("mask token added-token ID is not a non-negative integer")
    elif special_id != vocab_id:
        reasons.append("model vocabulary and added-token IDs disagree")
    if expected_vocab_size is not None and (
        not isinstance(expected_vocab_size, int)
        or expected_vocab_size < 1
        or not isinstance(vocab_id, int)
        or vocab_id < 0
        or vocab_id >= expected_vocab_size
    ):
        reasons.append("mask token ID is outside the target vocabulary")
    return {
        "token": MASK_TOKEN_CONTENT,
        "id": vocab_id if not reasons else None,
        "valid": not reasons,
        "source": str(path),
        "model_vocab_size": len(vocab) if isinstance(vocab, dict) else None,
        "target_vocab_size": expected_vocab_size,
        "model_vocab_id": vocab_id,
        "added_token_id": special_id,
        "reasons": reasons,
    }


def derive_north_candidate(
    target_config_path: str | Path = DEFAULT_TARGET_CONFIG,
    tokenizer_config_path: str | Path = DEFAULT_TOKENIZER_CONFIG,
    tokenizer_json_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read only JSON configs and return a candidate with audited choices explicit."""
    target_path = Path(target_config_path)
    tokenizer_path = Path(tokenizer_config_path)
    tokenizer_json = Path(tokenizer_json_path) if tokenizer_json_path is not None else target_path.parent / "tokenizer.json"
    target = json.loads(target_path.read_text())
    tokenizer = json.loads(tokenizer_path.read_text())
    layers = int(target["num_hidden_layers"])
    draft_layers = 5
    quant = target.get("quantization_config", {})
    vocab_size = target.get("vocab_size")
    mask_audit = audit_mask_token(tokenizer_json, expected_vocab_size=vocab_size)
    target_layer_ids = build_target_layer_ids(layers, draft_layers)
    teacher_manifest = teacher_feature_manifest_from_config(target_path, target_layer_ids)

    unresolved = [
        "Confirm Cohere2Moe output_hidden_states numbering before applying the reference layer_id + 1 convention.",
        "Verify every checkpoint shard before extraction; this manifest fingerprints config.json only and cannot prove teacher-weight identity.",
        "Implement exact AutoGPTQ expert-only teacher hidden-state extraction without dequantizing or changing deployment weights.",
        "Decide dense draft MLP versus any MoE-derived initializer; a five-layer dense draft is only a paper-shaped candidate.",
        "Define vLLM Cohere2Moe auxiliary-state plumbing and verify it against the running deployment integration.",
    ]
    if not mask_audit["valid"]:
        unresolved.append("Resolve the invalid <MASK_TOKEN> tokenizer audit before training.")

    return {
        "status": "CPU layout and bounded optional training-step contract tested; target/FlexAttention integration missing; not trainable",
        "deployment_target": {
            "model_path": str(target_path),
            "tokenizer_path": str(tokenizer_path),
            "architecture": target.get("architectures", [None])[0],
            "model_type": target.get("model_type"),
            "requires_exact_expert_only_autogptq": True,
            "precision_policy": "BF16 is permitted only as an initializer; final target remains int4 AutoGPTQ.",
            "quantization": {
                "bits": quant.get("bits"),
                "group_size": quant.get("group_size"),
                "data_type": quant.get("data_type"),
                "quant_method": quant.get("quant_method"),
                "provider": quant.get("provider"),
                "autoround_version": quant.get("autoround_version"),
                "num_experts": target.get("num_experts"),
                "num_experts_per_tok": target.get("num_experts_per_tok"),
            },
        },
        "derived_draft_candidate": {
            "hidden_size": target.get("hidden_size"),
            "head_dim": target.get("head_dim"),
            "num_attention_heads": target.get("num_attention_heads"),
            "num_key_value_heads": target.get("num_key_value_heads"),
            "hidden_act": target.get("hidden_act"),
            "rms_norm_eps": target.get("rms_norm_eps", target.get("layer_norm_eps")),
            "num_hidden_layers_candidate": draft_layers,
            "block_size_candidate": 16,
            "target_layer_ids_candidate": target_layer_ids,
            "target_layer_id_convention": "zero_based_transformer_block_index",
            "reference_hidden_state_indices_candidate": [layer_id + 1 for layer_id in target_layer_ids],
            "north_hidden_state_indexing_status": "unverified; Cohere2Moe runtime/extractor evidence required",
            "vocab_size": vocab_size,
            "max_position_embeddings": target.get("max_position_embeddings"),
            "shared_embedding_and_lm_head": "paper-aligned candidate; exact weight handoff unresolved",
            "mask_token_id": mask_audit["id"],
            "mask_token_audit": mask_audit,
        },
        "teacher_feature_manifest": teacher_manifest.to_dict(),
        "unresolved_choices": unresolved,
        "observed_tokenizer": {
            "tokenizer_class": tokenizer.get("tokenizer_class"),
            "eos_token": tokenizer.get("eos_token"),
            "pad_token": tokenizer.get("pad_token"),
            "local_files_only": tokenizer.get("local_files_only"),
            "tokenizer_json_path": str(tokenizer_json),
        },
    }
