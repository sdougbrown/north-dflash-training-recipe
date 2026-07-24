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


def _estimate_dflash_parameters(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    draft_layers: int,
    target_features: int,
) -> dict[str, int]:
    """Count local ``DFlashDraftModel`` dense trainable weights (no biases)."""
    attention_per_layer = (
        hidden_size * num_attention_heads * head_dim
        + 2 * hidden_size * num_key_value_heads * head_dim
        + num_attention_heads * head_dim * hidden_size
    )
    mlp_per_layer = 3 * hidden_size * intermediate_size
    # Each local DFlash layer has input/post RMSNorms at H and Q/K RMSNorms
    # at head_dim; the model additionally has hidden/final RMSNorms at H.
    norms = draft_layers * (2 * hidden_size + 2 * head_dim) + 2 * hidden_size
    feature_projection = target_features * hidden_size * hidden_size
    return {
        "attention_per_layer": attention_per_layer,
        "mlp_per_layer": mlp_per_layer,
        "feature_projection": feature_projection,
        "norm_weights": norms,
        "total": draft_layers * (attention_per_layer + mlp_per_layer) + feature_projection + norms,
    }


def _reviewed_draft_candidate(
    *,
    name: str,
    draft_layers: int,
    target_features: int,
    attention_layout: dict[str, Any],
    target_layers: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Make an explicit, non-selected DFlash geometry candidate."""
    estimates = _estimate_dflash_parameters(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        draft_layers=draft_layers,
        target_features=target_features,
    )
    # K and V only; assumes BF16/FP16 cache values and one query token retained.
    kv_bytes_per_token = draft_layers * 2 * num_key_value_heads * head_dim * 2
    try:
        layer_ids = build_target_layer_ids(target_layers, target_features)
        layer_id_status = "reference spread is representable for this target depth"
    except ValueError:
        # Keep config-only inspection usable for tiny fixture targets while
        # making an infeasible published geometry explicit rather than
        # silently reducing its target-feature count.
        layer_ids = None
        layer_id_status = "unavailable: target is too shallow for this target-feature count"
    return {
        "name": name,
        "draft_layers": draft_layers,
        "target_feature_count": target_features,
        "target_layer_ids_candidate": layer_ids,
        "target_layer_id_status": layer_id_status,
        "reference_hidden_state_indices_candidate": (
            [layer_id + 1 for layer_id in layer_ids] if layer_ids is not None else None
        ),
        "attention_layout": attention_layout,
        "estimated_dense_dflash_weights": estimates,
        "estimated_bf16_weight_bytes": estimates["total"] * 2,
        "estimated_kv_bytes_per_token_bf16_or_fp16": kv_bytes_per_token,
        "estimated_target_feature_bytes_per_clean_token_bf16": target_features * hidden_size * 2,
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
    hidden_size = int(target["hidden_size"])
    target_expert_intermediate_size = int(target["intermediate_size"])
    # Published Qwen3-Coder and Qwen3.6 DFlash drafts with H=2048 use a
    # dense SwiGLU width of 6144. North's 768 is a per-expert target width and
    # must not be reused as the dense draft MLP width.
    draft_intermediate_size = 3 * hidden_size
    num_attention_heads = int(target["num_attention_heads"])
    num_key_value_heads = int(target["num_key_value_heads"])
    head_dim = int(target["head_dim"])
    reviewed_draft_candidates = [
        _reviewed_draft_candidate(
            name="acceptance_first_qwen3_coder_shaped",
            draft_layers=8,
            target_features=5,
            attention_layout={"full_attention_layers": 8, "sliding_attention_layers": 0},
            target_layers=layers,
            hidden_size=hidden_size,
            intermediate_size=draft_intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        ),
        _reviewed_draft_candidate(
            name="long_context_memory_qwen36_shaped",
            draft_layers=6,
            target_features=8,
            attention_layout={
                "full_attention_layers": 1,
                "sliding_attention_layers": 5,
                "layer_types": [
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                ],
                "layer_order_source": "z-lab/Qwen3.6-35B-A3B-DFlash config.json",
                "eager_status": "blocked: local DFlash eager attention does not apply sliding_window",
            },
            target_layers=layers,
            hidden_size=hidden_size,
            intermediate_size=draft_intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        ),
    ]

    unresolved = [
        "Confirm Cohere2Moe output_hidden_states numbering before applying the reference layer_id + 1 convention.",
        "Verify every checkpoint shard before extraction; this manifest fingerprints config.json only and cannot prove teacher-weight identity.",
        "Implement exact AutoGPTQ expert-only teacher hidden-state extraction without dequantizing or changing deployment weights.",
        "Decide dense draft MLP versus any MoE-derived initializer; neither reviewed dense draft candidate is a decision.",
        "Define vLLM Cohere2Moe auxiliary-state plumbing and verify it against the running deployment integration.",
        "Choose between the reviewed 8-full/5-feature and 6-layer (5-SWA + 1-full)/8-feature candidates only after acceptance and long-context memory measurements.",
        "Do not instantiate the long-context candidate with eager attention: the local reference forwards sliding_window to eager_attention_forward, which does not enforce it.",
    ]
    if not mask_audit["valid"]:
        unresolved.append("Resolve the invalid <MASK_TOKEN> tokenizer audit before training.")

    return {
        "status": "CPU layout, optional training-step, and reference eager-DFlash adapter tested; North integration missing; not trainable",
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
        "draft_common": {
            "hidden_size": hidden_size,
            "intermediate_size": draft_intermediate_size,
            "target_expert_intermediate_size": target_expert_intermediate_size,
            "intermediate_size_source": "published dense Qwen3 DFlash geometry: 3 * hidden_size",
            "head_dim": head_dim,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "hidden_act": target.get("hidden_act"),
            "rms_norm_eps": target.get("rms_norm_eps", target.get("layer_norm_eps")),
            "block_size_candidate": 16,
            "target_layer_id_convention": "zero_based_transformer_block_index",
            "north_hidden_state_indexing_status": "unverified; Cohere2Moe runtime/extractor evidence required",
            "vocab_size": vocab_size,
            "max_position_embeddings": target.get("max_position_embeddings"),
            "shared_embedding_and_lm_head": "paper-aligned candidate; exact weight handoff unresolved",
            "mask_token_id": mask_audit["id"],
            "mask_token_audit": mask_audit,
        },
        "reviewed_draft_candidates": reviewed_draft_candidates,
        "draft_candidate_selection": "none; both candidates require review and North-specific gates",
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
