"""Derive a reviewable North candidate without loading model weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_TARGET_CONFIG = Path("/home/douglasbrown/Models/North-Mini-Code-1.0-int4-autoround-gptq-g32/config.json")
DEFAULT_TOKENIZER_CONFIG = Path("/home/douglasbrown/Models/North-Mini-Code-1.0-int4-autoround-gptq-g32/tokenizer_config.json")


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int = 5) -> list[int]:
    """Match the reference dflash/model.py layer-spread formula."""
    if num_target_layers < 4 or num_draft_layers < 1:
        raise ValueError("target must have at least 4 layers and draft at least 1")
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    return [int(round(start + i * (end - start) / (num_draft_layers - 1))) for i in range(num_draft_layers)]


def derive_north_candidate(
    target_config_path: str | Path = DEFAULT_TARGET_CONFIG,
    tokenizer_config_path: str | Path = DEFAULT_TOKENIZER_CONFIG,
) -> dict[str, Any]:
    """Read only JSON config and return a candidate with unresolved choices explicit."""
    target_path = Path(target_config_path)
    tokenizer_path = Path(tokenizer_config_path)
    target = json.loads(target_path.read_text())
    tokenizer = json.loads(tokenizer_path.read_text())
    layers = int(target["num_hidden_layers"])
    draft_layers = 5
    quant = target.get("quantization_config", {})

    return {
        "status": "scaffold-only / not trainable",
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
            "target_layer_ids_candidate": build_target_layer_ids(layers, draft_layers),
            "vocab_size": target.get("vocab_size"),
            "max_position_embeddings": target.get("max_position_embeddings"),
            "shared_embedding_and_lm_head": "paper-aligned candidate; exact weight handoff unresolved",
            "mask_token_id": None,
        },
        "unresolved_choices": [
            "Confirm whether layer IDs are zero-based module indices or hidden-state indices for Cohere2Moe outputs.",
            "Implement exact AutoGPTQ expert-only teacher hidden-state extraction without dequantizing or changing deployment weights.",
            "Choose a tokenizer-approved mask token; no mask token is asserted by this scaffold.",
            "Decide dense draft MLP versus any MoE-derived initializer; a five-layer dense draft is only a paper-shaped candidate.",
            "Define sparse block-attention mask and feature alignment for concatenated sampled blocks.",
            "Define vLLM Cohere2Moe auxiliary-state plumbing and verify it against the running deployment integration.",
        ],
        "observed_tokenizer": {
            "tokenizer_class": tokenizer.get("tokenizer_class"),
            "eos_token": tokenizer.get("eos_token"),
            "pad_token": tokenizer.get("pad_token"),
            "local_files_only": tokenizer.get("local_files_only"),
        },
    }
