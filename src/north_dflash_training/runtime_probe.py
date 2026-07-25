"""Guarded, random-only Hugging Face artifact for the North DFlash runtime probe.

This module is deliberately not a trainer and does not read target weights.  It
writes a tiny *North-shaped* random DFlash drafter only after checking the
locally inspected vLLM loader contract.  The artifact is for aux-state/runtime
loading plumbing, never token acceptance, quality, or training.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

NORTH_HIDDEN_SIZE = 2048
NORTH_INTERMEDIATE_SIZE = 6144
NORTH_NUM_HEADS = 32
NORTH_NUM_KV_HEADS = 4
NORTH_HEAD_DIM = 128
NORTH_VOCAB_SIZE = 262144
NORTH_MASK_TOKEN_ID = 1
NORTH_TARGET_LAYERS = 49
NORTH_MAX_POSITIONS = 500000
NORTH_ROPE_THETA = 50000
NORTH_RMS_NORM_EPS = 1e-6
DEFAULT_SEED = 20260724
DEFAULT_VLLM_SOURCE = Path("/home/douglasbrown/Code/_worktrees/vllm-north-dflash")
DEFAULT_IDENTITY_MANIFEST = Path("configs/north-int4-teacher-checkpoint-identity.json")


@dataclass(frozen=True)
class ProbeDimensions:
    """The dimensions that must agree with North at the runtime boundary."""

    hidden_size: int = NORTH_HIDDEN_SIZE
    intermediate_size: int = NORTH_INTERMEDIATE_SIZE
    num_attention_heads: int = NORTH_NUM_HEADS
    num_key_value_heads: int = NORTH_NUM_KV_HEADS
    head_dim: int = NORTH_HEAD_DIM
    vocab_size: int = NORTH_VOCAB_SIZE
    num_target_layers: int = NORTH_TARGET_LAYERS
    max_position_embeddings: int = NORTH_MAX_POSITIONS
    rope_theta: int = NORTH_ROPE_THETA
    rms_norm_eps: float = NORTH_RMS_NORM_EPS


@dataclass(frozen=True)
class ProbeGeometry:
    name: str
    draft_layers: int
    target_layer_ids: tuple[int, ...]
    block_size: int


SMOKE_GEOMETRY = ProbeGeometry("smoke", 1, (24,), 2)
FULL_GEOMETRY = ProbeGeometry("full", 8, (1, 12, 24, 35, 46), 16)
GEOMETRIES = {item.name: item for item in (SMOKE_GEOMETRY, FULL_GEOMETRY)}


def probe_geometry(name: str) -> ProbeGeometry:
    try:
        return GEOMETRIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown probe geometry {name!r}; choose one of {sorted(GEOMETRIES)}") from exc


def _validate_geometry(geometry: ProbeGeometry, dimensions: ProbeDimensions) -> None:
    if geometry.draft_layers < 1 or geometry.block_size < 2:
        raise ValueError("draft layers must be positive and block_size must be at least two")
    ids = geometry.target_layer_ids
    if not ids or tuple(sorted(ids)) != ids or len(set(ids)) != len(ids):
        raise ValueError("target_layer_ids must be non-empty, unique, and ascending")
    if any(not isinstance(layer, int) or layer < 0 or layer >= dimensions.num_target_layers for layer in ids):
        raise ValueError("target_layer_ids must be zero-based target block indices within num_target_layers")
    # North intentionally uses a 2048-wide residual stream with 32 x 128
    # attention projections; do not infer head_dim from hidden_size.
    if min(dimensions.hidden_size, dimensions.num_attention_heads, dimensions.head_dim) < 1:
        raise ValueError("hidden_size, num_attention_heads, and head_dim must be positive")
    if dimensions.num_attention_heads % dimensions.num_key_value_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")


def build_probe_config(
    geometry: str | ProbeGeometry = "smoke", *, dimensions: ProbeDimensions = ProbeDimensions()
) -> dict[str, Any]:
    """Build a DFlashDraftModel config without importing a model package.

    ``dflash_config.target_layer_ids`` is the local DFlash convention: zero-based
    target transformer-block indices.  vLLM's extractor deliberately consumes
    the corresponding one-based hidden-state entries in
    ``eagle_aux_hidden_state_layer_ids``; it derives those as ``target_id + 1``.
    Keeping both fields makes the conversion explicit rather than guessed.
    """
    selected = geometry if isinstance(geometry, ProbeGeometry) else probe_geometry(geometry)
    _validate_geometry(selected, dimensions)
    target_ids = list(selected.target_layer_ids)
    return {
        "_name_or_path": "RANDOM-NON-PRODUCTION-NORTH-DFLASH-RUNTIME-PROBE",
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "dtype": "bfloat16",
        "vocab_size": dimensions.vocab_size,
        "draft_vocab_size": dimensions.vocab_size,
        "target_hidden_size": dimensions.hidden_size,
        "num_target_layers": dimensions.num_target_layers,
        "hidden_size": dimensions.hidden_size,
        "intermediate_size": dimensions.intermediate_size,
        "num_hidden_layers": selected.draft_layers,
        "num_attention_heads": dimensions.num_attention_heads,
        "num_key_value_heads": dimensions.num_key_value_heads,
        "head_dim": dimensions.head_dim,
        "hidden_act": "silu",
        "rms_norm_eps": dimensions.rms_norm_eps,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "max_position_embeddings": dimensions.max_position_embeddings,
        "rope_theta": dimensions.rope_theta,
        "rope_parameters": {"rope_type": "default", "rope_theta": dimensions.rope_theta},
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "use_cache": True,
        "block_size": selected.block_size,
        "layer_types": ["full_attention"] * selected.draft_layers,
        "dflash_config": {
            "mask_token_id": NORTH_MASK_TOKEN_ID,
            "target_layer_ids": target_ids,
            "target_layer_id_convention": "zero_based_transformer_block_index",
            "sample_from_anchor": False,
            "causal": False,
            "use_aux_hidden_state": True,
        },
        # vLLM extracts hidden_states[layer_id] where hidden-state entry zero
        # is the embeddings output. This must remain target_id + 1.
        "eagle_aux_hidden_state_layer_ids": [layer + 1 for layer in target_ids],
        "runtime_probe": {
            "purpose": "aux_state_plumbing_and_runtime_loading_only",
            "random": True,
            "non_production": True,
            "not_for_acceptance": True,
            "not_for_training": True,
            "geometry": selected.name,
            "block_size_relation": "one clean anchor plus block_size - 1 masked speculative slots",
        },
    }


def estimate_weight_bytes(config: dict[str, Any], *, dtype_bytes: int = 2) -> int:
    """Estimate serialized DFlashDraftModel tensors, excluding shared target I/O."""
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    layers = int(config["num_hidden_layers"])
    features = len(config["dflash_config"]["target_layer_ids"])
    # Q, K, V, O; SwiGLU gate/up/down; two H norms plus Q/K head norms;
    # final + context hidden norms; and the aux feature projection. No embed/lm head.
    per_layer = hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim + heads * head_dim * hidden
    per_layer += 3 * hidden * intermediate + 2 * hidden + 2 * head_dim
    total = layers * per_layer + 2 * hidden + features * hidden * hidden
    return total * dtype_bytes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_link(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload.get("manifest_sha256"), str) or not isinstance(payload.get("config"), dict):
        raise ValueError("source checkpoint identity manifest lacks manifest_sha256/config")
    return {
        "identity_manifest_path": str(path),
        "identity_manifest_sha256": payload["manifest_sha256"],
        "target_config_sha256": payload["config"].get("sha256"),
        "target_config_size_bytes": payload["config"].get("size_bytes"),
        "target_weights_read": False,
    }


def verify_vllm_loader_contract(vllm_source: str | Path) -> dict[str, str]:
    """Fail closed unless the inspected vLLM tree has the required DFlash contract."""
    root = Path(vllm_source)
    files = {
        "registry": root / "vllm/model_executor/models/registry.py",
        "loader": root / "vllm/model_executor/models/qwen3_dflash.py",
        "sharing": root / "vllm/v1/worker/gpu/spec_decode/dflash/utils.py",
        "extractor": root / "vllm/v1/worker/gpu_model_runner.py",
    }
    try:
        text = {name: path.read_text() for name, path in files.items()}
    except OSError as exc:
        raise RuntimeError(f"cannot prove vLLM loader contract: {exc}") from exc
    required = {
        "registry": ('"DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM")',),
        "loader": (
            'name = "model." + name',
            'skip_substrs.append("embed_tokens")',
            'self.model._build_fused_kv_buffers()',
            'orig_to_new_stacked=',
        ),
        "sharing": ('_should_share(', '"has_own_embed_tokens"', '"has_own_lm_head"'),
        "extractor": ('i + 1 for i in (dflash_config.get("target_layer_ids") or [])',),
    }
    missing = [f"{name}:{needle}" for name, needles in required.items() for needle in needles if needle not in text[name]]
    if missing:
        raise RuntimeError("cannot prove vLLM loader contract; required evidence missing: " + ", ".join(missing))
    revision = "unavailable"
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"vllm_source": str(root), "vllm_revision": revision}


def _reference_model_class():
    try:
        from dflash.model import DFlashDraftModel
        from transformers import Qwen3Config
    except ImportError as exc:  # pragma: no cover - exercised only without optional refs.
        raise RuntimeError("generation requires the existing reference environment with torch, transformers, safetensors, and dflash") from exc
    return DFlashDraftModel, Qwen3Config


def _assert_reference_state_is_vllm_loadable(model) -> list[str]:
    """Check save_pretrained source names against the inspected vLLM mapper."""
    keys = sorted(model.state_dict())
    prohibited = [key for key in keys if "embed_tokens" in key or "lm_head" in key]
    if prohibited:
        raise RuntimeError("reference DFlash state unexpectedly duplicates target I/O: " + ", ".join(prohibited))
    allowed = ("fc.", "hidden_norm.", "norm.", "layers.")
    unexpected = [key for key in keys if not key.startswith(allowed)]
    if unexpected:
        raise RuntimeError("reference DFlash state has unproved vLLM names: " + ", ".join(unexpected))
    required = {"fc.weight", "hidden_norm.weight", "norm.weight"}
    if not required.issubset(keys) or not any("self_attn.q_proj.weight" in key for key in keys):
        raise RuntimeError("reference DFlash state is missing required DFlash tensors")
    return keys


def _free_space(path: Path) -> int:
    return shutil.disk_usage(path).free


def generate_runtime_probe(
    output: str | Path,
    *,
    geometry: str = "smoke",
    seed: int = DEFAULT_SEED,
    confirm_full_random_nonproduction: bool = False,
    vllm_source: str | Path = DEFAULT_VLLM_SOURCE,
    identity_manifest: str | Path = DEFAULT_IDENTITY_MANIFEST,
) -> dict[str, Any]:
    """Create a random BF16 checkpoint, refusing ambiguous or unsafe requests."""
    selected = probe_geometry(geometry)
    if selected.name == "full" and not confirm_full_random_nonproduction:
        raise ValueError("full geometry is blocked; pass --confirm-full-random-nonproduction explicitly")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    if destination.parent.exists() and not destination.parent.is_dir():
        raise NotADirectoryError(destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    config = build_probe_config(selected.name)
    estimated_bytes = estimate_weight_bytes(config)
    # Keep a conservative 20% serialization/filesystem margin; this does not
    # claim to cover RAM required by a full in-memory model construction.
    required_free_bytes = int(estimated_bytes * 1.20) + 4 * 1024 * 1024
    free_bytes = _free_space(destination.parent)
    if free_bytes < required_free_bytes:
        raise OSError(f"insufficient free space: need {required_free_bytes} bytes, have {free_bytes}")
    loader_evidence = verify_vllm_loader_contract(vllm_source)
    identity = _identity_link(Path(identity_manifest))
    DFlashDraftModel, Qwen3Config = _reference_model_class()
    import torch
    from safetensors import safe_open

    torch.manual_seed(seed)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = DFlashDraftModel(Qwen3Config(**config))
    finally:
        torch.set_default_dtype(previous_dtype)
    state_keys = _assert_reference_state_is_vllm_loadable(model)

    with tempfile.TemporaryDirectory(prefix=".north-dflash-runtime-probe-", dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        model.save_pretrained(staged, safe_serialization=True)
        weights_path = staged / "model.safetensors"
        if not weights_path.is_file():
            raise RuntimeError("save_pretrained did not produce model.safetensors; refusing unproved output")
        with safe_open(weights_path, framework="pt", device="cpu") as saved:
            saved_keys = sorted(saved.keys())
        if saved_keys != state_keys:
            raise RuntimeError("save_pretrained tensor names differ from the verified reference state")
        manifest = {
            "artifact_kind": "RANDOM_NON_PRODUCTION_NORTH_DFLASH_RUNTIME_PROBE",
            "purpose": "prove_vllm_target_aux_state_plumbing_and_runtime_loading_only",
            "prohibited_uses": ["acceptance", "quality_measurement", "training", "deployment"],
            "random_seed": seed,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "geometry": selected.name,
            "config": config,
            "target_layer_id_convention": {
                "dflash_config.target_layer_ids": "ascending zero-based target transformer-block indices",
                "eagle_aux_hidden_state_layer_ids": "target_layer_id + 1 because vLLM hidden_states[0] is the embeddings output",
            },
            "serialized_weights": {
                "filename": weights_path.name,
                "sha256": _sha256(weights_path),
                "size_bytes": weights_path.stat().st_size,
                "tensor_names": state_keys,
                "contains_target_embedding": False,
                "contains_target_lm_head": False,
            },
            "storage_preflight": {
                "estimated_weight_bytes": estimated_bytes,
                "required_free_bytes": required_free_bytes,
                "free_bytes_before_write": free_bytes,
                "dtype": "bfloat16",
            },
            "source_checkpoint_identity": identity,
            "vllm_loader_evidence": loader_evidence,
        }
        (staged / "runtime-probe-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        # Atomic publish preserves the no-overwrite policy for normal local use.
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {destination}")
        os.replace(staged, destination)
    return manifest
