"""Provisional exact-target teacher identity and feature-manifest primitives.

The CPU scaffold fingerprints the reviewed config and records the quantization
contract, but it deliberately does not hash checkpoint shards or extract hidden
states.  Consequently its manifests are *not* sufficient to prove that an
extractor loaded the exact teacher weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


MANIFEST_SCHEMA_VERSION = 2
LAYER_ID_CONVENTION = "zero_based_transformer_block_index"
CHECKPOINT_IDENTITY_STATUS = "not_verified"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class AutoGPTQIdentity:
    """Quantization fields that must remain unchanged at extraction time."""

    model_type: str | None
    architecture: str | None
    bits: int | None
    group_size: int | None
    data_type: str | None
    quant_method: str | None
    provider: str | None
    autoround_version: str | None
    num_experts: int | None
    num_experts_per_tok: int | None
    requires_exact_expert_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "architecture": self.architecture,
            "bits": self.bits,
            "group_size": self.group_size,
            "data_type": self.data_type,
            "quant_method": self.quant_method,
            "provider": self.provider,
            "autoround_version": self.autoround_version,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "requires_exact_expert_only": self.requires_exact_expert_only,
        }

    def is_complete(self) -> bool:
        return self.requires_exact_expert_only and all(
            value is not None
            for value in (
                self.model_type,
                self.architecture,
                self.bits,
                self.group_size,
                self.data_type,
                self.quant_method,
                self.provider,
                self.autoround_version,
                self.num_experts,
                self.num_experts_per_tok,
            )
        )


@dataclass(frozen=True)
class TeacherFeatureManifest:
    """A config-fingerprinted but checkpoint-unverified extractor contract.

    ``selected_layer_ids`` are zero-based transformer block indices, matching
    ``build_target_layer_ids``.  The reference Qwen implementation indexes an
    HF ``hidden_states`` list at ``layer_id + 1`` because index zero is the
    embedding output; this has not been established for Cohere2Moe and is not
    encoded as a North fact here.
    """

    teacher_config_path: str
    teacher_config_sha256: str
    target_num_hidden_layers: int
    selected_layer_ids: tuple[int, ...]
    quantization_identity: AutoGPTQIdentity
    selected_layer_id_convention: str = LAYER_ID_CONVENTION
    checkpoint_identity_status: str = CHECKPOINT_IDENTITY_STATUS
    feature_kind: str = "hidden_state"
    extraction_status: str = "not_implemented"
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema_version: {self.schema_version}")
        if len(self.teacher_config_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.teacher_config_sha256
        ):
            raise ValueError("teacher_config_sha256 must be a lowercase SHA-256 digest")
        if isinstance(self.target_num_hidden_layers, bool) or self.target_num_hidden_layers < 1:
            raise ValueError("target_num_hidden_layers must be positive")
        if not self.selected_layer_ids:
            raise ValueError("selected_layer_ids must not be empty")
        if len(set(self.selected_layer_ids)) != len(self.selected_layer_ids):
            raise ValueError("selected_layer_ids must be distinct")
        if any(
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer < 0
            or layer >= self.target_num_hidden_layers
            for layer in self.selected_layer_ids
        ):
            raise ValueError("selected_layer_ids must be in the target's zero-based layer range")
        if self.selected_layer_id_convention != LAYER_ID_CONVENTION:
            raise ValueError("unexpected selected-layer ID convention")
        if self.checkpoint_identity_status != CHECKPOINT_IDENTITY_STATUS:
            raise ValueError("this scaffold cannot claim checkpoint identity verification")
        if not self.quantization_identity.is_complete():
            raise ValueError("quantization identity is incomplete; refusing exact-teacher claim")
        if self.extraction_status != "not_implemented":
            raise ValueError("this CPU-only slice does not implement feature extraction")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "teacher_config_path": self.teacher_config_path,
            "teacher_config_sha256": self.teacher_config_sha256,
            "target_num_hidden_layers": self.target_num_hidden_layers,
            "selected_layer_ids": list(self.selected_layer_ids),
            "selected_layer_id_convention": self.selected_layer_id_convention,
            "quantization_identity": self.quantization_identity.to_dict(),
            "checkpoint_identity_status": self.checkpoint_identity_status,
            "feature_kind": self.feature_kind,
            "extraction_status": self.extraction_status,
        }


def teacher_feature_manifest_from_config(
    target_config_path: str | Path,
    selected_layer_ids: tuple[int, ...] | list[int],
) -> TeacherFeatureManifest:
    """Build a config-fingerprinted, checkpoint-unverified manifest from JSON."""
    path = Path(target_config_path)
    config_bytes = path.read_bytes()
    target = json.loads(config_bytes)
    quant = target.get("quantization_config", {})
    identity = AutoGPTQIdentity(
        model_type=target.get("model_type"),
        architecture=(target.get("architectures") or [None])[0],
        bits=quant.get("bits"),
        group_size=quant.get("group_size"),
        data_type=quant.get("data_type"),
        quant_method=quant.get("quant_method"),
        provider=quant.get("provider"),
        autoround_version=quant.get("autoround_version"),
        num_experts=target.get("num_experts"),
        num_experts_per_tok=target.get("num_experts_per_tok"),
    )
    return TeacherFeatureManifest(
        teacher_config_path=str(path),
        teacher_config_sha256=_sha256_bytes(config_bytes),
        target_num_hidden_layers=target["num_hidden_layers"],
        selected_layer_ids=tuple(selected_layer_ids),
        quantization_identity=identity,
    )


def validate_teacher_feature_manifest(value: Mapping[str, Any]) -> None:
    """Validate config-level identity and reject an unproved exact-weight claim."""
    required = {
        "schema_version",
        "teacher_config_path",
        "teacher_config_sha256",
        "target_num_hidden_layers",
        "selected_layer_ids",
        "selected_layer_id_convention",
        "quantization_identity",
        "checkpoint_identity_status",
        "feature_kind",
        "extraction_status",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"missing manifest fields: {sorted(missing)}")
    identity_data = value["quantization_identity"]
    if not isinstance(identity_data, Mapping):
        raise ValueError("quantization_identity must be an object")
    try:
        identity = AutoGPTQIdentity(**dict(identity_data))
        TeacherFeatureManifest(
            teacher_config_path=value["teacher_config_path"],
            teacher_config_sha256=value["teacher_config_sha256"],
            target_num_hidden_layers=value["target_num_hidden_layers"],
            selected_layer_ids=tuple(value["selected_layer_ids"]),
            selected_layer_id_convention=value["selected_layer_id_convention"],
            quantization_identity=identity,
            checkpoint_identity_status=value["checkpoint_identity_status"],
            feature_kind=value["feature_kind"],
            extraction_status=value["extraction_status"],
            schema_version=value["schema_version"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid teacher feature manifest: {exc}") from exc
