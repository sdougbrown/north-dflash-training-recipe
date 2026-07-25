"""Immutable draft-only checkpoints for retained DFlash pilots.

This is deliberately a narrow, CPU-testable checkpoint boundary: it saves draft
adapter tensors, a request ledger, a response-token ledger, and a trusted-local
optimizer state.  It never saves teacher features, target weights, or shared
vocabulary tensors.

The response ledger (``response-ledger.json``) records the exact
:class:`ResponseExample` values processed during the retained interval and is
written as canonical JSON alongside the request ledger.

``optimizer.pt`` is written with :func:`torch.save` and loaded via
``weights_only=True`` from a retained private byte buffer; it is therefore safe
only from a trusted-local checkpoint path.  This module is not a production
attention implementation or a claim that any draft adapter has production-ready
attention.
"""

from __future__ import annotations

import ctypes
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping

try:
    import torch
    from safetensors.torch import load as safe_load
    from safetensors.torch import save_file as safe_save_file
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Checkpoint save/resume requires PyTorch and safetensors; install "
        "north-dflash-training-scaffold[runtime]."
    ) from exc

from .feature_stream import BoundedFeatureRing, TeacherRuntimeIdentity
from .online_step import BoundedOptimizerStepResult
from .schema import ResponseExample
from .training import DFlashTrainingStep

CHECKPOINT_SCHEMA_VERSION = 2
MANIFEST_FILENAME = "checkpoint-manifest.json"
MANIFEST_SHA256_FILENAME = "checkpoint-manifest.sha256"
DRAFT_WEIGHTS_FILENAME = "draft-model.safetensors"
OPTIMIZER_FILENAME = "optimizer.pt"
REQUEST_LEDGER_FILENAME = "request-ledger.json"
RESPONSE_LEDGER_FILENAME = "response-ledger.json"
_DATA_FILENAMES = (
    DRAFT_WEIGHTS_FILENAME,
    OPTIMIZER_FILENAME,
    REQUEST_LEDGER_FILENAME,
    RESPONSE_LEDGER_FILENAME,
)
_ALLOWED_FILENAMES = frozenset(
    (_DATA_FILENAMES + (MANIFEST_FILENAME, MANIFEST_SHA256_FILENAME))
)
_SHA256_LENGTH = 64
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

# Allowlisted state-dict key prefixes for retained adapters.
_ALLOWLISTED_ADAPTER_PREFIXES: dict[str, tuple[str, ...]] = {
    "north_dflash_training.transformers_draft_adapter.TransformersDFlashDraftAdapter": (
        "draft_model.",
    ),
    "north_dflash_training.training.SyntheticDraftAdapter": (
        "target_projection.",
        "noise_projection.",
        "position_projection.",
        "output_projection.",
    ),
}

# Fields present in each request-ledger entry.
_LEDGER_FIELDS = frozenset(
    {
        "request_id",
        "source_token_count",
        "context_length",
        "query_count",
        "active_label_count",
        "loss",
        "gradient_norm",
        "gradients_with_nonzero_values",
        "updated_parameter_tensors",
        "feature_bytes_released",
        "connector_sha256",
        "connector_file_bytes_released",
        "lock_file_released",
    }
)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _require_sha256(value: object, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not deterministic JSON: {exc}") from exc


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash contiguous raw bytes through a uint8 view (including BF16 tensors)."""
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("expected tensor")
    raw = tensor.detach().to(device="cpu").contiguous().view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"checkpoint file missing: {path.name}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"checkpoint entry must be a regular non-symlink file: {path.name}")


# ---------------------------------------------------------------------------
# Response-ledger helpers
# ---------------------------------------------------------------------------


def _response_examples_canonical_bytes(
    examples: Sequence[ResponseExample | Mapping[str, Any]],
) -> bytes:
    """Produce deterministic canonical JSON from a list of response examples.

    Each entry is serialised via :meth:`ResponseExample.to_dict` and sorted by
    key to ensure a repeatable SHA-256 regardless of input type.
    """
    entries: list[dict[str, Any]] = []
    for example in examples:
        if isinstance(example, ResponseExample):
            entries.append(example.to_dict())
        elif isinstance(example, MappingABC):
            entries.append(ResponseExample.from_mapping(example).to_dict())
        else:
            raise TypeError("response ledger entries must be ResponseExample or mapping")
    return _canonical_json(entries)


def _response_ledger_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_response_examples_against_requests(
    examples: Sequence[ResponseExample | Mapping[str, Any]],
    request_ledger: "SavedRequestLedger",
) -> None:
    if len(examples) != len(request_ledger.entries):
        raise ValueError("response and request ledger entry counts must match")
    for index, (raw_example, request) in enumerate(
        zip(examples, request_ledger.entries, strict=True)
    ):
        example = (
            raw_example
            if isinstance(raw_example, ResponseExample)
            else ResponseExample.from_mapping(raw_example)
        )
        metadata = example.metadata
        if not isinstance(metadata, MappingABC) or metadata.get("request_id") != request["request_id"]:
            raise ValueError(f"response ledger request_id mismatch at entry {index}")
        if len(example.prompt_tokens) + len(example.response_tokens) != request["source_token_count"]:
            raise ValueError(f"response ledger token count mismatch at entry {index}")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SavedRequestLedger:
    """Validated immutable records for completed optimizer requests.

    Each entry mirrors :class:`BoundedOptimizerStepResult` plus connector-
    release fields so the ledger alone proves the connector was released.
    """

    entries: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        request_ids: set[str] = set()
        for index, entry in enumerate(self.entries):
            if not isinstance(entry, MappingABC) or set(entry) != _LEDGER_FIELDS:
                raise ValueError(
                    f"ledger entry {index} must contain exactly {sorted(_LEDGER_FIELDS)}"
                )
            request_id = entry["request_id"]
            if not isinstance(request_id, str) or not request_id.strip() or request_id in request_ids:
                raise ValueError("ledger request_id values must be nonempty and unique")
            request_ids.add(request_id)
            for field_name in _LEDGER_FIELDS - {"request_id", "loss", "gradient_norm", "connector_sha256", "lock_file_released"}:
                value = entry[field_name]
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"ledger {field_name} must be a positive integer")
            for field_name in ("loss", "gradient_norm"):
                value = entry[field_name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"ledger {field_name} must be finite")
            sha256 = entry["connector_sha256"]
            if not _is_sha256(sha256):
                raise ValueError("ledger connector_sha256 must be a lowercase SHA-256 hex digest")
            if not isinstance(entry["lock_file_released"], bool) or not entry["lock_file_released"]:
                raise ValueError("ledger lock_file_released must be true")

    def to_list(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self.entries]

    @classmethod
    def from_list(cls, entries: Sequence[Mapping[str, Any]]) -> "SavedRequestLedger":
        if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
            raise ValueError("request ledger must be a JSON array")
        return cls(entries=tuple(dict(entry) for entry in entries))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_list())).hexdigest()


@dataclass(frozen=True)
class TiedVocabIdentity:
    """Metadata-only identity of the shared input vocabulary tensor."""

    sha256: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        _require_sha256(self.sha256, "tied vocabulary sha256")
        if not self.shape or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in self.shape
        ):
            raise ValueError("tied vocabulary shape must contain positive integers")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("tied vocabulary dtype must be nonempty")

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TiedVocabIdentity":
        return cls(
            sha256=_tensor_sha256(tensor),
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).removeprefix("torch."),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "shape": list(self.shape), "dtype": self.dtype}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TiedVocabIdentity":
        if not isinstance(value, MappingABC) or set(value) != {"sha256", "shape", "dtype"}:
            raise ValueError("tied_vocab_identity has invalid fields")
        shape = value["shape"]
        if not isinstance(shape, list):
            raise ValueError("tied vocabulary shape must be a list")
        return cls(sha256=value["sha256"], shape=tuple(shape), dtype=value["dtype"])


@dataclass(frozen=True)
class CheckpointManifest:
    """Canonical metadata protected by an externally supplied root digest."""

    schema_version: int
    step_count: int
    target_name: str
    runtime_image_id: str
    backend: str
    checkpoint_manifest_sha256: str
    selected_layer_ids: tuple[int, ...]
    hidden_size: int
    request_ledger_sha256: str
    request_ledger_entry_count: int
    response_ledger_sha256: str
    response_ledger_entry_count: int
    tied_vocab_identity: TiedVocabIdentity
    draft_architecture: dict[str, Any] | None
    optimizer_type: str
    optimizer_param_groups: list[list[dict[str, Any]]]
    files: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint schema version: {self.schema_version}")
        _require_nonnegative_int(self.step_count, "step_count")
        _require_nonnegative_int(self.request_ledger_entry_count, "request_ledger_entry_count")
        _require_nonnegative_int(self.response_ledger_entry_count, "response_ledger_entry_count")
        if self.request_ledger_entry_count != self.step_count:
            raise ValueError("request ledger entry count must equal step_count")
        if self.response_ledger_entry_count != self.step_count:
            raise ValueError("response ledger entry count must equal step_count")
        if not isinstance(self.target_name, str) or not self.target_name or not isinstance(self.backend, str) or not self.backend:
            raise ValueError("manifest target_name and backend must be nonempty")
        if not isinstance(self.runtime_image_id, str) or not self.runtime_image_id.startswith("sha256:"):
            raise ValueError("manifest runtime_image_id must be a pinned digest")
        _require_sha256(self.runtime_image_id.removeprefix("sha256:"), "runtime_image_id")
        for field_name in (
            "checkpoint_manifest_sha256",
            "request_ledger_sha256",
            "response_ledger_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if not self.selected_layer_ids or tuple(sorted(set(self.selected_layer_ids))) != self.selected_layer_ids:
            raise ValueError("selected_layer_ids must be nonempty, unique, and ascending")
        if any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in self.selected_layer_ids):
            raise ValueError("selected_layer_ids must be non-negative integers")
        if isinstance(self.hidden_size, bool) or not isinstance(self.hidden_size, int) or self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if not isinstance(self.tied_vocab_identity, TiedVocabIdentity):
            raise ValueError("manifest must include tied_vocab_identity")
        if not isinstance(self.optimizer_type, str) or not self.optimizer_type:
            raise ValueError("optimizer_type must be nonempty")
        # draft_architecture must be an object with at least adapter_type; config may be null.
        if self.draft_architecture is not None:
            if not isinstance(self.draft_architecture, MappingABC):
                raise ValueError("draft_architecture must be an object or null")
            if set(self.draft_architecture) - {"adapter_type", "config"}:
                raise ValueError("draft_architecture has invalid fields")
            if not isinstance(self.draft_architecture.get("adapter_type"), str) or not self.draft_architecture["adapter_type"]:
                raise ValueError("draft architecture adapter_type must be nonempty")
            config_val = self.draft_architecture.get("config")
            if config_val is not None and not isinstance(config_val, MappingABC):
                raise ValueError("draft architecture config must be a mapping or null")
            _canonical_json(self.draft_architecture)
        else:
            raise ValueError("draft_architecture must not be None; use adapter_type only when there is no config")
        if not isinstance(self.optimizer_param_groups, list):
            raise ValueError("optimizer_param_groups must be a list")
        for group_idx, group in enumerate(self.optimizer_param_groups):
            if not isinstance(group, list):
                raise ValueError(f"optimizer_param_groups[{group_idx}] must be a list")
            for param_idx, param_info in enumerate(group):
                if not isinstance(param_info, MappingABC) or set(param_info) != {"name", "shape", "dtype"}:
                    raise ValueError(
                        f"optimizer_param_groups[{group_idx}][{param_idx}] has invalid fields"
                    )
                if not isinstance(param_info["name"], str) or not param_info["name"]:
                    raise ValueError("optimizer parameter name must be nonempty")
                if not isinstance(param_info["shape"], list):
                    raise ValueError("optimizer parameter shape must be a list")
                if not isinstance(param_info["dtype"], str) or not param_info["dtype"]:
                    raise ValueError("optimizer parameter dtype must be nonempty")
        _validate_file_records(self.files)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "step_count": self.step_count,
            "target_name": self.target_name,
            "runtime_image_id": self.runtime_image_id,
            "backend": self.backend,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "selected_layer_ids": list(self.selected_layer_ids),
            "hidden_size": self.hidden_size,
            "request_ledger_sha256": self.request_ledger_sha256,
            "request_ledger_entry_count": self.request_ledger_entry_count,
            "response_ledger_sha256": self.response_ledger_sha256,
            "response_ledger_entry_count": self.response_ledger_entry_count,
            "tied_vocab_identity": self.tied_vocab_identity.to_dict(),
            "draft_architecture": self.draft_architecture,
            "optimizer_type": self.optimizer_type,
            "optimizer_param_groups": self.optimizer_param_groups,
            "files": [dict(record) for record in self.files],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointManifest":
        expected = {
            "schema_version", "step_count", "target_name", "runtime_image_id", "backend",
            "checkpoint_manifest_sha256", "selected_layer_ids", "hidden_size",
            "request_ledger_sha256", "request_ledger_entry_count",
            "response_ledger_sha256", "response_ledger_entry_count",
            "tied_vocab_identity", "draft_architecture",
            "optimizer_type", "optimizer_param_groups", "files",
        }
        if not isinstance(value, MappingABC) or set(value) != expected:
            raise ValueError(f"checkpoint manifest has invalid fields: {sorted(set(value) ^ expected)}")
        try:
            selected = value["selected_layer_ids"]
            files = value["files"]
            if not isinstance(selected, list) or not isinstance(files, list):
                raise ValueError("manifest selected_layer_ids and files must be lists")
            architecture = value["draft_architecture"]
            if architecture is not None and not isinstance(architecture, MappingABC):
                raise ValueError("manifest draft_architecture must be an object or null")
            return cls(
                schema_version=value["schema_version"],
                step_count=value["step_count"],
                target_name=value["target_name"],
                runtime_image_id=value["runtime_image_id"],
                backend=value["backend"],
                checkpoint_manifest_sha256=value["checkpoint_manifest_sha256"],
                selected_layer_ids=tuple(selected),
                hidden_size=value["hidden_size"],
                request_ledger_sha256=value["request_ledger_sha256"],
                request_ledger_entry_count=value["request_ledger_entry_count"],
                response_ledger_sha256=value["response_ledger_sha256"],
                response_ledger_entry_count=value["response_ledger_entry_count"],
                tied_vocab_identity=TiedVocabIdentity.from_dict(value["tied_vocab_identity"]),
                draft_architecture=dict(architecture) if architecture is not None else None,
                optimizer_type=value["optimizer_type"],
                optimizer_param_groups=value["optimizer_param_groups"],
                files=tuple(dict(record) for record in files),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid checkpoint manifest: {exc}") from exc


def _validate_file_records(records: tuple[dict[str, Any], ...]) -> None:
    if len(records) != len(_DATA_FILENAMES):
        raise ValueError(f"manifest must contain exactly {len(_DATA_FILENAMES)} retained data files")
    names: set[str] = set()
    for record in records:
        if not isinstance(record, MappingABC) or set(record) != {"relative_path", "size_bytes", "sha256"}:
            raise ValueError("manifest file record has invalid fields")
        name = record["relative_path"]
        if name not in _DATA_FILENAMES or name in names or Path(name).name != name:
            raise ValueError("manifest has duplicate or unsafe file path")
        names.add(name)
        _require_nonnegative_int(record["size_bytes"], f"size for {name}")
        _require_sha256(record["sha256"], f"sha256 for {name}")
    if names != set(_DATA_FILENAMES):
        raise ValueError("manifest does not describe the exact retained data files")


# ---------------------------------------------------------------------------
# Checkpoint tree validation
# ---------------------------------------------------------------------------


def _validate_checkpoint_tree(directory: Path) -> None:
    try:
        metadata = directory.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"checkpoint directory not found: {directory}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("checkpoint path must be a non-symlink directory")
    with os.scandir(directory) as entries:
        found = {entry.name for entry in entries}
    if found != _ALLOWED_FILENAMES:
        raise ValueError(f"checkpoint has unexpected or missing files: {sorted(found ^ _ALLOWED_FILENAMES)}")
    for name in _ALLOWED_FILENAMES:
        _regular_file(directory / name)


def _file_record(path: Path) -> dict[str, Any]:
    _regular_file(path)
    return {"relative_path": path.name, "size_bytes": path.stat().st_size, "sha256": _file_sha256(path)}


# ---------------------------------------------------------------------------
# Adapter and architecture helpers
# ---------------------------------------------------------------------------


def _adapter_type(adapter: torch.nn.Module) -> str:
    return f"{type(adapter).__module__}.{type(adapter).__qualname__}"


def _draft_architecture(adapter: torch.nn.Module) -> dict[str, Any] | None:
    """Return ``{adapter_type, config}``; config may be null when absent.

    ``adapter_type`` is always recorded so that the manifest alone identifies
    the adapter class even when there is no serialisable config.
    """
    config = getattr(adapter, "config", None)
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        wrapped_model = getattr(adapter, "draft_model", None)
        if wrapped_model is None:
            wrapped_model = getattr(adapter, "model", None)
        config = getattr(wrapped_model, "config", None)
        to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if not isinstance(value, MappingABC):
            raise ValueError("adapter config.to_dict() must return a mapping")
        normalized = json.loads(_canonical_json(value).decode("utf-8"))
        return {
            "adapter_type": _adapter_type(adapter),
            "config": normalized,
        }
    # No config available — still record the adapter type with null config.
    return {
        "adapter_type": _adapter_type(adapter),
        "config": None,
    }


def _optimizer_type(optimizer: torch.optim.Optimizer) -> str:
    return f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"


def _validate_adapter_prefixes(adapter: torch.nn.Module) -> None:
    """Reject retained checkpoints for unknown adapter classes."""
    module_type = _adapter_type(adapter)
    prefixes = _ALLOWLISTED_ADAPTER_PREFIXES.get(module_type)
    if prefixes is None:
        raise ValueError(
            f"unsupported adapter class for retained checkpoint: {module_type}; "
            f"supported types: {sorted(_ALLOWLISTED_ADAPTER_PREFIXES)}"
        )
    state = adapter.state_dict()
    for key in state:
        if not any(key.startswith(prefix) for prefix in prefixes):
            raise ValueError(
                f"state key {key!r} does not match any allowlisted prefix "
                f"for {module_type}: {prefixes}"
            )


# ---------------------------------------------------------------------------
# Shared vocabulary and optimizer validation
# ---------------------------------------------------------------------------


def _shared_vocab_parameters(training_step: DFlashTrainingStep) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []
    seen: set[int] = set()
    for module in (training_step.shared_weights.embedding, training_step.shared_weights.lm_head):
        if module is not None:
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    result.append(parameter)
                    seen.add(id(parameter))
    return result


def _validate_tied_output_handoff(shared_weights) -> None:
    """Require tied-output embedding handoff and reject untied heads."""
    if not shared_weights.tied_output_embedding:
        raise ValueError(
            "retained checkpoint requires tied_output_embedding=True; "
            "untied lm_head checkpoints are not supported"
        )
    if shared_weights.lm_head is not None:
        raise ValueError(
            "retained checkpoint requires lm_head=None; "
            "untied lm_head is not None"
        )


def _validate_draft_and_optimizer(
    training_step: DFlashTrainingStep, optimizer: torch.optim.Optimizer
) -> tuple[dict[str, torch.Tensor], TiedVocabIdentity]:
    """Validate training step and optimizer consistency, return draft state and tied identity."""
    # Require tied-output handoff.
    _validate_tied_output_handoff(training_step.shared_weights)
    # Require training_step.selected_layer_ids == runtime_identity.selected_layer_ids.
    # This is checked at save time via the runtime_identity parameter.

    embedding_weight = getattr(training_step.shared_weights.embedding, "weight", None)
    if not isinstance(embedding_weight, torch.Tensor) or not embedding_weight.is_floating_point():
        raise ValueError("shared embedding must expose a floating weight tensor")
    shared = _shared_vocab_parameters(training_step)
    shared_ids = {id(parameter) for parameter in shared}
    named_parameters = dict(training_step.adapter.named_parameters())
    if any(id(parameter) in shared_ids for parameter in named_parameters.values()):
        raise ValueError("shared vocabulary parameters are registered in the draft adapter")
    state = training_step.adapter.state_dict()
    if not state:
        raise ValueError("draft adapter has no state dict entries")

    # Validate allowlisted prefixes.
    _validate_adapter_prefixes(training_step.adapter)

    shared_signatures = {(tuple(value.shape), _tensor_sha256(value)) for value in shared}
    draft_state: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        lower_name = name.lower()
        if any(token in lower_name for token in ("embedding", "embed", "lm_head", "lm-head")):
            raise ValueError(f"draft state contains forbidden shared vocabulary name: {name}")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"draft state entry is not a tensor: {name}")
        if (tuple(value.shape), _tensor_sha256(value)) in shared_signatures:
            raise ValueError(f"draft state serializes a shared vocabulary tensor copy: {name}")
        draft_state[name] = value.detach().clone().contiguous().cpu()
    expected = {id(parameter) for parameter in named_parameters.values() if parameter.requires_grad}
    if not expected:
        raise ValueError("draft adapter has no trainable parameters")
    owned: list[torch.Tensor] = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    owned_ids = [id(parameter) for parameter in owned]
    if len(owned_ids) != len(set(owned_ids)) or set(owned_ids) != expected:
        raise ValueError("optimizer must own exactly the draft trainable parameters")
    if set(owned_ids) & shared_ids:
        raise ValueError("optimizer must not contain shared vocabulary parameters")
    return draft_state, TiedVocabIdentity.from_tensor(embedding_weight)


def _validate_runtime(manifest: CheckpointManifest, runtime: TeacherRuntimeIdentity) -> None:
    expected = (
        runtime.target_name,
        runtime.runtime_image_id,
        runtime.backend,
        runtime.checkpoint_manifest_sha256,
        runtime.selected_layer_ids,
        runtime.hidden_size,
    )
    actual = (
        manifest.target_name,
        manifest.runtime_image_id,
        manifest.backend,
        manifest.checkpoint_manifest_sha256,
        manifest.selected_layer_ids,
        manifest.hidden_size,
    )
    if actual != expected:
        raise ValueError("runtime identity does not match checkpoint manifest")


def _load_validated_ledger(directory: Path, manifest: CheckpointManifest) -> SavedRequestLedger:
    raw_bytes = _load_validated_file_bytes(directory, manifest, REQUEST_LEDGER_FILENAME)
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid request ledger: {exc}") from exc
    ledger = SavedRequestLedger.from_list(data)
    if len(ledger.entries) != manifest.request_ledger_entry_count:
        raise ValueError("request ledger entry count does not match manifest")
    if ledger.sha256 != manifest.request_ledger_sha256:
        raise ValueError("request ledger SHA-256 does not match manifest")
    return ledger


def _validate_and_load_response_ledger(
    directory: Path,
    manifest: CheckpointManifest,
    request_ledger: SavedRequestLedger,
) -> list[dict[str, Any]]:
    """Load response-ledger.json, validate its SHA-256 and entry count."""
    raw_bytes = _load_validated_file_bytes(directory, manifest, RESPONSE_LEDGER_FILENAME)
    actual_sha = _response_ledger_sha256(raw_bytes)
    if actual_sha != manifest.response_ledger_sha256:
        raise ValueError("response ledger SHA-256 does not match manifest")
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid response ledger: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("response ledger must be a JSON array")
    if len(data) != manifest.response_ledger_entry_count:
        raise ValueError("response ledger entry count does not match manifest")
    # Validate and canonicalise each entry.
    examples: list[dict[str, Any]] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, MappingABC):
            raise ValueError(f"response ledger entry {index} must be a JSON object")
        example = ResponseExample.from_mapping(entry)
        examples.append(example.to_dict())
    _validate_response_examples_against_requests(examples, request_ledger)
    return examples


# ---------------------------------------------------------------------------
# Optimizer validation helpers
# ---------------------------------------------------------------------------


def _validate_optimizer_class(optimizer: torch.optim.Optimizer) -> None:
    """Restrict retained optimizer support to torch.optim.AdamW."""
    opt_type = _optimizer_type(optimizer)
    if opt_type != "torch.optim.adamw.AdamW":
        raise ValueError(
            f"retained optimizer support is restricted to torch.optim.AdamW; "
            f"got {opt_type}"
        )


def _extract_optimizer_param_groups(
    optimizer: torch.optim.Optimizer,
    named_parameters: dict[str, torch.Tensor],
) -> list[list[dict[str, Any]]]:
    """Extract ordered parameter names, shapes, and dtypes per group.

    The name-to-tensor mapping is derived from the adapter's ``named_parameters``
    to avoid relying on optimizer-consumed ID ordering.  Each group lists its
    parameters in the order they appear in the group's ``"params"`` list.
    """
    id_to_name: dict[int, str] = {id(tensor): name for name, tensor in named_parameters.items()}
    groups: list[list[dict[str, Any]]] = []
    for group in optimizer.param_groups:
        param_list: list[dict[str, Any]] = []
        for param in group["params"]:
            pid = id(param)
            name = id_to_name.get(pid)
            if name is None:
                raise ValueError(f"optimizer parameter (id={pid}) not found in adapter named_parameters")
            param_list.append({
                "name": name,
                "shape": list(param.shape),
                "dtype": str(param.dtype),
            })
        groups.append(param_list)
    return groups


def _validate_loaded_optimizer_state(
    state: object,
    optimizer: torch.optim.Optimizer,
    manifest: CheckpointManifest,
    named_parameters: dict[str, torch.Tensor],
) -> None:
    """Validate the loaded optimizer pickle against the manifest and caller.

    Checks:
    - Structural validity (state, param_groups).
    - Group count matches manifest.
    - Parameter names, shapes, and dtypes match manifest (ordered comparison).
    - state is restricted to torch.optim.AdamW schema.
    - All saved state tensor shapes match current parameter shapes.
    """
    if not isinstance(state, MappingABC) or set(state) != {"state", "param_groups"}:
        raise ValueError("optimizer checkpoint has invalid structure")
    saved_groups = state["param_groups"]
    if not isinstance(saved_groups, list) or len(saved_groups) != len(optimizer.param_groups):
        raise ValueError("optimizer checkpoint parameter groups do not match caller optimizer")
    current_layout = _extract_optimizer_param_groups(optimizer, named_parameters)
    if current_layout != manifest.optimizer_param_groups:
        raise ValueError("optimizer parameter name/order layout does not match checkpoint")

    for group_idx, (saved_group, manifest_group) in enumerate(
        zip(saved_groups, manifest.optimizer_param_groups, strict=True)
    ):
        if not isinstance(saved_group, MappingABC) or not isinstance(saved_group.get("params"), list):
            raise ValueError(f"optimizer checkpoint has invalid param group {group_idx}")
        saved_params = saved_group["params"]
        # First pass: check parameter names, shapes, dtypes against manifest.
        if len(saved_params) != len(manifest_group):
            raise ValueError(
                f"optimizer param group {group_idx} length ({len(saved_params)}) "
                f"does not match manifest ({len(manifest_group)})"
            )
        for param_idx, (saved_id, manifest_entry) in enumerate(
            zip(saved_params, manifest_group, strict=True)
        ):
            if isinstance(saved_id, bool) or not isinstance(saved_id, int):
                raise ValueError(f"optimizer param identifier is invalid in group {group_idx}, param {param_idx}")
            # Look up the current parameter by manifest name.
            manifest_name = manifest_entry["name"]
            current_param = named_parameters.get(manifest_name)
            if current_param is None:
                raise ValueError(
                    f"optimizer manifest parameter {manifest_name!r} not found in current adapter"
                )
            # Check shape and dtype match.
            manifest_shape = tuple(manifest_entry["shape"])
            manifest_dtype = manifest_entry["dtype"]
            if tuple(current_param.shape) != manifest_shape:
                raise ValueError(
                    f"optimizer parameter {manifest_name!r} shape mismatch: "
                    f"manifest {manifest_shape}, current {tuple(current_param.shape)}"
                )
            if str(current_param.dtype) != manifest_dtype:
                raise ValueError(
                    f"optimizer parameter {manifest_name!r} dtype mismatch: "
                    f"manifest {manifest_dtype}, current {current_param.dtype}"
                )

    # Validate saved state tensors.
    saved_state = state["state"]
    if not isinstance(saved_state, MappingABC):
        raise ValueError("optimizer checkpoint state must be a mapping")
    # Collect all unique parameter IDs referenced in param_groups.
    all_saved_ids: set[int] = set()
    for saved_group in saved_groups:
        all_saved_ids.update(saved_group["params"])
    if set(saved_state) - all_saved_ids:
        raise ValueError("optimizer checkpoint state has orphaned entries not referenced by param_groups")
    if all_saved_ids - set(saved_state):
        raise ValueError("optimizer checkpoint param_groups reference missing state entries")

    # Build PID -> expected shape from manifest (ordered by param groups).
    pid_to_expected: dict[int, tuple[str, tuple[int, ...]]] = {}
    for saved_group, manifest_group in zip(saved_groups, manifest.optimizer_param_groups, strict=True):
        for saved_pid, manifest_entry in zip(saved_group["params"], manifest_group, strict=True):
            if isinstance(saved_pid, bool) or not isinstance(saved_pid, int):
                raise ValueError("optimizer param identifier is invalid")
            pid_to_expected[saved_pid] = (
                manifest_entry["name"],
                tuple(manifest_entry["shape"]),
            )

    # Validate each saved state entry has AdamW schema and matching tensor shapes.
    for pid, state_entry in saved_state.items():
        if not isinstance(state_entry, MappingABC):
            raise ValueError(f"optimizer state entry for pid {pid} must be a mapping")
        # AdamW expected state fields: step (int), exp_avg (tensor), exp_avg_sq (tensor).
        if set(state_entry) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(
                f"optimizer state entry for pid {pid} has unexpected fields: "
                f"{set(state_entry)}; expected step, exp_avg, exp_avg_sq"
            )
        step = state_entry["step"]
        if isinstance(step, torch.Tensor):
            if step.numel() != 1:
                raise ValueError(f"optimizer state entry for pid {pid} step must be scalar")
            step = float(step.item())
        if (
            isinstance(step, bool)
            or not isinstance(step, (int, float))
            or not math.isfinite(step)
            or step < 0
            or float(step).is_integer() is False
        ):
            raise ValueError(f"optimizer state entry for pid {pid} step must be a non-negative integer")
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = state_entry[key]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"optimizer state entry for pid {pid} {key} must be a tensor")
        # Look up expected shape from manifest.
        expected_info = pid_to_expected.get(pid)
        if expected_info is None:
            continue  # state without manifest entry (shouldn't happen after earlier validation)
        expected_name, expected_shape = expected_info
        expected_dtype = named_parameters[expected_name].dtype
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = state_entry[key]
            if tuple(tensor.shape) != expected_shape or tensor.dtype != expected_dtype:
                raise ValueError(
                    f"optimizer state entry for pid {pid} {key} shape/dtype "
                    f"does not match expected parameter {expected_name!r}"
                )


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_replace(staging: Path, final: Path) -> None:
    """Atomically publish on Linux without a check-then-rename overwrite race."""
    if os.path.lexists(final):
        raise FileExistsError(f"checkpoint directory already exists: {final}")
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:  # pragma: no cover - Linux CI and retained pilot hosts use renameat2.
        raise RuntimeError("safe no-replace checkpoint publication requires renameat2") from exc
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(staging), -100, os.fsencode(final), 1) != 0:  # RENAME_NOREPLACE
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"checkpoint directory already exists: {final}")
        raise OSError(error, os.strerror(error), str(final))


def _retained_read_no_follow(path: Path) -> bytes:
    """Read the full file content with O_NOFOLLOW to avoid symlink TOCTOU."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        # Verify it's still a regular file (not a symlink that changed under us).
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"not a regular file after O_NOFOLLOW open: {path.name}")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _load_validated_file_bytes(
    directory: Path,
    manifest: CheckpointManifest,
    filename: str,
) -> bytes:
    data = _retained_read_no_follow(directory / filename)
    record = next(
        (item for item in manifest.files if item["relative_path"] == filename),
        None,
    )
    if record is None:
        raise ValueError(f"checkpoint manifest lacks file record: {filename}")
    if len(data) != record["size_bytes"]:
        raise ValueError(f"{filename} size does not match manifest after O_NOFOLLOW")
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"{filename} SHA-256 does not match manifest after O_NOFOLLOW")
    return data


def _load_validated_optimizer_bytes(
    directory: Path, manifest: CheckpointManifest
) -> bytes:
    """Read optimizer.pt through O_NOFOLLOW, re-check hash, return retained bytes."""
    return _load_validated_file_bytes(directory, manifest, OPTIMIZER_FILENAME)


def _load_validated_draft_weights_bytes(
    directory: Path, manifest: CheckpointManifest
) -> bytes:
    """Read draft-model.safetensors through O_NOFOLLOW, re-check hash."""
    return _load_validated_file_bytes(directory, manifest, DRAFT_WEIGHTS_FILENAME)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_checkpoint(
    directory: str | Path,
    *,
    ring: BoundedFeatureRing,
    training_step: DFlashTrainingStep,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    request_ledger: SavedRequestLedger | Sequence[Mapping[str, Any]],
    response_examples: Sequence[ResponseExample | Mapping[str, Any]],
    runtime_identity: TeacherRuntimeIdentity,
) -> CheckpointManifest:
    """Write an immutable draft-only checkpoint and its manifest-digest sidecar.

    The returned ``manifest.sha256`` must be retained outside this directory and
    supplied as ``expected_manifest_sha256`` when verifying or resuming.

    ``response_examples`` must contain exactly ``completed_steps`` entries whose
    ``request_id``-derived order matches the completed request ledger.
    """
    _require_positive_int(completed_steps, "completed_steps")
    final = Path(directory).absolute()
    if os.path.lexists(final):
        raise FileExistsError(f"checkpoint directory already exists: {final}")
    if len(ring):
        raise ValueError("cannot checkpoint with active ring items")
    if ring.runtime_identity != runtime_identity:
        raise ValueError("ring runtime identity does not match runtime_identity")
    ledger = request_ledger if isinstance(request_ledger, SavedRequestLedger) else SavedRequestLedger.from_list(request_ledger)
    if len(ledger.entries) != completed_steps:
        raise ValueError("request ledger entry count does not match completed_steps")

    # Bind exact prompt/response token IDs to committed request order and counts.
    _validate_response_examples_against_requests(response_examples, ledger)
    response_bytes = _response_examples_canonical_bytes(response_examples)
    response_sha = _response_ledger_sha256(response_bytes)

    # Validate training_step.selected_layer_ids == runtime_identity.selected_layer_ids.
    if tuple(training_step.selected_layer_ids) != tuple(runtime_identity.selected_layer_ids):
        raise ValueError("training_step.selected_layer_ids must equal runtime_identity.selected_layer_ids")

    # Validate optimizer class.
    _validate_optimizer_class(optimizer)

    draft_state, tied_vocab = _validate_draft_and_optimizer(training_step, optimizer)

    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=final.parent))
    # We intentionally do not remove staging after an error: it is unique to this
    # invocation, and avoiding recursive cleanup cannot delete another process's path.
    safe_save_file(draft_state, staging / DRAFT_WEIGHTS_FILENAME, metadata={"format": "torch"})
    torch.save(optimizer.state_dict(), staging / OPTIMIZER_FILENAME)
    (staging / REQUEST_LEDGER_FILENAME).write_bytes(_canonical_json(ledger.to_list()))
    (staging / RESPONSE_LEDGER_FILENAME).write_bytes(response_bytes)

    # Build optimizer_param_groups from current optimizer and adapter state.
    optimizer_param_groups = _extract_optimizer_param_groups(
        optimizer, dict(training_step.adapter.named_parameters())
    )

    manifest = CheckpointManifest(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        step_count=completed_steps,
        target_name=runtime_identity.target_name,
        runtime_image_id=runtime_identity.runtime_image_id,
        backend=runtime_identity.backend,
        checkpoint_manifest_sha256=runtime_identity.checkpoint_manifest_sha256,
        selected_layer_ids=runtime_identity.selected_layer_ids,
        hidden_size=runtime_identity.hidden_size,
        request_ledger_sha256=ledger.sha256,
        request_ledger_entry_count=len(ledger.entries),
        response_ledger_sha256=response_sha,
        response_ledger_entry_count=len(response_examples),
        tied_vocab_identity=tied_vocab,
        draft_architecture=_draft_architecture(training_step.adapter),
        optimizer_type=_optimizer_type(optimizer),
        optimizer_param_groups=optimizer_param_groups,
        files=tuple(_file_record(staging / name) for name in _DATA_FILENAMES),
    )
    manifest_bytes = _canonical_json(manifest.to_dict())
    (staging / MANIFEST_FILENAME).write_bytes(manifest_bytes)
    (staging / MANIFEST_SHA256_FILENAME).write_text(manifest.sha256 + "\n", encoding="ascii")
    for name in _ALLOWED_FILENAMES:
        _fsync_file(staging / name)
    _fsync_directory(staging)
    _publish_no_replace(staging, final)
    _fsync_directory(final.parent)
    return manifest


def verify_checkpoint_directory(
    directory: str | Path,
    *,
    runtime_identity: TeacherRuntimeIdentity,
    expected_manifest_sha256: str,
) -> CheckpointManifest:
    """Verify the trusted external manifest root, tree shape, and data hashes."""
    _require_sha256(expected_manifest_sha256, "expected_manifest_sha256")
    directory = Path(directory).absolute()
    _validate_checkpoint_tree(directory)
    manifest_path = directory / MANIFEST_FILENAME
    actual_manifest_sha = _file_sha256(manifest_path)
    if actual_manifest_sha != expected_manifest_sha256:
        raise ValueError("checkpoint manifest digest does not match the expected external root")
    sidecar = (directory / MANIFEST_SHA256_FILENAME).read_text(encoding="ascii")
    if sidecar != expected_manifest_sha256 + "\n":
        raise ValueError("checkpoint manifest digest sidecar does not match the expected external root")
    try:
        manifest = CheckpointManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid checkpoint manifest: {exc}") from exc
    if manifest.sha256 != actual_manifest_sha:
        raise ValueError("checkpoint manifest is not canonical")
    _validate_runtime(manifest, runtime_identity)
    for record in manifest.files:
        path = directory / record["relative_path"]
        _regular_file(path)
        if path.stat().st_size != record["size_bytes"] or _file_sha256(path) != record["sha256"]:
            raise ValueError(f"checkpoint file hash or size mismatch: {record['relative_path']}")
    return manifest


def resume_checkpoint(
    directory: str | Path,
    *,
    training_step: DFlashTrainingStep,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    runtime_identity: TeacherRuntimeIdentity,
    expected_manifest_sha256: str,
) -> tuple[CheckpointManifest, SavedRequestLedger, list[dict[str, Any]]]:
    """Load a verified checkpoint from a trusted-local path.

    Returns ``(manifest, request_ledger, response_examples_as_dicts)``.

    Ledger and all structural checks happen before mutating the caller's adapter
    or optimizer.  The optimizer pickle remains a trusted-local-only caveat.
    All file loads use O_NOFOLLOW-retained private bytes to close the
    verify-then-reopen TOCTOU window.
    """
    directory = Path(directory).absolute()
    manifest = verify_checkpoint_directory(
        directory, runtime_identity=runtime_identity, expected_manifest_sha256=expected_manifest_sha256
    )

    # Load ledgers from verified tree.
    ledger = _load_validated_ledger(directory, manifest)
    response_examples = _validate_and_load_response_ledger(directory, manifest, ledger)

    # Validate training_step.selected_layer_ids == runtime_identity.selected_layer_ids.
    if tuple(training_step.selected_layer_ids) != tuple(runtime_identity.selected_layer_ids):
        raise ValueError("training_step.selected_layer_ids must equal runtime_identity.selected_layer_ids")

    # Validate tied-output handoff.
    _validate_tied_output_handoff(training_step.shared_weights)

    _, tied_vocab = _validate_draft_and_optimizer(training_step, optimizer)
    if tied_vocab != manifest.tied_vocab_identity:
        raise ValueError("tied vocabulary identity does not match checkpoint")

    # Validate adapter class/config identity.
    arch = _draft_architecture(training_step.adapter)
    if arch != manifest.draft_architecture:
        raise ValueError("draft architecture/config does not match checkpoint")

    if _optimizer_type(optimizer) != manifest.optimizer_type:
        raise ValueError("optimizer type does not match checkpoint")

    # Validate optimizer class is AdamW.
    _validate_optimizer_class(optimizer)

    # Parse exactly the O_NOFOLLOW-retained bytes; do not reopen a verified path.
    draft_bytes = _load_validated_draft_weights_bytes(directory, manifest)
    saved_state = {
        name: value.to(device=device)
        for name, value in safe_load(draft_bytes).items()
    }

    current_state = training_step.adapter.state_dict()
    if set(saved_state) != set(current_state):
        raise ValueError("draft adapter state keys do not match checkpoint")
    for name, value in saved_state.items():
        current = current_state[name]
        if value.shape != current.shape or value.dtype != current.dtype:
            raise ValueError(f"draft adapter state shape or dtype mismatch: {name}")

    # Load optimizer via O_NOFOLLOW retained bytes with weights_only=True.
    optimizer_bytes = _load_validated_optimizer_bytes(directory, manifest)
    optimizer_state = torch.load(io.BytesIO(optimizer_bytes), map_location=device, weights_only=True)
    _validate_loaded_optimizer_state(
        optimizer_state, optimizer, manifest, dict(training_step.adapter.named_parameters())
    )

    # Validate allowlisted adapter prefixes.
    _validate_adapter_prefixes(training_step.adapter)

    training_step.adapter.load_state_dict(saved_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    return manifest, ledger, response_examples
