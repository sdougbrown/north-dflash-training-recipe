"""Bounded consumption of vLLM hidden-state connector artifacts.

The connector file is a transient handoff, not a feature dataset. This module
validates one request at a time, takes ownership of bounded tensor clones, and
places request-preserving batches into a fail-closed in-memory ring. It never
writes or deletes connector files; the orchestration layer owns that lifecycle.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import re

try:
    import torch
    from safetensors.torch import load_file
except ImportError as exc:  # pragma: no cover - optional runtime dependency.
    raise ImportError(
        "Feature streaming requires the runtime optional dependencies; install "
        "north-dflash-training-scaffold[runtime]."
    ) from exc

from .training import TeacherFeatureBundle

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class TeacherRuntimeIdentity:
    """Exact verifier/runtime contract for one isolated feature stream."""

    target_name: str
    checkpoint_manifest_sha256: str
    runtime_image_id: str
    backend: str
    selected_layer_ids: tuple[int, ...]
    hidden_size: int = 2048
    prefix_caching_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.target_name or not self.backend:
            raise ValueError("target and backend must be identified")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.runtime_image_id) is None:
            raise ValueError("runtime image ID must be a pinned sha256 digest")
        if _SHA256.fullmatch(self.checkpoint_manifest_sha256) is None:
            raise ValueError("checkpoint manifest must be a lowercase SHA-256 digest")
        if not self.selected_layer_ids or self.selected_layer_ids != tuple(
            sorted(set(self.selected_layer_ids))
        ):
            raise ValueError("selected layer IDs must be non-empty, unique, and ascending")
        if any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in self.selected_layer_ids
        ):
            raise ValueError("selected layer IDs must be non-negative integers")
        if isinstance(self.hidden_size, bool) or self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.prefix_caching_enabled:
            raise ValueError("teacher feature extraction requires prefix caching disabled")


@dataclass(frozen=True)
class StreamedTeacherBatch:
    """One request boundary and its owned, detached teacher features."""

    request_id: str
    runtime_identity: TeacherRuntimeIdentity
    token_ids: torch.Tensor
    features: TeacherFeatureBundle
    feature_bytes: int

    @property
    def token_count(self) -> int:
        return int(self.token_ids.numel())

    def validate(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.token_ids.dtype != torch.int64 or self.token_ids.ndim != 2:
            raise ValueError("token_ids must be int64 [B, C]")
        if self.token_ids.shape != (
            self.features.batch_size,
            self.features.context_length,
        ):
            raise ValueError("token IDs and teacher features must align")
        self.features.validate(expected_layer_ids=self.runtime_identity.selected_layer_ids)
        if self.features.hidden_size != self.runtime_identity.hidden_size:
            raise ValueError("teacher feature width does not match runtime identity")
        expected_bytes = sum(
            state.numel() * state.element_size() for state in self.features.hidden_states
        )
        if self.feature_bytes != expected_bytes:
            raise ValueError("feature byte count does not match owned tensors")


def load_connector_feature_batch(
    path: str | Path,
    *,
    request_id: str,
    runtime_identity: TeacherRuntimeIdentity,
    expected_token_ids: torch.Tensor,
) -> StreamedTeacherBatch:
    """Validate and take bounded ownership of one safetensors connector result.

    vLLM writes ``hidden_states`` as ``[C, L, H]`` and ``token_ids`` as ``[C]``.
    The returned training bundle owns clones shaped ``[1, C, H]`` so the caller
    may release its transient connector artifact after this function returns.
    """
    if runtime_identity.prefix_caching_enabled:
        raise ValueError("teacher feature extraction requires prefix caching disabled")
    source = Path(path)
    if not source.is_file():
        raise ValueError("connector feature path must be an existing regular file")
    tensors = load_file(source, device="cpu")
    if set(tensors) != {"hidden_states", "token_ids"}:
        raise ValueError("connector artifact must contain only hidden_states and token_ids")
    hidden_states = tensors["hidden_states"]
    token_ids = tensors["token_ids"]
    layer_count = len(runtime_identity.selected_layer_ids)
    if hidden_states.ndim != 3 or hidden_states.shape[1:] != (
        layer_count,
        runtime_identity.hidden_size,
    ):
        raise ValueError("hidden_states must be [C, selected_layers, hidden_size]")
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("teacher connector hidden states must remain BF16")
    if not torch.isfinite(hidden_states).all().item():
        raise ValueError("teacher connector hidden states must be finite")
    if torch.count_nonzero(hidden_states).item() == 0:
        raise ValueError("teacher connector hidden states must not be all zero")
    if token_ids.dtype != torch.int64 or token_ids.shape != (hidden_states.shape[0],):
        raise ValueError("connector token_ids must be int64 [C]")
    if (
        expected_token_ids.dtype != torch.int64
        or expected_token_ids.ndim != 1
        or expected_token_ids.device.type != "cpu"
    ):
        raise ValueError("expected_token_ids must be CPU int64 [C]")
    if not torch.equal(token_ids, expected_token_ids):
        raise ValueError("connector token IDs do not match the request ledger")

    owned_states = tuple(
        hidden_states[:, index, :].unsqueeze(0).clone().detach()
        for index in range(layer_count)
    )
    clean_positions = torch.arange(hidden_states.shape[0], dtype=torch.int64)
    bundle = TeacherFeatureBundle(
        selected_layer_ids=runtime_identity.selected_layer_ids,
        hidden_states=owned_states,
        clean_positions=clean_positions,
    )
    result = StreamedTeacherBatch(
        request_id=request_id,
        runtime_identity=runtime_identity,
        token_ids=token_ids.unsqueeze(0).clone(),
        features=bundle,
        feature_bytes=sum(state.numel() * state.element_size() for state in owned_states),
    )
    result.validate()
    return result


class BoundedFeatureRing:
    """Fail-closed FIFO ring with request boundaries and producer backpressure.

    Overflow rejects the new batch without evicting unconsumed features. This
    prevents silent teacher/example misalignment while keeping memory strictly
    bounded by item, token, and byte limits.
    """

    def __init__(
        self,
        *,
        runtime_identity: TeacherRuntimeIdentity,
        max_items: int,
        max_tokens: int,
        max_bytes: int,
    ) -> None:
        for name, value in {
            "max_items": max_items,
            "max_tokens": max_tokens,
            "max_bytes": max_bytes,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.runtime_identity = runtime_identity
        self.max_items = max_items
        self.max_tokens = max_tokens
        self.max_bytes = max_bytes
        self._items: deque[StreamedTeacherBatch] = deque()
        self._request_ids: set[str] = set()
        self._tokens = 0
        self._bytes = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def token_count(self) -> int:
        return self._tokens

    @property
    def feature_bytes(self) -> int:
        return self._bytes

    def put(self, batch: StreamedTeacherBatch) -> None:
        batch.validate()
        if batch.runtime_identity != self.runtime_identity:
            raise ValueError("mixed target/runtime identities are forbidden")
        if batch.request_id in self._request_ids:
            raise ValueError("duplicate active request_id")
        if (
            len(self._items) + 1 > self.max_items
            or self._tokens + batch.token_count > self.max_tokens
            or self._bytes + batch.feature_bytes > self.max_bytes
        ):
            raise BufferError("bounded feature ring is full; consumer backpressure required")
        self._items.append(batch)
        self._request_ids.add(batch.request_id)
        self._tokens += batch.token_count
        self._bytes += batch.feature_bytes

    def popleft(self) -> StreamedTeacherBatch:
        if not self._items:
            raise IndexError("bounded feature ring is empty")
        batch = self._items.popleft()
        self._request_ids.remove(batch.request_id)
        self._tokens -= batch.token_count
        self._bytes -= batch.feature_bytes
        return batch
