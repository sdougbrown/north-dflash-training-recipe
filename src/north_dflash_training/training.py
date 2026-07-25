"""Optional-PyTorch, CPU-testable DFlash training-step contract.

This module deliberately contains no teacher loader.  A caller must provide
already-extracted, detached clean target states in :class:`TeacherFeatureBundle`.
The contract applies the committed packed layout: target keys are the clean
prefix ``[0, C)`` and draft query keys follow them.  It predicts each masked
future at its own query position (there is no causal-LM label shift).

The small adapter in this module is only a synthetic gradient-flow fixture.  It
is not an implementation or approximation of the Qwen3 DFlash draft.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - exercised in dependency-free environments.
    raise ImportError(
        "PyTorch training support requires the optional dependency; install "
        "north-dflash-training-scaffold[torch]."
    ) from exc

from .schema import IGNORE_INDEX
from .torch_layout import TorchSparseTrainingBatch


@dataclass(frozen=True)
class TeacherFeatureBundle:
    """Detached hidden states for selected layers at clean target positions.

    ``hidden_states[i]`` belongs exactly to ``selected_layer_ids[i]``; layer
    IDs are intentionally not sorted or re-derived.  Each state has shape
    ``[B, C, H]``.  ``clean_positions`` must be the contiguous absolute clean
    prefix ``0..C-1`` used by the packed layout's context keys.  Extraction of
    these states from North's exact AutoGPTQ teacher remains unimplemented.
    """

    selected_layer_ids: tuple[int, ...]
    hidden_states: tuple[torch.Tensor, ...]
    clean_positions: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.hidden_states[0].shape[0]

    @property
    def context_length(self) -> int:
        return self.hidden_states[0].shape[1]

    @property
    def hidden_size(self) -> int:
        return self.hidden_states[0].shape[2]

    def validate(self, *, expected_layer_ids: tuple[int, ...] | None = None) -> None:
        if not self.selected_layer_ids or len(self.selected_layer_ids) != len(self.hidden_states):
            raise ValueError("selected_layer_ids and hidden_states must be non-empty and equally sized")
        if len(set(self.selected_layer_ids)) != len(self.selected_layer_ids):
            raise ValueError("selected_layer_ids must be distinct and preserve extractor order")
        if any(not isinstance(layer_id, int) or isinstance(layer_id, bool) or layer_id < 0
               for layer_id in self.selected_layer_ids):
            raise ValueError("selected_layer_ids must be non-negative integers")
        if expected_layer_ids is not None and self.selected_layer_ids != expected_layer_ids:
            raise ValueError("teacher feature layer order does not match the draft contract")
        first = self.hidden_states[0]
        if first.ndim != 3 or not first.is_floating_point():
            raise ValueError("teacher states must be floating [B, C, H] tensors")
        if first.requires_grad:
            raise ValueError("teacher states must be detached/frozen before the draft step")
        for state in self.hidden_states:
            if state.shape != first.shape or state.dtype != first.dtype or state.device != first.device:
                raise ValueError("all selected teacher states must share shape, dtype, and device")
            if state.requires_grad:
                raise ValueError("teacher states must be detached/frozen before the draft step")
        if self.clean_positions.dtype != torch.int64 or self.clean_positions.shape != (self.context_length,):
            raise ValueError("clean_positions must be an int64 [C] tensor")
        if self.clean_positions.device != first.device:
            raise ValueError("teacher states and clean_positions must share a device")
        expected_positions = torch.arange(self.context_length, device=first.device, dtype=torch.int64)
        if not torch.equal(self.clean_positions, expected_positions):
            raise ValueError("clean_positions must be the contiguous clean prefix 0..C-1")


def concatenate_target_features(
    bundle: TeacherFeatureBundle,
    *,
    expected_layer_ids: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Concatenate selected states in declared layer order into ``[B, C, L*H]``."""
    bundle.validate(expected_layer_ids=expected_layer_ids)
    return torch.cat(bundle.hidden_states, dim=-1)


@dataclass(frozen=True)
class FrozenSharedWeights:
    """Frozen direct references for target input and output vocabulary weights.

    North/Cohere2 has no separate ``lm_head`` tensor: its target logits processor
    projects with ``model.embed_tokens`` because word embeddings are tied. The
    tied handoff therefore uses the exact embedding parameter for both lookup
    and ``F.linear`` output projection. No vocabulary weights are copied or
    registered under :class:`DFlashTrainingStep`.
    """

    embedding: nn.Module
    lm_head: nn.Module | None
    tied_output_embedding: bool = False
    mask_token_id: int | None = None

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.eval()

    @classmethod
    def handoff(cls, embedding: nn.Module, lm_head: nn.Module) -> "FrozenSharedWeights":
        cls._freeze(embedding)
        cls._freeze(lm_head)
        return cls(embedding=embedding, lm_head=lm_head)

    @classmethod
    def handoff_tied_embedding(
        cls,
        embedding: nn.Module,
        *,
        mask_token_id: int,
    ) -> "FrozenSharedWeights":
        weight = getattr(embedding, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError("tied output handoff requires a rank-two embedding weight")
        if (
            isinstance(mask_token_id, bool)
            or not isinstance(mask_token_id, int)
            or mask_token_id < 0
            or mask_token_id >= weight.shape[0]
        ):
            raise ValueError("mask_token_id must index the shared embedding vocabulary")
        cls._freeze(embedding)
        probe_ids = torch.tensor([[mask_token_id]], dtype=torch.int64, device=weight.device)
        probe = embedding(probe_ids)
        if probe.shape != (1, 1, weight.shape[1]) or not torch.equal(
            probe[0, 0], weight[mask_token_id]
        ):
            raise ValueError("mask token lookup must use the tied shared embedding row")
        return cls(
            embedding=embedding,
            lm_head=None,
            tied_output_embedding=True,
            mask_token_id=mask_token_id,
        )

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        if embeddings.ndim != 3 or embeddings.shape[:2] != input_ids.shape:
            raise ValueError("shared embedding must return [B, Q, H]")
        return embeddings

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tied_output_embedding:
            weight = getattr(self.embedding, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                raise ValueError("tied shared embedding weight is unavailable")
            logits = F.linear(hidden_states, weight)
        else:
            if self.lm_head is None:
                raise ValueError("untied output handoff requires an LM head")
            logits = self.lm_head(hidden_states)
        if logits.ndim != 3 or logits.shape[:2] != hidden_states.shape[:2]:
            raise ValueError("shared output projection must return [B, Q, V]")
        return logits


@runtime_checkable
class DraftAdapter(Protocol):
    """The architecture-specific draft boundary used by the training step."""

    def forward(
        self,
        *,
        target_features: torch.Tensor,
        noise_embeddings: torch.Tensor,
        position_ids: torch.Tensor,
        dense_visibility: torch.Tensor,
    ) -> torch.Tensor: ...


class DraftAdapterModule(nn.Module, ABC):
    """``nn.Module`` base for implementations of :class:`DraftAdapter`."""

    @abstractmethod
    def forward(
        self,
        *,
        target_features: torch.Tensor,
        noise_embeddings: torch.Tensor,
        position_ids: torch.Tensor,
        dense_visibility: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class SyntheticDraftAdapter(DraftAdapterModule):
    """Tiny trainable adapter for CPU contract tests, not a production draft.

    It projects the ordered concatenated teacher features, averages only the
    layout-visible clean context for each query, and combines that with frozen
    noise embeddings and a scalar absolute-position feature.  It deliberately
    does not implement DFlash attention, RoPE, caching, or a Qwen architecture.
    """

    def __init__(self, *, target_feature_width: int, hidden_size: int) -> None:
        super().__init__()
        if target_feature_width < 1 or hidden_size < 1:
            raise ValueError("target_feature_width and hidden_size must be positive")
        self.target_projection = nn.Linear(target_feature_width, hidden_size)
        self.noise_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.position_projection = nn.Linear(1, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        *,
        target_features: torch.Tensor,
        noise_embeddings: torch.Tensor,
        position_ids: torch.Tensor,
        dense_visibility: torch.Tensor,
    ) -> torch.Tensor:
        if target_features.ndim != 3 or noise_embeddings.ndim != 3:
            raise ValueError("target_features and noise_embeddings must be rank-three")
        batch_size, context_length, _ = target_features.shape
        if noise_embeddings.shape[0] != batch_size:
            raise ValueError("target features and noise embeddings must share batch size")
        query_length = noise_embeddings.shape[1]
        if position_ids.shape != (batch_size, query_length) or position_ids.dtype != torch.int64:
            raise ValueError("position_ids must be int64 [B, Q]")
        if dense_visibility.shape != (query_length, context_length + query_length):
            raise ValueError("dense_visibility must be [Q, C + Q]")
        if dense_visibility.dtype != torch.bool:
            raise ValueError("dense_visibility must be boolean")
        projected_context = self.target_projection(target_features)
        visible_context = dense_visibility[:, :context_length].to(dtype=projected_context.dtype)
        divisor = visible_context.sum(dim=-1, keepdim=True).clamp_min(1)
        context_per_query = torch.einsum("qc,bch->bqh", visible_context / divisor, projected_context)
        position_feature = position_ids.to(dtype=noise_embeddings.dtype).unsqueeze(-1)
        return self.output_projection(torch.tanh(
            context_per_query + self.noise_projection(noise_embeddings) + self.position_projection(position_feature)
        ))


def weighted_masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_weights: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Weighted mean CE over exactly the masked, non-ignored future labels.

    The denominator is the sum of active exponential weights, rather than the
    number of packed positions.  Ignored anchors contribute neither numerator
    nor denominator.  The layout already aligns logits and labels at the same
    query position, so this function deliberately does not shift either tensor.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits must be [B, Q, V] and labels [B, Q]")
    if labels.dtype != torch.int64 or loss_mask.shape != labels.shape or loss_mask.dtype != torch.bool:
        raise ValueError("labels must be int64 and loss_mask must be bool [B, Q]")
    if loss_weights.shape != labels.shape or not loss_weights.is_floating_point():
        raise ValueError("loss_weights must be floating [B, Q]")
    if not torch.equal(loss_mask, labels.ne(ignore_index)):
        raise ValueError("loss_mask must select exactly the non-ignored future labels")
    if not torch.isfinite(loss_weights).all() or (loss_weights[loss_mask] <= 0).any():
        raise ValueError("active loss weights must be finite and positive")
    active_weight = loss_weights[loss_mask]
    if active_weight.numel() == 0:
        raise ValueError("at least one masked future label is required")
    per_position = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=ignore_index, reduction="none"
    ).reshape_as(loss_weights)
    return (per_position[loss_mask] * active_weight).sum() / active_weight.sum()


@dataclass(frozen=True)
class TrainingStepOutput:
    logits: torch.Tensor
    loss: torch.Tensor
    target_features: torch.Tensor


class DFlashTrainingStep(nn.Module):
    """Compose frozen shared target weights, a draft adapter, and packed data."""

    def __init__(
        self,
        *,
        adapter: DraftAdapterModule,
        shared_weights: FrozenSharedWeights,
        selected_layer_ids: tuple[int, ...],
    ) -> None:
        super().__init__()
        if not selected_layer_ids:
            raise ValueError("selected_layer_ids must not be empty")
        self.adapter = adapter
        self.shared_weights = shared_weights
        self.selected_layer_ids = selected_layer_ids

    def forward(self, batch: TorchSparseTrainingBatch, teacher_features: TeacherFeatureBundle) -> TrainingStepOutput:
        batch.validate()
        target_features = concatenate_target_features(
            teacher_features, expected_layer_ids=self.selected_layer_ids
        )
        if target_features.shape[:2] != (batch.input_ids.shape[0], batch.context_length):
            raise ValueError("teacher feature batch/context dimensions do not align with packed layout")
        if target_features.device != batch.input_ids.device:
            raise ValueError("teacher features and packed batch must share a device")
        noise_embeddings = self.shared_weights.embed(batch.input_ids)
        hidden_states = self.adapter(
            target_features=target_features,
            noise_embeddings=noise_embeddings,
            position_ids=batch.absolute_query_positions,
            dense_visibility=batch.dense_visibility,
        )
        if hidden_states.shape != noise_embeddings.shape:
            raise ValueError("draft adapter must return [B, Q, shared_hidden_size]")
        logits = self.shared_weights.logits(hidden_states)
        loss = weighted_masked_cross_entropy(logits, batch.labels, batch.loss_mask, batch.loss_weights)
        return TrainingStepOutput(logits=logits, loss=loss, target_features=target_features)
