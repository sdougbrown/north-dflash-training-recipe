"""Optional PyTorch materialization of the audited DFlash batch layout.

Importing :mod:`north_dflash_training` remains dependency-free.  Import this
module only in an environment with the ``torch`` extra installed.

The materialized batch has one packed batch item.  Query tensors are shaped
``[1, Q]``.  The attention oracle is shaped ``[Q, C + Q]``: its first ``C``
keys are frozen clean target-context positions ``0..C-1`` and its final ``Q``
keys are the packed query tokens.  It is deliberately a cross-attention-style
mask, rather than a claim that a complete draft-model input layout exists.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

try:
    import torch
except ImportError as exc:  # pragma: no cover - only exercised without the optional extra.
    raise ImportError(
        "PyTorch layout support requires the optional dependency; install "
        "north-dflash-training-scaffold[torch]."
    ) from exc

from .layout import SparseTrainingBatchLayout


@dataclass(frozen=True)
class TorchSparseTrainingBatch:
    """Single-example tensor inputs plus an auditable attention oracle.

    ``dense_visibility`` has no batch/head dimensions because its rule is
    independent of both.  FlexAttention receives the same rule through
    :func:`make_flex_attention_visibility_predicate`.
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    loss_weights: torch.Tensor
    block_ids: torch.Tensor
    anchor_positions: torch.Tensor
    absolute_query_positions: torch.Tensor
    context_positions: torch.Tensor
    dense_visibility: torch.Tensor
    mask_token_id: int

    @property
    def query_tokens(self) -> torch.Tensor:
        """Alias for ``input_ids`` matching the dependency-free layout."""
        return self.input_ids

    @property
    def position_weights(self) -> torch.Tensor:
        """Alias for ``loss_weights`` matching the dependency-free layout."""
        return self.loss_weights

    @property
    def num_queries(self) -> int:
        return self.input_ids.shape[1]

    @property
    def context_length(self) -> int:
        return self.context_positions.shape[0]

    @property
    def key_length(self) -> int:
        return self.context_length + self.num_queries

    def validate(self) -> None:
        """Validate tensor dtypes, shapes, and the dense visibility oracle."""
        query_shape = (1, self.num_queries)
        integer_tensors = (
            ("input_ids", self.input_ids),
            ("labels", self.labels),
            ("block_ids", self.block_ids),
            ("anchor_positions", self.anchor_positions),
            ("absolute_query_positions", self.absolute_query_positions),
        )
        for name, value in integer_tensors:
            if value.shape != query_shape or value.dtype != torch.int64:
                raise ValueError(f"{name} must be an int64 tensor shaped [1, Q]")
        if self.loss_mask.shape != query_shape or self.loss_mask.dtype != torch.bool:
            raise ValueError("loss_mask must be a bool tensor shaped [1, Q]")
        if self.loss_weights.shape != query_shape or not self.loss_weights.is_floating_point():
            raise ValueError("loss_weights must be a floating tensor shaped [1, Q]")
        if (
            self.context_positions.shape != (self.context_length,)
            or self.context_positions.dtype != torch.int64
        ):
            raise ValueError("context_positions must be an int64 tensor shaped [C]")
        if self.dense_visibility.shape != (self.num_queries, self.key_length):
            raise ValueError("dense_visibility must be a bool tensor shaped [Q, C + Q]")
        if self.dense_visibility.dtype != torch.bool:
            raise ValueError("dense_visibility must have bool dtype")
        tensors = (
            self.labels,
            self.loss_mask,
            self.loss_weights,
            self.block_ids,
            self.anchor_positions,
            self.absolute_query_positions,
            self.context_positions,
            self.dense_visibility,
        )
        if any(value.device != self.input_ids.device for value in tensors):
            raise ValueError("all tensor inputs and the visibility oracle must share a device")
        expected_context = torch.arange(
            self.context_length, dtype=torch.int64, device=self.input_ids.device
        )
        if not torch.equal(self.context_positions, expected_context):
            raise ValueError("context_positions must be contiguous absolute positions starting at zero")
        expected = dense_visibility_oracle(self)
        if not torch.equal(self.dense_visibility, expected):
            raise ValueError("dense_visibility does not match the DFlash visibility contract")


def dense_visibility_oracle(batch: TorchSparseTrainingBatch) -> torch.Tensor:
    """Return the boolean ``[Q, C + Q]`` visibility rule for ``batch``.

    A query sees every frozen context key through its absolute clean anchor and
    only query keys from its own sampled block.  Thus neither future target
    context nor another sampled block can leak into a row.
    """
    query_count = batch.num_queries
    context_count = batch.context_length
    device = batch.input_ids.device
    context_visible = torch.arange(context_count, device=device).unsqueeze(0) <= (
        batch.anchor_positions[0].unsqueeze(1)
    )
    query_visible = batch.block_ids[0].unsqueeze(1) == batch.block_ids[0].unsqueeze(0)
    return torch.cat((context_visible, query_visible), dim=1).to(dtype=torch.bool)


def build_torch_training_batch(
    layout: SparseTrainingBatchLayout,
    *,
    device: torch.device | str | None = None,
    weight_dtype: torch.dtype = torch.float32,
) -> TorchSparseTrainingBatch:
    """Materialize one audited layout as CPU/GPU-neutral PyTorch tensors.

    The default device is CPU.  ``weight_dtype`` must be floating point; token
    IDs, labels, positions, and block IDs are always ``torch.int64`` and the
    loss mask/oracle are always ``torch.bool``.
    """
    layout.validate()
    if not torch.empty((), dtype=weight_dtype).is_floating_point():
        raise ValueError("weight_dtype must be a floating-point torch dtype")

    query_count = layout.num_queries
    context_count = max(layout.block_anchor_positions, default=-1) + 1
    tensor_kwargs = {"device": device}
    batch = TorchSparseTrainingBatch(
        input_ids=torch.tensor(layout.query_tokens, dtype=torch.int64, **tensor_kwargs).unsqueeze(
            0
        ),
        labels=torch.tensor(layout.labels, dtype=torch.int64, **tensor_kwargs).unsqueeze(0),
        loss_mask=torch.tensor(layout.loss_mask, dtype=torch.bool, **tensor_kwargs).unsqueeze(0),
        loss_weights=torch.tensor(
            layout.position_weights, dtype=weight_dtype, **tensor_kwargs
        ).unsqueeze(0),
        block_ids=torch.tensor(layout.block_ids, dtype=torch.int64, **tensor_kwargs).unsqueeze(0),
        anchor_positions=torch.tensor(
            layout.anchor_positions, dtype=torch.int64, **tensor_kwargs
        ).unsqueeze(0),
        absolute_query_positions=torch.tensor(
            layout.absolute_query_positions, dtype=torch.int64, **tensor_kwargs
        ).unsqueeze(0),
        context_positions=torch.arange(context_count, dtype=torch.int64, **tensor_kwargs),
        dense_visibility=torch.empty(
            (query_count, context_count + query_count), dtype=torch.bool, **tensor_kwargs
        ),
        mask_token_id=layout.mask_token_id,
    )
    materialized = replace(batch, dense_visibility=dense_visibility_oracle(batch))
    materialized.validate()
    return materialized


def make_flex_attention_visibility_predicate(
    batch: TorchSparseTrainingBatch,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return FlexAttention's ``mask_mod(batch, head, q_idx, kv_idx)`` rule.

    The predicate only describes visibility.  It does not construct query/KV
    states, inject teacher features, or validate any attention kernel.
    """
    batch.validate()
    if batch.num_queries == 0:
        raise ValueError("FlexAttention predicate requires at least one query")

    context_count = batch.context_length
    block_ids = batch.block_ids[0]
    anchors = batch.anchor_positions[0]
    last_query_index = batch.num_queries - 1

    def visibility_predicate(
        _batch_index: torch.Tensor,
        _head_index: torch.Tensor,
        query_index: torch.Tensor,
        key_index: torch.Tensor,
    ) -> torch.Tensor:
        is_context = key_index < context_count
        # Clamp before indexing so torch.where evaluates safely for context keys.
        query_key_index = (key_index - context_count).clamp(min=0, max=last_query_index)
        same_block = block_ids[query_index] == block_ids[query_key_index]
        visible_context = key_index <= anchors[query_index]
        return torch.where(is_context, visible_context, same_block)

    return visibility_predicate


def flex_attention_block_mask_supported() -> bool:
    """Whether this PyTorch build exposes FlexAttention block-mask construction."""
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except (ImportError, AttributeError):
        return False
    return callable(create_block_mask)


def build_flex_attention_block_mask(
    batch: TorchSparseTrainingBatch,
    *,
    block_size: int = 128,
):
    """Build a FlexAttention ``BlockMask`` from the audited predicate.

    This is construction-only compatibility.  It neither invokes
    ``flex_attention`` nor establishes CPU/GPU kernel availability or
    correctness.
    """
    if not flex_attention_block_mask_supported():
        raise RuntimeError("this PyTorch build does not expose FlexAttention create_block_mask")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    from torch.nn.attention.flex_attention import create_block_mask

    return create_block_mask(
        make_flex_attention_visibility_predicate(batch),
        B=1,
        H=1,
        Q_LEN=batch.num_queries,
        KV_LEN=batch.key_length,
        device=batch.input_ids.device,
        BLOCK_SIZE=block_size,
    )
