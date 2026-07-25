"""Dependency-free CPU representation of sampled DFlash batch layout.

This module deliberately describes visibility as Python tuples rather than a
framework mask.  It is suitable for auditing and CPU tests, but is not a
FlexAttention or GPU implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .sampling import SampledBlocks
from .schema import IGNORE_INDEX
from .weights import default_gamma, exponential_loss_weights


@dataclass(frozen=True)
class SparseTrainingBatchLayout:
    """Concatenated query blocks and their audited dependencies.

    The arrays are flattened in ascending sampled-anchor order.  ``anchor_positions``
    is repeated for each query and is the absolute clean position whose frozen
    target feature may be injected for that query. ``target_context_positions``
    contains every clean target position from the beginning of the sequence
    strictly before that anchor, never the anchor or a future target position.

    ``query_visibility`` is a CPU-only relation over flattened query indices:
    queries in the same block see one another in both directions and queries in
    different blocks do not see one another.  It is intentionally not a tensor
    or a FlexAttention mask.
    """

    query_tokens: tuple[int, ...]
    labels: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    block_ids: tuple[int, ...]
    anchor_positions: tuple[int, ...]
    absolute_query_positions: tuple[int, ...]
    position_weights: tuple[float, ...]
    loss_gamma: float
    query_visibility: tuple[tuple[bool, ...], ...]
    target_context_positions: tuple[tuple[int, ...], ...]
    block_anchor_positions: tuple[int, ...]
    block_size: int
    mask_token_id: int

    @property
    def input_tokens(self) -> tuple[int, ...]:
        """Compatibility name for the flattened query input IDs."""
        return self.query_tokens

    @property
    def loss_weights(self) -> tuple[float, ...]:
        return self.position_weights

    @property
    def target_context_anchor_positions(self) -> tuple[tuple[int, ...], ...]:
        return self.target_context_positions

    @property
    def num_queries(self) -> int:
        return len(self.query_tokens)

    @property
    def num_blocks(self) -> int:
        return len(self.block_anchor_positions)

    def can_query_see_query(self, query_index: int, key_index: int) -> bool:
        return self.query_visibility[query_index][key_index]

    def target_context_for_query(self, query_index: int) -> tuple[int, ...]:
        return self.target_context_positions[query_index]

    def validate(self) -> None:
        """Raise ``ValueError`` if the layout violates its dependency contract."""
        arrays = (
            self.labels,
            self.loss_mask,
            self.block_ids,
            self.anchor_positions,
            self.absolute_query_positions,
            self.position_weights,
            self.target_context_positions,
        )
        if any(len(values) != self.num_queries for values in arrays):
            raise ValueError("all flattened layout arrays must have equal length")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        if self.num_queries != self.num_blocks * self.block_size:
            raise ValueError("each sampled block must contribute block_size queries")
        expected_ids = tuple(index // self.block_size for index in range(self.num_queries))
        if self.block_ids != expected_ids:
            raise ValueError("block_ids must be contiguous and deterministic")
        if tuple(self.block_anchor_positions) != tuple(
            self.anchor_positions[index * self.block_size]
            for index in range(self.num_blocks)
        ):
            raise ValueError("block anchor positions do not match flattened anchors")
        if tuple(self.block_anchor_positions) != tuple(sorted(self.block_anchor_positions)):
            raise ValueError("sampled blocks must be ordered by ascending anchor position")
        if len(set(self.block_anchor_positions)) != self.num_blocks:
            raise ValueError("sampled blocks must have distinct anchor positions")

        expected_visibility = tuple(
            tuple(self.block_ids[row] == self.block_ids[column] for column in range(self.num_queries))
            for row in range(self.num_queries)
        )
        if self.query_visibility != expected_visibility:
            raise ValueError("query visibility permits inter-block leakage or is not bidirectional")

        if not math.isfinite(self.loss_gamma) or self.loss_gamma <= 0:
            raise ValueError("loss_gamma must be finite and greater than zero")
        expected_weights = exponential_loss_weights(
            self.block_size, gamma=self.loss_gamma, include_anchor=True
        )
        for query_index, block_id in enumerate(self.block_ids):
            offset = query_index % self.block_size
            anchor = self.anchor_positions[query_index]
            if self.absolute_query_positions[query_index] != anchor + offset:
                raise ValueError("absolute query positions do not follow each block anchor")
            if self.target_context_positions[query_index] != tuple(range(anchor)):
                raise ValueError("target context must contain clean positions strictly before the block anchor")
            if not math.isclose(self.position_weights[query_index], expected_weights[offset]):
                raise ValueError("position weights do not follow the DFlash loss-decay convention")
            if offset == 0:
                if self.loss_mask[query_index] or self.labels[query_index] != IGNORE_INDEX:
                    raise ValueError("clean anchors must be ignored by the loss")
            elif not self.loss_mask[query_index] or self.labels[query_index] == IGNORE_INDEX:
                raise ValueError("masked futures must have non-ignored loss labels")
            if block_id < 0 or block_id >= self.num_blocks:
                raise ValueError("invalid block ID")


def build_training_batch_layout(
    sampled: SampledBlocks,
    *,
    gamma: float | None = None,
) -> SparseTrainingBatchLayout:
    """Concatenate sampled blocks and materialize their CPU-auditable relation."""
    if tuple(block.anchor_position for block in sampled.blocks) != tuple(
        sorted(block.anchor_position for block in sampled.blocks)
    ):
        raise ValueError("sampled blocks must be ordered by ascending anchor position")

    resolved_gamma = default_gamma(sampled.block_size) if gamma is None else gamma
    weights = exponential_loss_weights(sampled.block_size, gamma=resolved_gamma, include_anchor=True)
    query_tokens = tuple(token for block in sampled.blocks for token in block.input_tokens)
    labels = tuple(label for block in sampled.blocks for label in block.labels)
    loss_mask = tuple(value for block in sampled.blocks for value in block.loss_mask)
    block_ids = tuple(
        block_id
        for block_id, _block in enumerate(sampled.blocks)
        for _ in range(sampled.block_size)
    )
    block_anchor_positions = tuple(block.absolute_anchor_position for block in sampled.blocks)
    anchor_positions = tuple(
        anchor
        for anchor in block_anchor_positions
        for _ in range(sampled.block_size)
    )
    absolute_query_positions = tuple(
        anchor + offset
        for anchor in block_anchor_positions
        for offset in range(sampled.block_size)
    )
    position_weights = tuple(weight for _ in sampled.blocks for weight in weights)
    target_context_positions = tuple(tuple(range(anchor)) for anchor in anchor_positions)
    query_visibility = tuple(
        tuple(block_ids[row] == block_ids[column] for column in range(len(block_ids)))
        for row in range(len(block_ids))
    )

    layout = SparseTrainingBatchLayout(
        query_tokens=query_tokens,
        labels=labels,
        loss_mask=loss_mask,
        block_ids=block_ids,
        anchor_positions=anchor_positions,
        absolute_query_positions=absolute_query_positions,
        position_weights=position_weights,
        loss_gamma=resolved_gamma,
        query_visibility=query_visibility,
        target_context_positions=target_context_positions,
        block_anchor_positions=block_anchor_positions,
        block_size=sampled.block_size,
        mask_token_id=sampled.mask_token_id,
    )
    layout.validate()
    return layout


# This name emphasizes that no model-specific batching is involved.
concatenate_sampled_blocks = build_training_batch_layout
