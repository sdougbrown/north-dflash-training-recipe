"""Deterministic random anchor/block construction from DFlash 4.2/A.1."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterator

from .schema import IGNORE_INDEX, ResponseExample


@dataclass(frozen=True)
class AnchorBlock:
    """One clean-anchor plus masked-future training block.

    Positions are response-relative unless named ``absolute_*``. The first
    input token is clean; every future input token is ``mask_token_id``. The
    anchor is ignored by the loss and the remaining labels are the clean
    response futures.
    """

    anchor_position: int
    absolute_anchor_position: int
    input_tokens: tuple[int, ...]
    labels: tuple[int, ...]
    loss_mask: tuple[bool, ...]

    @property
    def future_positions(self) -> tuple[int, ...]:
        return tuple(range(self.anchor_position + 1, self.anchor_position + len(self.input_tokens)))


@dataclass(frozen=True)
class SampledBlocks:
    blocks: tuple[AnchorBlock, ...]
    eligible_anchor_positions: tuple[int, ...]
    seed: int
    block_size: int
    mask_token_id: int

    @property
    def anchor_positions(self) -> tuple[int, ...]:
        return tuple(block.anchor_position for block in self.blocks)

    def __iter__(self) -> Iterator[AnchorBlock]:
        return iter(self.blocks)


def sample_anchor_blocks(
    example: ResponseExample,
    *,
    block_size: int = 16,
    max_anchors: int = 512,
    mask_token_id: int,
    seed: int = 0,
) -> SampledBlocks:
    """Sample up to ``max_anchors`` distinct full blocks from a response.

    This follows the paper's training construction: an anchor is a clean
    response token and positions ``anchor + 1 .. anchor + block_size - 1`` are
    masked and predicted in parallel. Tail positions that cannot form a full
    block are not sampled. A local RNG makes repeated calls independent of
    process-global random state; sorted sampled positions make concatenation
    order stable while retaining random coverage.
    """

    if block_size < 2:
        raise ValueError("block_size must be at least 2 (one anchor plus a future)")
    if max_anchors < 0:
        raise ValueError("max_anchors must be non-negative")
    if isinstance(mask_token_id, bool) or not isinstance(mask_token_id, int) or mask_token_id < 0:
        raise ValueError("mask_token_id must be a non-negative integer")

    response_length = len(example.response_tokens)
    eligible = tuple(range(0, max(0, response_length - block_size + 1)))
    count = min(max_anchors, len(eligible))
    rng = random.Random(seed)
    anchors = tuple(sorted(rng.sample(eligible, count)))

    blocks = tuple(
        AnchorBlock(
            anchor_position=anchor,
            absolute_anchor_position=len(example.prompt_tokens) + anchor,
            input_tokens=(example.response_tokens[anchor],) + (mask_token_id,) * (block_size - 1),
            labels=(IGNORE_INDEX,) + tuple(example.response_tokens[anchor + 1 : anchor + block_size]),
            loss_mask=(False,) + (True,) * (block_size - 1),
        )
        for anchor in anchors
    )
    return SampledBlocks(
        blocks=blocks,
        eligible_anchor_positions=eligible,
        seed=seed,
        block_size=block_size,
        mask_token_id=mask_token_id,
    )
