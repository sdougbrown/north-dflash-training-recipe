"""Paper-aligned position-dependent cross-entropy weights."""

from __future__ import annotations

import math

DEFAULT_GAMMA = {16: 7.0, 10: 5.0, 8: 4.0}


def default_gamma(block_size: int) -> float:
    """Return the DFlash A.1 default, or fail rather than guessing."""
    try:
        return DEFAULT_GAMMA[block_size]
    except KeyError as exc:
        raise ValueError(
            f"no paper default gamma for block_size={block_size}; pass gamma explicitly"
        ) from exc


def exponential_loss_weights(
    block_size: int,
    gamma: float | None = None,
    *,
    include_anchor: bool = False,
) -> tuple[float, ...]:
    """Compute ``exp(-(k-1)/gamma)`` for predicted positions in a block.

    DFlash predicts ``block_size - 1`` future positions, so by default the
    returned vector is indexed by k=1..block_size-1 and starts at 1.0. Set
    ``include_anchor=True`` to prepend an ignored anchor bookkeeping weight;
    the first predicted future still starts at 1.0.
    """
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    resolved_gamma = default_gamma(block_size) if gamma is None else gamma
    if resolved_gamma <= 0 or not math.isfinite(resolved_gamma):
        raise ValueError("gamma must be finite and greater than zero")
    future_weights = tuple(
        math.exp(-position / resolved_gamma) for position in range(block_size - 1)
    )
    return (1.0,) + future_weights if include_anchor else future_weights
