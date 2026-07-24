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
    ``include_anchor=True`` to get a length-``block_size`` vector for tensor
    bookkeeping; its first value is also 1.0, although the anchor is not a
    prediction target.
    """
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    resolved_gamma = default_gamma(block_size) if gamma is None else gamma
    if resolved_gamma <= 0 or not math.isfinite(resolved_gamma):
        raise ValueError("gamma must be finite and greater than zero")
    count = block_size if include_anchor else block_size - 1
    return tuple(math.exp(-position / resolved_gamma) for position in range(count))
