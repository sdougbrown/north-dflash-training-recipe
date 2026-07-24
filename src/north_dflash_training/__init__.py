"""Testable, architecture-independent pieces of the North DFlash recipe.

This package deliberately does not load a model, import transformers, or start
an inference server. It is a scaffold, not a training implementation.
"""

from .cache import FeatureCacheEstimate, estimate_feature_cache
from .candidate import derive_north_candidate
from .sampling import AnchorBlock, SampledBlocks, sample_anchor_blocks
from .schema import ResponseExample
from .weights import default_gamma, exponential_loss_weights

__all__ = [
    "AnchorBlock",
    "FeatureCacheEstimate",
    "ResponseExample",
    "SampledBlocks",
    "default_gamma",
    "derive_north_candidate",
    "estimate_feature_cache",
    "exponential_loss_weights",
    "sample_anchor_blocks",
]
