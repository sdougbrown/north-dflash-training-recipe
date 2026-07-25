"""Testable, architecture-independent pieces of the North DFlash recipe.

This package deliberately does not load a model, import transformers, or start
an inference server. It is a scaffold, not a training implementation.
"""

from importlib import import_module

from .cache import FeatureCacheEstimate, estimate_feature_cache
from .checkpoint_identity import (
    CheckpointFileDigest,
    CheckpointIdentityManifest,
    build_checkpoint_identity_manifest,
    verify_checkpoint_identity,
)
from .candidate import audit_mask_token, derive_north_candidate
from .layout import SparseTrainingBatchLayout, build_training_batch_layout, concatenate_sampled_blocks
from .sampling import AnchorBlock, SampledBlocks, sample_anchor_blocks

_SAVE_RESUME_EXPORTS = frozenset(
    {
        "CheckpointManifest",
        "SavedRequestLedger",
        "TiedVocabIdentity",
        "resume_checkpoint",
        "save_checkpoint",
        "verify_checkpoint_directory",
    }
)


def __getattr__(name: str):
    """Lazily expose optional PyTorch/safetensors checkpoint APIs."""
    if name in _SAVE_RESUME_EXPORTS:
        return getattr(import_module(".save_resume", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .schema import ResponseExample
from .teacher import AutoGPTQIdentity, TeacherFeatureManifest, teacher_feature_manifest_from_config
from .weights import default_gamma, exponential_loss_weights

__all__ = [
    "AnchorBlock",
    "CheckpointFileDigest",
    "CheckpointIdentityManifest",
    "CheckpointManifest",
    "AutoGPTQIdentity",
    "FeatureCacheEstimate",
    "SparseTrainingBatchLayout",
    "TeacherFeatureManifest",
    "ResponseExample",
    "SampledBlocks",
    "SavedRequestLedger",
    "TiedVocabIdentity",
    "audit_mask_token",
    "build_checkpoint_identity_manifest",
    "build_training_batch_layout",
    "concatenate_sampled_blocks",
    "default_gamma",
    "derive_north_candidate",
    "estimate_feature_cache",
    "exponential_loss_weights",
    "sample_anchor_blocks",
    "save_checkpoint",
    "resume_checkpoint",
    "teacher_feature_manifest_from_config",
    "verify_checkpoint_directory",
    "verify_checkpoint_identity",
]
