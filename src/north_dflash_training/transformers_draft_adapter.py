"""Capability-gated skeleton for the local ``dflash.DFlashDraftModel`` API.

This is intentionally separate from :mod:`north_dflash_training.training`.
The base package never imports PyTorch or Transformers, and this adapter is not
an executable North integration.  It only records the locally inspected Qwen
reference call shape: ``target_hidden``, ``noise_embedding``, and
``position_ids``.
"""

from __future__ import annotations

import importlib.util

from .training import DraftAdapterModule


def transformers_dflash_available() -> bool:
    """Whether both optional Transformers and the reference dflash package resolve."""
    return importlib.util.find_spec("transformers") is not None and importlib.util.find_spec("dflash") is not None


class TransformersDFlashDraftAdapter(DraftAdapterModule):
    """Non-running boundary around a locally supplied Qwen ``DFlashDraftModel``.

    The reference model's forward signature can consume concatenated target
    features and shared noise embeddings, but its architecture-specific rotary
    positions, cache behavior, and attention-mask representation have not been
    reconciled with North's committed ``[Q, C + Q]`` visibility contract.  Its
    forward is therefore deliberately blocked rather than silently converting a
    mask or claiming a working Transformers integration.
    """

    def __init__(self, draft_model) -> None:
        super().__init__()
        self.draft_model = draft_model

    @classmethod
    def from_reference_model(cls, draft_model) -> "TransformersDFlashDraftAdapter":
        if not transformers_dflash_available():
            raise RuntimeError(
                "DFlash Transformers adapter is unavailable; install compatible Transformers and expose dflash."
            )
        # Delayed import keeps the optional dependency boundary intact.  The
        # class check is intentionally local-version-specific, matching model.py.
        from dflash.model import DFlashDraftModel

        if not isinstance(draft_model, DFlashDraftModel):
            raise TypeError("draft_model must be dflash.model.DFlashDraftModel")
        return cls(draft_model)

    def forward(self, *, target_features, noise_embeddings, position_ids, dense_visibility):
        raise NotImplementedError(
            "No Transformers DFlash forward is enabled: validate North hidden-state extraction, "
            "target/KV positional alignment, and dense visibility-to-attention-mask semantics first."
        )
