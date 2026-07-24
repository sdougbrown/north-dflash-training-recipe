"""Eager-attention adapter for the locally inspected ``DFlashDraftModel``.

This optional module is intentionally isolated from the dependency-free base
package.  It adapts the committed training contract to the reference model,
not to North: the reference model projects ``target_hidden [B, C, L*H]`` and
uses its own Qwen3 DFlash attention implementation.

The local reference's ``apply_rotary_pos_emb`` consumes RoPE cos/sin for every
key and takes the final ``Q`` entries for query rotation.  Consequently this
adapter supplies ``[0, ..., C-1]`` for clean context keys followed by the
packed queries' absolute positions.  Transformers 4.57.1's
``eager_attention_forward`` adds a four-dimensional additive mask to attention
scores, so the audited boolean visibility oracle is represented as
``[B, 1, Q, C + Q]`` with zero for visible keys and ``finfo(dtype).min`` for
blocked keys.  A boolean mask must not be passed to eager attention: it would
be added as 0/1 rather than acting as a visibility mask.

Caching, FlexAttention, and non-eager attention implementations are outside
this adapter's evidence boundary.
"""

from __future__ import annotations

import importlib.util

from .training import DraftAdapterModule

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency boundary.
    raise ImportError(
        "TransformersDFlashDraftAdapter requires PyTorch; install "
        "north-dflash-training-scaffold[torch]."
    ) from exc


def transformers_dflash_available() -> bool:
    """Whether both optional Transformers and the reference dflash package resolve."""
    return importlib.util.find_spec("transformers") is not None and importlib.util.find_spec("dflash") is not None


def dense_visibility_to_eager_attention_mask(
    dense_visibility: torch.Tensor,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Convert the audited ``[Q, C + Q]`` boolean oracle to eager Qwen3 mask.

    The reference Qwen3 eager function slices and *adds* ``attention_mask`` to
    scores shaped ``[B, heads, Q, C + Q]``.  This returns the broadcastable
    additive shape ``[B, 1, Q, C + Q]``.  The same visibility rule applies to
    every batch item; batched per-example rules are deliberately not inferred.
    """
    if dense_visibility.ndim != 2 or dense_visibility.dtype != torch.bool:
        raise ValueError("dense_visibility must be a boolean [Q, C + Q] tensor")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("eager attention mask dtype must be floating point")
    if dense_visibility.device != device:
        raise ValueError("dense_visibility and model inputs must share a device")
    visible = dense_visibility.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    return torch.zeros(visible.shape, dtype=dtype, device=device).masked_fill(
        ~visible, torch.finfo(dtype).min
    )


def full_dflash_position_ids(
    query_position_ids: torch.Tensor,
    *,
    context_length: int,
) -> torch.Tensor:
    """Join clean context positions ``0..C-1`` and packed query positions."""
    if query_position_ids.ndim != 2 or query_position_ids.dtype != torch.int64:
        raise ValueError("position_ids must be an int64 [B, Q] tensor")
    if context_length < 1:
        raise ValueError("target context must contain at least one clean position")
    context = torch.arange(
        context_length, dtype=torch.int64, device=query_position_ids.device
    ).unsqueeze(0).expand(query_position_ids.shape[0], -1)
    return torch.cat((context, query_position_ids), dim=1)


class TransformersDFlashDraftAdapter(DraftAdapterModule):
    """Run only a real local ``dflash.model.DFlashDraftModel`` with eager attention.

    The adapter does not prepare a cache or reinterpret the dense rule as a
    FlexAttention block mask.  It is appropriate only for the committed
    no-cache CPU training forward.
    """

    def __init__(self, draft_model) -> None:
        super().__init__()
        if not transformers_dflash_available():
            raise RuntimeError(
                "DFlash Transformers adapter is unavailable; install compatible Transformers and expose dflash."
            )
        from dflash.model import DFlashDraftModel

        if not isinstance(draft_model, DFlashDraftModel):
            raise TypeError("draft_model must be dflash.model.DFlashDraftModel")
        if getattr(draft_model.config, "_attn_implementation", None) != "eager":
            raise ValueError(
                "DFlash adapter supports only config._attn_implementation == 'eager'; "
                "other Transformers attention backends have not been audited."
            )
        self.draft_model = draft_model

    @classmethod
    def from_reference_model(cls, draft_model) -> "TransformersDFlashDraftAdapter":
        """Validate that ``draft_model`` is the installed reference model."""
        return cls(draft_model)

    def forward(self, *, target_features, noise_embeddings, position_ids, dense_visibility):
        if target_features.ndim != 3 or noise_embeddings.ndim != 3:
            raise ValueError("target_features and noise_embeddings must be rank-three")
        batch_size, context_length, target_width = target_features.shape
        noise_batch, query_length, hidden_size = noise_embeddings.shape
        if batch_size != noise_batch:
            raise ValueError("target_features and noise_embeddings must share batch size")
        if context_length < 1 or query_length < 1:
            raise ValueError("target and query lengths must be positive")
        if not target_features.is_floating_point() or not noise_embeddings.is_floating_point():
            raise ValueError("target_features and noise_embeddings must be floating point")
        if target_features.device != noise_embeddings.device:
            raise ValueError("target_features and noise_embeddings must share a device")
        if target_features.dtype != noise_embeddings.dtype:
            raise ValueError("target_features and noise_embeddings must share a dtype")
        if position_ids.shape != (batch_size, query_length) or position_ids.dtype != torch.int64:
            raise ValueError("position_ids must be int64 [B, Q]")
        if position_ids.device != noise_embeddings.device:
            raise ValueError("position_ids and model inputs must share a device")
        if dense_visibility.shape != (query_length, context_length + query_length):
            raise ValueError("dense_visibility must be [Q, C + Q]")
        if dense_visibility.dtype != torch.bool:
            raise ValueError("dense_visibility must be boolean")
        if dense_visibility.device != noise_embeddings.device:
            raise ValueError("dense_visibility and model inputs must share a device")
        if hidden_size != self.draft_model.config.hidden_size:
            raise ValueError("noise embedding width does not match the DFlash hidden size")
        if target_width != self.draft_model.fc.in_features:
            raise ValueError("target feature width does not match DFlash fc.in_features")

        full_position_ids = full_dflash_position_ids(position_ids, context_length=context_length)
        attention_mask = dense_visibility_to_eager_attention_mask(
            dense_visibility,
            batch_size=batch_size,
            dtype=noise_embeddings.dtype,
            device=noise_embeddings.device,
        )
        hidden_states = self.draft_model(
            target_hidden=target_features,
            noise_embedding=noise_embeddings,
            position_ids=full_position_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.shape != noise_embeddings.shape:
            raise RuntimeError("reference DFlash forward did not return [B, Q, H] hidden states")
        return hidden_states
