"""Transactional bounded handoff from one teacher artifact to one optimizer step.

This module joins the existing request-preserving feature ring, deterministic
sampling/layout code, frozen target vocabulary weights, and a real draft adapter.
It never starts a teacher server, writes a feature corpus, or saves a checkpoint.
The ring head is acknowledged only after ``optimizer.step()`` succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional runtime dependency.
    raise ImportError(
        "Online optimization requires PyTorch; install "
        "north-dflash-training-scaffold[runtime]."
    ) from exc

from .feature_stream import BoundedFeatureRing, StreamedTeacherBatch
from .layout import build_training_batch_layout
from .sampling import sample_anchor_blocks
from .schema import ResponseExample
from .torch_layout import TorchSparseTrainingBatch, build_torch_training_batch
from .training import DFlashTrainingStep, TeacherFeatureBundle


@dataclass(frozen=True)
class PreparedStreamedTrainingStep:
    """One ring item materialized as packed draft queries and clean features."""

    request_id: str
    source_token_count: int
    prompt_length: int
    packed_batch: TorchSparseTrainingBatch
    teacher_features: TeacherFeatureBundle


@dataclass(frozen=True)
class BoundedOptimizerStepResult:
    """Auditable scalar result; it deliberately contains no model state."""

    request_id: str
    source_token_count: int
    context_length: int
    query_count: int
    active_label_count: int
    loss: float
    gradient_norm: float
    gradients_with_nonzero_values: int
    updated_parameter_tensors: int
    feature_bytes_released: int


def prepare_streamed_training_step(
    streamed: StreamedTeacherBatch,
    *,
    prompt_length: int,
    block_size: int,
    max_anchors: int,
    mask_token_id: int,
    seed: int,
    loss_gamma: float | None,
    device: torch.device,
    feature_dtype: torch.dtype,
) -> PreparedStreamedTrainingStep:
    """Align one exact clean token stream with deterministic packed queries.

    The connector artifact covers the complete clean ``prompt + response``
    sequence. Only clean states strictly before the largest sampled absolute
    anchor are retained, matching runtime DFlash: the anchor token has been
    sampled but has not yet passed through the target when the draft proposes.
    """
    streamed.validate()
    token_count = streamed.token_count
    if (
        isinstance(prompt_length, bool)
        or not isinstance(prompt_length, int)
        or prompt_length < 0
        or prompt_length >= token_count
    ):
        raise ValueError("prompt_length must leave a non-empty response")
    if not torch.empty((), dtype=feature_dtype).is_floating_point():
        raise ValueError("feature_dtype must be floating point")

    clean_tokens = tuple(int(token) for token in streamed.token_ids[0].tolist())
    example = ResponseExample(
        prompt_tokens=clean_tokens[:prompt_length],
        response_tokens=clean_tokens[prompt_length:],
        metadata={"request_id": streamed.request_id},
    )
    sampled = sample_anchor_blocks(
        example,
        block_size=block_size,
        max_anchors=max_anchors,
        mask_token_id=mask_token_id,
        seed=seed,
    )
    if not sampled.blocks:
        raise ValueError("response is too short to produce a full sampled block")
    packed = build_torch_training_batch(
        build_training_batch_layout(sampled, gamma=loss_gamma),
        device=device,
    )
    context_length = packed.context_length
    if context_length > streamed.features.context_length:
        raise ValueError("packed context exceeds the clean teacher feature stream")

    states = tuple(
        state[:, :context_length, :]
        .to(device=device, dtype=feature_dtype)
        .clone()
        .detach()
        for state in streamed.features.hidden_states
    )
    features = TeacherFeatureBundle(
        selected_layer_ids=streamed.features.selected_layer_ids,
        hidden_states=states,
        clean_positions=torch.arange(context_length, dtype=torch.int64, device=device),
    )
    features.validate(expected_layer_ids=streamed.runtime_identity.selected_layer_ids)
    return PreparedStreamedTrainingStep(
        request_id=streamed.request_id,
        source_token_count=token_count,
        prompt_length=prompt_length,
        packed_batch=packed,
        teacher_features=features,
    )


def _unique_parameters(modules) -> tuple[torch.nn.Parameter, ...]:
    seen: set[int] = set()
    result = []
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return tuple(result)


def run_one_bounded_optimizer_step(
    *,
    ring: BoundedFeatureRing,
    training_step: DFlashTrainingStep,
    optimizer: torch.optim.Optimizer,
    prompt_length: int,
    block_size: int,
    max_anchors: int,
    mask_token_id: int,
    seed: int,
    loss_gamma: float | None = None,
    max_gradient_norm: float = 1.0,
) -> BoundedOptimizerStepResult:
    """Consume exactly one ring head after one successful draft-only update."""
    if not math.isfinite(max_gradient_norm) or max_gradient_norm <= 0:
        raise ValueError("max_gradient_norm must be finite and positive")
    streamed = ring.peekleft()
    if streamed.runtime_identity != ring.runtime_identity:
        raise ValueError("ring head runtime identity drifted")
    if training_step.selected_layer_ids != ring.runtime_identity.selected_layer_ids:
        raise ValueError("training step layer order does not match the teacher runtime")
    shared_mask_id = training_step.shared_weights.mask_token_id
    if shared_mask_id is not None and shared_mask_id != mask_token_id:
        raise ValueError("mask token does not match the frozen shared embedding handoff")

    trainable = tuple(parameter for parameter in training_step.parameters() if parameter.requires_grad)
    if not trainable:
        raise ValueError("draft training step has no trainable parameters")
    optimizer_parameters = tuple(
        parameter for group in optimizer.param_groups for parameter in group["params"]
    )
    if len({id(parameter) for parameter in optimizer_parameters}) != len(optimizer_parameters):
        raise ValueError("optimizer contains duplicate parameter references")
    if {id(parameter) for parameter in optimizer_parameters} != {
        id(parameter) for parameter in trainable
    }:
        raise ValueError("optimizer must contain exactly the draft trainable parameters")

    embedding_weight = getattr(training_step.shared_weights.embedding, "weight", None)
    if not isinstance(embedding_weight, torch.Tensor) or not embedding_weight.is_floating_point():
        raise ValueError("frozen shared embedding must expose a floating weight")
    device = trainable[0].device
    if any(parameter.device != device for parameter in trainable):
        raise ValueError("all draft parameters must share one device")
    if embedding_weight.device != device:
        raise ValueError("draft and frozen shared embedding must share a device")

    frozen = _unique_parameters(
        (training_step.shared_weights.embedding, training_step.shared_weights.lm_head)
    )
    if any(parameter.requires_grad or parameter.grad is not None for parameter in frozen):
        raise ValueError("shared vocabulary parameters must be frozen and gradient-free")

    prepared = prepare_streamed_training_step(
        streamed,
        prompt_length=prompt_length,
        block_size=block_size,
        max_anchors=max_anchors,
        mask_token_id=mask_token_id,
        seed=seed,
        loss_gamma=loss_gamma,
        device=device,
        feature_dtype=embedding_weight.dtype,
    )
    versions_before = tuple(parameter._version for parameter in trainable)
    optimizer.zero_grad(set_to_none=True)
    output = training_step(prepared.packed_batch, prepared.teacher_features)
    if not torch.isfinite(output.loss).item():
        raise FloatingPointError("draft loss is not finite")
    output.loss.backward()

    gradients = tuple(parameter.grad for parameter in trainable if parameter.grad is not None)
    if not gradients:
        raise RuntimeError("draft backward produced no gradients")
    if any(not torch.isfinite(gradient).all().item() for gradient in gradients):
        raise FloatingPointError("draft gradients are not finite")
    nonzero_gradients = sum(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
    if nonzero_gradients == 0:
        raise RuntimeError("draft backward produced only zero gradients")
    gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(trainable, max_gradient_norm)
    gradient_norm = float(gradient_norm_tensor.detach().cpu())
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("draft gradient norm is not finite")

    optimizer.step()
    updated = sum(
        parameter._version > version
        for parameter, version in zip(trainable, versions_before, strict=True)
    )
    if updated == 0:
        raise RuntimeError("optimizer did not update any draft parameter tensor")
    if any(parameter.grad is not None for parameter in frozen):
        raise RuntimeError("shared vocabulary weights received gradients")

    acknowledged = ring.ackleft(prepared.request_id)
    if acknowledged is not streamed:  # pragma: no cover - guarded by ring semantics.
        raise RuntimeError("ring acknowledgement removed the wrong request")
    return BoundedOptimizerStepResult(
        request_id=prepared.request_id,
        source_token_count=prepared.source_token_count,
        context_length=prepared.packed_batch.context_length,
        query_count=prepared.packed_batch.num_queries,
        active_label_count=int(prepared.packed_batch.loss_mask.sum().item()),
        loss=float(output.loss.detach().cpu()),
        gradient_norm=gradient_norm,
        gradients_with_nonzero_values=nonzero_gradients,
        updated_parameter_tensors=updated,
        feature_bytes_released=streamed.feature_bytes,
    )
