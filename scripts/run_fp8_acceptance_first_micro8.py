#!/usr/bin/env python3
"""Run the first retained eight-example FP8 DFlash micro-pilot.

This script is intentionally target-specific. It records exact vLLM-returned
prompt/response token IDs, keeps at most eight five-feature requests in memory,
stops at a marker before touching CUDA training state, performs eight updates,
releases every connector handoff, and publishes one draft-only checkpoint.
It is not a quality, acceptance, geometry-selection, or deployment result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import torch
import yaml
from dflash.model import DFlashDraftModel
from safetensors import safe_open
from torch import nn
from transformers import Qwen3Config

from north_dflash_training.connector_lifecycle import (
    OwnedConnectorHandoff,
    ingest_connector_handoff,
    release_connector_after_optimizer_step,
)
from north_dflash_training.feature_stream import BoundedFeatureRing, TeacherRuntimeIdentity
from north_dflash_training.online_step import run_one_bounded_optimizer_step
from north_dflash_training.save_resume import (
    SavedRequestLedger,
    resume_checkpoint,
    save_checkpoint,
    verify_checkpoint_directory,
)
from north_dflash_training.schema import ResponseExample
from north_dflash_training.training import DFlashTrainingStep, FrozenSharedWeights
from north_dflash_training.transformers_draft_adapter import TransformersDFlashDraftAdapter

SOURCE_SHA256 = "57a329c953b698c6e5c68b7ae84198801984d572436066a39c648d7548ec7dd4"
CHECKPOINT_MANIFEST_SHA256 = "35812fdf32f497a558f31bbea43e7d69f8c1cd43c66530c7499de2f293ae2bb6"
RUNTIME_IMAGE_ID = "sha256:d89ae2666c80ea7d64c903670cd5b8a643f03e878c65aa21b534bc6792dba637"
SHARED_EMBEDDING_SHA256 = "d9497bf06b9a4efa9463207860a7fca3e7c051f4a857ca010ec4863086586d19"
TARGET_BLOCKS = (1, 12, 24, 35, 46)
EXTRACTOR_ENTRIES = (2, 13, 25, 36, 47)
CASE_IDS = (
    "cruxeval-o: sample_6",
    "cruxeval-o: sample_25",
    "cruxeval-o: sample_27",
    "cruxeval-o: sample_30",
    "cruxeval-o: sample_32",
    "cruxeval-o: sample_89",
    "cruxeval-o: sample_94",
    "cruxeval-o: sample_95",
)
PROMPT = """You are given a Python function. Determine what it returns when called with the provided input.

```python
{code}
```

What does `{input}` return?

Think through the function carefully. Explain the important execution steps, then end with a line of the form `Final answer: <Python literal>`.
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().to(device="cpu").contiguous().view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def post_json(url: str, payload: dict, *, timeout: float = 900) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"request returned HTTP {response.status}")
        return json.load(response)


def load_cases(source: Path) -> list[dict]:
    if sha256_file(source) != SOURCE_SHA256:
        raise RuntimeError("local CRUXEval source hash mismatch")
    raw = yaml.safe_load(source.read_text())
    by_id = {item["description"]: item for item in raw}
    if any(case_id not in by_id for case_id in CASE_IDS):
        raise RuntimeError("configured CRUXEval case is missing")
    return [by_id[case_id] for case_id in CASE_IDS]


def wait_for_stable_artifact(path: Path, *, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    previous = None
    stable = 0
    while time.monotonic() < deadline:
        if path.is_file():
            metadata = path.stat()
            current = (metadata.st_size, metadata.st_mtime_ns)
            if current == previous and metadata.st_size > 0:
                stable += 1
                if stable >= 3:
                    return
            else:
                previous = current
                stable = 0
        time.sleep(0.1)
    raise TimeoutError(f"connector artifact did not stabilize: {path}")


def require_teacher_stopped(marker: Path, server: str) -> None:
    deadline = time.monotonic() + 900
    while not marker.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("teacher shutdown marker was not received")
        time.sleep(1)
    try:
        urlopen(f"{server}/health", timeout=2)
    except URLError:
        return
    raise RuntimeError("teacher server remains reachable after shutdown marker")


def load_exact_embedding(model_root: Path, device: torch.device) -> tuple[nn.Embedding, str]:
    shard = model_root / "model-00001-of-00007.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor("model.embed_tokens.weight")
    if weight.shape != (262144, 2048) or weight.dtype != torch.bfloat16:
        raise RuntimeError("exact tied embedding shape/dtype mismatch")
    digest = tensor_sha256(weight)
    if digest != SHARED_EMBEDDING_SHA256:
        raise RuntimeError("exact tied embedding hash mismatch")
    return nn.Embedding.from_pretrained(weight, freeze=True).to(device), digest


def build_draft(device: torch.device) -> DFlashDraftModel:
    config = Qwen3Config(
        architectures=["DFlashDraftModel"],
        vocab_size=262144,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=8,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=500000,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_theta=50000.0,
        num_target_layers=49,
        block_size=16,
        layer_types=["full_attention"] * 8,
        dflash_config={"target_layer_ids": list(TARGET_BLOCKS), "mask_token_id": 1},
        attn_implementation="eager",
    )
    config._attn_implementation = "eager"
    model = DFlashDraftModel(config).to(device=device, dtype=torch.bfloat16)
    model.train()
    if model.fc.in_features != 5 * 2048 or len(model.layers) != 8:
        raise RuntimeError("acceptance-first draft geometry mismatch")
    return model


def build_training_step(
    draft: DFlashDraftModel,
    embedding: nn.Embedding,
) -> DFlashTrainingStep:
    return DFlashTrainingStep(
        adapter=TransformersDFlashDraftAdapter.from_reference_model(draft),
        shared_weights=FrozenSharedWeights.handoff_tied_embedding(embedding, mask_token_id=1),
        selected_layer_ids=EXTRACTOR_ENTRIES,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8093")
    parser.add_argument("--served-model", default="north-fp8-acceptance-first-micro8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = args.run
    run.mkdir(parents=True, exist_ok=True)
    result_path = run / "pilot-result.json"
    checkpoint_path = run / "checkpoint-step-000008"
    if result_path.exists() or checkpoint_path.exists():
        raise FileExistsError("pilot output already exists")
    responses_dir = run / "teacher-responses"
    responses_dir.mkdir(exist_ok=False)

    identity = TeacherRuntimeIdentity(
        target_name="NorthFP8Target",
        checkpoint_manifest_sha256=CHECKPOINT_MANIFEST_SHA256,
        runtime_image_id=RUNTIME_IMAGE_ID,
        backend="TRITON_FP8_MOE",
        selected_layer_ids=EXTRACTOR_ENTRIES,
        hidden_size=2048,
        prefix_caching_enabled=False,
    )
    ring = BoundedFeatureRing(
        runtime_identity=identity,
        max_items=8,
        max_tokens=8192,
        max_bytes=167772160,
    )
    cases = load_cases(args.source)
    handoffs: list[OwnedConnectorHandoff] = []
    examples: list[ResponseExample] = []
    prompt_lengths: list[int] = []

    for index, case in enumerate(cases):
        request_id = f"fp8-micro8-{index:02d}-{case['description'].split()[-1]}"
        variables = case["vars"]
        generation = post_json(
            f"{args.server}/v1/chat/completions",
            {
                "model": args.served_model,
                "messages": [{"role": "user", "content": PROMPT.format(
                    code=variables["code"], input=variables["input"]
                )}],
                "max_completion_tokens": 512,
                "temperature": 0.0,
                "seed": 20260725 + index,
                "return_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        prompt_ids = generation.get("prompt_token_ids")
        response_ids = generation["choices"][0].get("token_ids")
        if not isinstance(prompt_ids, list) or not isinstance(response_ids, list):
            raise RuntimeError("vLLM did not return exact token IDs")
        if len(response_ids) < 16:
            raise RuntimeError(f"teacher response is too short for block size 16: {request_id}")
        clean_ids = prompt_ids + response_ids
        if len(clean_ids) >= 4096 or len(clean_ids) > ring.max_tokens:
            raise RuntimeError(f"teacher sequence exceeds bounded pilot limits: {request_id}")
        response_record = responses_dir / f"{index:02d}.json"
        response_record.write_text(json.dumps(generation, indent=2, sort_keys=True) + "\n")

        example = ResponseExample(
            prompt_tokens=tuple(prompt_ids),
            response_tokens=tuple(response_ids),
            metadata={
                "request_id": request_id,
                "case_id": case["description"],
                "source_sha256": SOURCE_SHA256,
                "target": identity.target_name,
                "backend": identity.backend,
                "temperature": 0.0,
                "seed": 20260725 + index,
            },
        )
        examples.append(example)
        prompt_lengths.append(len(prompt_ids))

        feature_path = run / f"feature-{index:02d}.safetensors"
        extraction = post_json(
            f"{args.server}/v1/completions",
            {
                "model": args.served_model,
                "prompt": clean_ids,
                "max_tokens": 1,
                "temperature": 0.0,
                "kv_transfer_params": {
                    "hidden_states_path": f"/features/{feature_path.name}",
                    "include_output_tokens": False,
                },
            },
        )
        if extraction.get("usage", {}).get("prompt_tokens") != len(clean_ids):
            raise RuntimeError("extraction prompt-token count drifted")
        wait_for_stable_artifact(feature_path)
        handoffs.append(
            ingest_connector_handoff(
                feature_path,
                request_id=request_id,
                runtime_identity=identity,
                expected_token_ids=torch.tensor(clean_ids, dtype=torch.int64),
                ring=ring,
            )
        )

    ledger_path = run / "response-ledger.jsonl"
    ledger_path.write_text("".join(example.to_json() + "\n" for example in examples))
    response_ledger_sha256 = sha256_file(ledger_path)
    (run / "ring-ready.json").write_text(json.dumps({
        "items": len(ring),
        "tokens": ring.token_count,
        "feature_bytes": ring.feature_bytes,
        "response_ledger_sha256": response_ledger_sha256,
        "request_ids": [handoff.request_id for handoff in handoffs],
    }, indent=2, sort_keys=True) + "\n")

    require_teacher_stopped(run / "teacher-stopped", args.server)
    device = torch.device("cuda")
    torch.manual_seed(20260725)
    embedding, embedding_hash_before = load_exact_embedding(args.model, device)
    draft = build_draft(device)
    training_step = build_training_step(draft, embedding)
    optimizer = torch.optim.AdamW(training_step.parameters(), lr=2e-5, weight_decay=0.01)
    draft_fc_before = tensor_sha256(draft.fc.weight)

    request_records = []
    step_results = []
    release_results = []
    for index, (handoff, prompt_length) in enumerate(zip(handoffs, prompt_lengths, strict=True)):
        step_result = run_one_bounded_optimizer_step(
            ring=ring,
            training_step=training_step,
            optimizer=optimizer,
            prompt_length=prompt_length,
            block_size=16,
            max_anchors=1,
            mask_token_id=1,
            seed=20260725 + index,
            loss_gamma=7.0,
            max_gradient_norm=1.0,
        )
        release = release_connector_after_optimizer_step(
            handoff,
            optimizer_result=step_result,
            ring=ring,
        )
        record = asdict(step_result)
        record.update({
            "connector_sha256": release.feature_sha256,
            "connector_file_bytes_released": release.feature_file_bytes_released,
            "lock_file_released": release.lock_file_released,
        })
        request_records.append(record)
        step_results.append(asdict(step_result))
        release_results.append(asdict(release))

    if len(ring) or ring.token_count or ring.feature_bytes:
        raise RuntimeError("ring is not empty after eight optimizer steps")
    if any((run / f"feature-{index:02d}.safetensors").exists() for index in range(8)):
        raise RuntimeError("transient feature file remains after optimizer acknowledgement")
    torch.cuda.synchronize()
    draft_fc_after = tensor_sha256(draft.fc.weight)
    embedding_hash_after_steps = tensor_sha256(embedding.weight)
    if draft_fc_after == draft_fc_before:
        raise RuntimeError("eight-step pilot did not change draft projection")
    if embedding_hash_after_steps != embedding_hash_before:
        raise RuntimeError("shared tied vocabulary changed during pilot")
    if any(not math.isfinite(item["loss"]) for item in step_results):
        raise RuntimeError("pilot produced a non-finite loss")

    saved_ledger = SavedRequestLedger(entries=tuple(request_records))
    manifest = save_checkpoint(
        checkpoint_path,
        ring=ring,
        training_step=training_step,
        optimizer=optimizer,
        completed_steps=8,
        request_ledger=saved_ledger,
        response_examples=examples,
        runtime_identity=identity,
    )
    root = {
        "checkpoint": str(checkpoint_path),
        "manifest_sha256": manifest.sha256,
        "response_ledger_jsonl_sha256": response_ledger_sha256,
    }
    (run / "checkpoint-root.json").write_text(json.dumps(root, indent=2, sort_keys=True) + "\n")
    verify_checkpoint_directory(
        checkpoint_path,
        runtime_identity=identity,
        expected_manifest_sha256=manifest.sha256,
    )

    saved_fc_hash = draft_fc_after
    del optimizer, training_step, draft
    gc.collect()
    torch.cuda.empty_cache()
    fresh_draft = build_draft(device)
    fresh_step = build_training_step(fresh_draft, embedding)
    fresh_optimizer = torch.optim.AdamW(fresh_step.parameters(), lr=2e-5, weight_decay=0.01)
    resumed_manifest, resumed_requests, resumed_responses = resume_checkpoint(
        checkpoint_path,
        training_step=fresh_step,
        optimizer=fresh_optimizer,
        device=device,
        runtime_identity=identity,
        expected_manifest_sha256=manifest.sha256,
    )
    torch.cuda.synchronize()
    resumed_fc_hash = tensor_sha256(fresh_draft.fc.weight)
    embedding_hash_after_resume = tensor_sha256(embedding.weight)
    if resumed_fc_hash != saved_fc_hash:
        raise RuntimeError("resumed draft projection hash does not match saved state")
    if embedding_hash_after_resume != embedding_hash_before:
        raise RuntimeError("shared vocabulary changed during checkpoint resume")
    if len(resumed_requests.entries) != 8 or len(resumed_responses) != 8:
        raise RuntimeError("checkpoint ledgers did not round-trip")

    checkpoint_files = {}
    for path in sorted(checkpoint_path.iterdir()):
        checkpoint_files[path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    report = {
        "status": "pass",
        "scope": "retained eight-example micro-pilot; not quality or geometry selection",
        "teacher": asdict(identity),
        "source": {
            "path": str(args.source),
            "sha256": SOURCE_SHA256,
            "case_ids": list(CASE_IDS),
        },
        "draft": {
            "candidate": "acceptance_first_qwen3_coder_shaped",
            "layers": 8,
            "features": 5,
            "block_size": 16,
            "dtype": "torch.bfloat16",
            "fc_sha256_before": draft_fc_before,
            "fc_sha256_after": draft_fc_after,
            "fc_sha256_after_resume": resumed_fc_hash,
        },
        "optimization": {
            "steps": step_results,
            "loss_first": step_results[0]["loss"],
            "loss_last": step_results[-1]["loss"],
        },
        "connector_releases": release_results,
        "shared_vocabulary": {
            "sha256_before": embedding_hash_before,
            "sha256_after_steps": embedding_hash_after_steps,
            "sha256_after_resume": embedding_hash_after_resume,
            "gradient_is_none": embedding.weight.grad is None,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "manifest_sha256": resumed_manifest.sha256,
            "step_count": resumed_manifest.step_count,
            "files": checkpoint_files,
            "round_trip_resume": True,
        },
        "response_ledger": {
            "path": str(ledger_path),
            "sha256": response_ledger_sha256,
            "entries": len(examples),
        },
    }
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
