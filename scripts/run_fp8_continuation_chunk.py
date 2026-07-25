#!/usr/bin/env python3
"""Continue the retained FP8 pilot by one bounded teacher/training chunk."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path

import torch
import yaml

import run_fp8_acceptance_first_micro8 as base
from north_dflash_training.connector_lifecycle import (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--previous-checkpoint", type=Path, required=True)
    parser.add_argument("--previous-manifest-sha256", required=True)
    parser.add_argument("--expected-previous-steps", type=int, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8093")
    parser.add_argument("--served-model", default="north-fp8-retained-pilot")
    return parser.parse_args()


def load_selected_cases(source: Path, case_ids: tuple[str, ...]) -> list[dict]:
    if base.sha256_file(source) != base.SOURCE_SHA256:
        raise RuntimeError("local CRUXEval source hash mismatch")
    raw = yaml.safe_load(source.read_text())
    by_id = {item["description"]: item for item in raw}
    if len(case_ids) > 8 or not case_ids or len(set(case_ids)) != len(case_ids):
        raise RuntimeError("a continuation chunk requires one to eight unique cases")
    if any(case_id not in by_id for case_id in case_ids):
        raise RuntimeError("configured CRUXEval case is missing")
    return [by_id[case_id] for case_id in case_ids]


def main() -> None:
    args = parse_args()
    if args.expected_previous_steps < 1:
        raise RuntimeError("continuation requires a retained prior step")
    case_ids = tuple(args.case_id)
    cases = load_selected_cases(args.source, case_ids)
    final_steps = args.expected_previous_steps + len(cases)
    run = args.run
    run.mkdir(parents=True, exist_ok=True)
    features_mount = Path("/features")
    if not features_mount.is_dir() or not os.path.samefile(features_mount, run):
        raise RuntimeError("/features and --run must bind-mount the same host run directory")
    if any(run.glob("*.safetensors")) or any(run.glob("*.safetensors.lock")):
        raise RuntimeError("continuation run directory contains a stale connector artifact")
    result_path = run / "chunk-result.json"
    checkpoint_path = run / f"checkpoint-step-{final_steps:06d}"
    if result_path.exists() or checkpoint_path.exists():
        raise FileExistsError("continuation output already exists")
    responses_dir = run / "teacher-responses"
    responses_dir.mkdir(exist_ok=False)

    identity = TeacherRuntimeIdentity(
        target_name="NorthFP8Target",
        checkpoint_manifest_sha256=base.CHECKPOINT_MANIFEST_SHA256,
        runtime_image_id=base.RUNTIME_IMAGE_ID,
        backend="TRITON_FP8_MOE",
        selected_layer_ids=base.EXTRACTOR_ENTRIES,
        hidden_size=2048,
        prefix_caching_enabled=False,
    )
    ring = BoundedFeatureRing(
        runtime_identity=identity,
        max_items=len(cases),
        max_tokens=8192,
        max_bytes=167772160,
    )
    handoffs = []
    examples: list[ResponseExample] = []
    prompt_lengths: list[int] = []
    generation_releases: list[dict] = []

    for offset, case in enumerate(cases):
        step_number = args.expected_previous_steps + offset + 1
        request_id = f"fp8-retained-{step_number:04d}-{case['description'].split()[-1]}"
        variables = case["vars"]
        generation_feature_path = run / f"generation-{offset:02d}.safetensors"
        generation = base.post_json(
            f"{args.server}/v1/chat/completions",
            {
                "model": args.served_model,
                "messages": [{
                    "role": "user",
                    "content": base.PROMPT.format(
                        code=variables["code"], input=variables["input"]
                    ),
                }],
                "max_completion_tokens": 512,
                "temperature": 0.0,
                "seed": 20260725 + step_number,
                "return_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": True},
                "kv_transfer_params": {
                    "hidden_states_path": f"/features/{generation_feature_path.name}",
                    "include_output_tokens": False,
                },
            },
        )
        prompt_ids = generation.get("prompt_token_ids")
        response_ids = generation["choices"][0].get("token_ids")
        if not isinstance(prompt_ids, list) or not isinstance(response_ids, list):
            raise RuntimeError("vLLM did not return exact token IDs")
        if len(response_ids) < 16:
            raise RuntimeError(f"teacher response is too short: {request_id}")
        clean_ids = prompt_ids + response_ids
        if len(clean_ids) >= 4096:
            raise RuntimeError(f"teacher sequence exceeds pilot limit: {request_id}")
        response_record = responses_dir / f"{offset:02d}.json"
        response_record.write_text(json.dumps(generation, indent=2, sort_keys=True) + "\n")
        generation_releases.append(base.release_generation_connector(generation_feature_path))

        example = ResponseExample(
            prompt_tokens=tuple(prompt_ids),
            response_tokens=tuple(response_ids),
            metadata={
                "request_id": request_id,
                "case_id": case["description"],
                "source_sha256": base.SOURCE_SHA256,
                "target": identity.target_name,
                "backend": identity.backend,
                "temperature": 0.0,
                "seed": 20260725 + step_number,
            },
        )
        examples.append(example)
        prompt_lengths.append(len(prompt_ids))

        feature_path = run / f"feature-{offset:02d}.safetensors"
        extraction = base.post_json(
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
        base.wait_for_stable_artifact(feature_path)
        handoffs.append(
            ingest_connector_handoff(
                feature_path,
                request_id=request_id,
                runtime_identity=identity,
                expected_token_ids=torch.tensor(clean_ids, dtype=torch.int64),
                ring=ring,
            )
        )

    current_ledger_path = run / "current-response-ledger.jsonl"
    current_ledger_path.write_text("".join(example.to_json() + "\n" for example in examples))
    current_ledger_sha256 = base.sha256_file(current_ledger_path)
    (run / "ring-ready.json").write_text(json.dumps({
        "items": len(ring),
        "tokens": ring.token_count,
        "feature_bytes": ring.feature_bytes,
        "previous_steps": args.expected_previous_steps,
        "final_steps": final_steps,
        "response_ledger_sha256": current_ledger_sha256,
        "request_ids": [handoff.request_id for handoff in handoffs],
        "generation_connector_releases": generation_releases,
    }, indent=2, sort_keys=True) + "\n")

    base.require_teacher_stopped(run / "teacher-stopped", args.server)
    device = torch.device("cuda")
    embedding, embedding_hash_before = base.load_exact_embedding(args.model, device)
    draft = base.build_draft(device)
    training_step = base.build_training_step(draft, embedding)
    optimizer = torch.optim.AdamW(training_step.parameters(), lr=2e-5, weight_decay=0.01)
    previous_manifest, previous_requests, previous_responses = resume_checkpoint(
        args.previous_checkpoint,
        training_step=training_step,
        optimizer=optimizer,
        device=device,
        runtime_identity=identity,
        expected_manifest_sha256=args.previous_manifest_sha256,
    )
    if previous_manifest.step_count != args.expected_previous_steps:
        raise RuntimeError("previous checkpoint step count drifted")
    draft_fc_before = base.tensor_sha256(draft.fc.weight)

    current_request_records = []
    step_results = []
    release_results = []
    for offset, (handoff, prompt_length) in enumerate(
        zip(handoffs, prompt_lengths, strict=True)
    ):
        step_number = args.expected_previous_steps + offset + 1
        step_result = run_one_bounded_optimizer_step(
            ring=ring,
            training_step=training_step,
            optimizer=optimizer,
            prompt_length=prompt_length,
            block_size=16,
            max_anchors=1,
            mask_token_id=1,
            seed=20260725 + step_number,
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
        current_request_records.append(record)
        step_results.append(asdict(step_result))
        release_results.append(asdict(release))

    if len(ring) or ring.token_count or ring.feature_bytes:
        raise RuntimeError("ring is not empty after continuation chunk")
    if any(run.glob("*.safetensors")) or any(run.glob("*.safetensors.lock")):
        raise RuntimeError("transient connector artifact remains")
    torch.cuda.synchronize()
    draft_fc_after = base.tensor_sha256(draft.fc.weight)
    embedding_hash_after_steps = base.tensor_sha256(embedding.weight)
    if draft_fc_after == draft_fc_before:
        raise RuntimeError("continuation chunk did not change draft projection")
    if embedding_hash_after_steps != embedding_hash_before:
        raise RuntimeError("shared vocabulary changed during continuation")
    if any(not math.isfinite(item["loss"]) for item in step_results):
        raise RuntimeError("continuation produced a non-finite loss")

    combined_requests = SavedRequestLedger(
        entries=previous_requests.entries + tuple(current_request_records)
    )
    combined_responses = tuple(previous_responses) + tuple(examples)
    manifest = save_checkpoint(
        checkpoint_path,
        ring=ring,
        training_step=training_step,
        optimizer=optimizer,
        completed_steps=final_steps,
        request_ledger=combined_requests,
        response_examples=combined_responses,
        runtime_identity=identity,
    )
    (run / "checkpoint-root.json").write_text(json.dumps({
        "checkpoint": str(checkpoint_path),
        "manifest_sha256": manifest.sha256,
        "parent_checkpoint": str(args.previous_checkpoint),
        "parent_manifest_sha256": args.previous_manifest_sha256,
        "current_response_ledger_sha256": current_ledger_sha256,
    }, indent=2, sort_keys=True) + "\n")
    verify_checkpoint_directory(
        checkpoint_path,
        runtime_identity=identity,
        expected_manifest_sha256=manifest.sha256,
    )

    saved_fc_hash = draft_fc_after
    del optimizer, training_step, draft
    gc.collect()
    torch.cuda.empty_cache()
    fresh_draft = base.build_draft(device)
    fresh_step = base.build_training_step(fresh_draft, embedding)
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
    resumed_fc_hash = base.tensor_sha256(fresh_draft.fc.weight)
    embedding_hash_after_resume = base.tensor_sha256(embedding.weight)
    if resumed_fc_hash != saved_fc_hash:
        raise RuntimeError("resumed continuation draft does not match")
    if embedding_hash_after_resume != embedding_hash_before:
        raise RuntimeError("shared vocabulary changed during resume")
    if len(resumed_requests.entries) != final_steps or len(resumed_responses) != final_steps:
        raise RuntimeError("cumulative ledgers did not round-trip")

    checkpoint_files = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": base.sha256_file(path)}
        for path in sorted(checkpoint_path.iterdir())
    }
    report = {
        "status": "pass",
        "scope": "bounded continuation calibration chunk; not quality or geometry selection",
        "teacher": asdict(identity),
        "source": {
            "path": str(args.source),
            "sha256": base.SOURCE_SHA256,
            "case_ids": list(case_ids),
        },
        "parent": {
            "checkpoint": str(args.previous_checkpoint),
            "manifest_sha256": previous_manifest.sha256,
            "step_count": previous_manifest.step_count,
        },
        "draft": {
            "fc_sha256_before": draft_fc_before,
            "fc_sha256_after": draft_fc_after,
            "fc_sha256_after_resume": resumed_fc_hash,
        },
        "optimization": {
            "steps": step_results,
            "current_step_count": len(step_results),
            "cumulative_step_count": final_steps,
        },
        "connector_releases": release_results,
        "generation_connector_releases": generation_releases,
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
    }
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
