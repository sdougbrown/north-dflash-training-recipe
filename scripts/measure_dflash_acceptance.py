#!/usr/bin/env python3
"""Measure DFlash speculative acceptance on explicitly held-out prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

import yaml

from run_fp8_acceptance_first_micro8 import PROMPT, SOURCE_SHA256, sha256_file

METRICS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8094")
    parser.add_argument("--served-model", default="north-fp8-dflash-pilot40")
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def get_text(url: str) -> str:
    with urlopen(url, timeout=30) as response:
        return response.read().decode()


def post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
        method="POST",
    )
    with urlopen(request, timeout=900) as response:
        return json.load(response)


def metric_value(text: str, name: str) -> float:
    pattern = re.compile(rf"^{re.escape(name)}\{{[^\n]*\}}\s+([0-9.eE+-]+)$", re.MULTILINE)
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(f"required metric is missing: {name}")
    return float(match.group(1))


def position_values(text: str) -> list[float]:
    values = []
    for position in range(15):
        pattern = re.compile(
            rf'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{{[^\n]*position="{position}"[^\n]*\}}\s+([0-9.eE+-]+)$',
            re.MULTILINE,
        )
        match = pattern.search(text)
        if match is None:
            raise RuntimeError(f"position metric is missing: {position}")
        values.append(float(match.group(1)))
    return values


def main() -> None:
    args = parse_args()
    run = args.run
    run.mkdir(parents=True, exist_ok=True)
    result_path = run / "acceptance-result.json"
    responses = run / "acceptance-responses"
    if result_path.exists() or responses.exists():
        raise FileExistsError("acceptance output already exists")
    responses.mkdir()
    if sha256_file(args.source) != SOURCE_SHA256:
        raise RuntimeError("held-out source hash mismatch")
    raw = yaml.safe_load(args.source.read_text())
    by_id = {item["description"]: item for item in raw}
    case_ids = tuple(args.case_id)
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise RuntimeError("held-out case IDs must be non-empty and unique")
    if any(case_id not in by_id for case_id in case_ids):
        raise RuntimeError("held-out case is missing")

    before_text = get_text(f"{args.server}/metrics")
    (run / "metrics-before.txt").write_text(before_text)
    before = {name: metric_value(before_text, name) for name in METRICS}
    positions_before = position_values(before_text)
    records = []
    started = time.perf_counter()
    for index, case_id in enumerate(case_ids):
        variables = by_id[case_id]["vars"]
        payload = {
            "model": args.served_model,
            "messages": [{
                "role": "user",
                "content": PROMPT.format(code=variables["code"], input=variables["input"]),
            }],
            "max_completion_tokens": args.max_tokens,
            "temperature": 0.0,
            "seed": 20260800 + index,
            "ignore_eos": True,
            "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        request_started = time.perf_counter()
        response = post_json(f"{args.server}/v1/chat/completions", payload)
        elapsed = time.perf_counter() - request_started
        (responses / f"{index:02d}.json").write_text(
            json.dumps(response, indent=2, sort_keys=True) + "\n"
        )
        records.append({
            "case_id": case_id,
            "elapsed_seconds": elapsed,
            "prompt_tokens": response["usage"]["prompt_tokens"],
            "completion_tokens": response["usage"]["completion_tokens"],
        })
    elapsed_total = time.perf_counter() - started
    after_text = get_text(f"{args.server}/metrics")
    (run / "metrics-after.txt").write_text(after_text)
    after = {name: metric_value(after_text, name) for name in METRICS}
    positions_after = position_values(after_text)
    delta = {name: after[name] - before[name] for name in METRICS}
    position_delta = [a - b for a, b in zip(positions_after, positions_before, strict=True)]
    drafts = delta[METRICS[0]]
    draft_tokens = delta[METRICS[1]]
    accepted = delta[METRICS[2]]
    if drafts <= 0 or draft_tokens <= 0:
        raise RuntimeError("DFlash produced no speculative drafts")
    completion_tokens = sum(record["completion_tokens"] for record in records)
    report = {
        "status": "pass",
        "scope": "held-out speculative acceptance; not held-out quality",
        "source_sha256": SOURCE_SHA256,
        "case_ids": list(case_ids),
        "requests": records,
        "metrics_delta": delta,
        "accepted_tokens_per_position": position_delta,
        "mean_accepted_draft_tokens_per_draft": accepted / drafts,
        "mean_emitted_tokens_per_speculative_step": 1.0 + accepted / drafts,
        "draft_token_acceptance_rate": accepted / draft_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_seconds": elapsed_total,
        "observed_completion_tokens_per_second": completion_tokens / elapsed_total,
    }
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
