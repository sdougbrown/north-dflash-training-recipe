#!/usr/bin/env python3
"""Measure DFlash acceptance on an immutable JSONL prompt slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen


METRICS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


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
    pattern = re.compile(
        rf"^{re.escape(name)}\{{[^\n]*\}}\s+([0-9.eE+-]+)$", re.MULTILINE
    )
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(f"required metric is missing: {name}")
    return float(match.group(1))


def position_values(text: str) -> dict[int, float]:
    pattern = re.compile(
        r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^\n]*position="([0-9]+)"[^\n]*\}\s+([0-9.eE+-]+)$',
        re.MULTILINE,
    )
    values = {
        int(match.group(1)): float(match.group(2)) for match in pattern.finditer(text)
    }
    if not values:
        raise RuntimeError("no per-position acceptance metrics were exported")
    expected = set(range(max(values) + 1))
    if set(values) != expected:
        raise RuntimeError(
            f"per-position acceptance metrics are not contiguous: {sorted(values)}"
        )
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompt_slice(
    path: Path, prompt_field: str, start: int, limit: int
) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start:
                continue
            if len(rows) == limit:
                break
            row = json.loads(line)
            prompt = row.get(prompt_field)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"row {index} has no nonempty {prompt_field!r}")
            rows.append((index, prompt))
    if len(rows) != limit:
        raise ValueError(f"requested {limit} prompts at {start}, found {len(rows)}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--prompt-field", default="instruction")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8096")
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.start < 0 or args.limit < 1 or args.max_tokens < 1:
        parser.error("start must be nonnegative; limit/max-tokens must be positive")
    return args


def main() -> None:
    args = parse_args()
    if sha256_file(args.source) != args.expected_source_sha256:
        raise RuntimeError("held-out source hash mismatch")
    prompts = load_prompt_slice(args.source, args.prompt_field, args.start, args.limit)
    if args.run.exists():
        raise FileExistsError(args.run)
    args.run.mkdir(parents=True)
    responses = args.run / "acceptance-responses"
    responses.mkdir()

    before_text = get_text(f"{args.server}/metrics")
    (args.run / "metrics-before.txt").write_text(before_text)
    before = {name: metric_value(before_text, name) for name in METRICS}
    positions_before = position_values(before_text)

    records = []
    started = time.perf_counter()
    for request_index, (source_index, prompt) in enumerate(prompts):
        payload = {
            "model": args.served_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": args.max_tokens,
            "temperature": 0.0,
            "seed": 20260900 + source_index,
            "ignore_eos": True,
            "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        request_started = time.perf_counter()
        response = post_json(f"{args.server}/v1/chat/completions", payload)
        request_elapsed = time.perf_counter() - request_started
        (responses / f"{request_index:03d}-source-{source_index:03d}.json").write_text(
            json.dumps(response, indent=2, sort_keys=True) + "\n"
        )
        records.append(
            {
                "source_index": source_index,
                "elapsed_seconds": request_elapsed,
                "prompt_tokens": response["usage"]["prompt_tokens"],
                "completion_tokens": response["usage"]["completion_tokens"],
            }
        )

    elapsed = time.perf_counter() - started
    after_text = get_text(f"{args.server}/metrics")
    (args.run / "metrics-after.txt").write_text(after_text)
    after = {name: metric_value(after_text, name) for name in METRICS}
    positions_after = position_values(after_text)
    if positions_after.keys() != positions_before.keys():
        raise RuntimeError("per-position metric set changed during measurement")

    delta = {name: after[name] - before[name] for name in METRICS}
    position_delta = [
        positions_after[position] - positions_before[position]
        for position in positions_before
    ]
    drafts = delta[METRICS[0]]
    draft_tokens = delta[METRICS[1]]
    accepted = delta[METRICS[2]]
    if drafts <= 0 or draft_tokens <= 0:
        raise RuntimeError("DFlash produced no speculative drafts")
    completion_tokens = sum(record["completion_tokens"] for record in records)
    report = {
        "status": "pass",
        "scope": "held-out speculative acceptance; not held-out response quality",
        "source_sha256": args.expected_source_sha256,
        "source_indices": [index for index, _ in prompts],
        "requests": records,
        "metrics_delta": delta,
        "accepted_tokens_per_position": position_delta,
        "mean_accepted_draft_tokens_per_draft": accepted / drafts,
        "mean_emitted_tokens_per_speculative_step": 1.0 + accepted / drafts,
        "draft_token_acceptance_rate": accepted / draft_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_seconds": elapsed,
        "observed_completion_tokens_per_second": completion_tokens / elapsed,
    }
    (args.run / "acceptance-result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
