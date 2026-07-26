#!/usr/bin/env python3
"""Measure warm sequential generation on an immutable JSONL prompt slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from measure_dflash_jsonl_acceptance import load_prompt_slice, post_json, sha256_file


def token_id_root(token_ids: list[list[int]]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-token-root-sha256")
    parser.add_argument("--prompt-field", default="instruction")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8096")
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.start < 0 or min(args.limit, args.max_tokens, args.warmup_tokens) < 1:
        parser.error("start must be nonnegative; limits must be positive")
    return args


def request_payload(model: str, prompt: str, max_tokens: int, seed: int) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "ignore_eos": True,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def main() -> None:
    args = parse_args()
    if sha256_file(args.source) != args.expected_source_sha256:
        raise RuntimeError("source hash mismatch")
    prompts = load_prompt_slice(args.source, args.prompt_field, args.start, args.limit)
    if args.run.exists():
        raise FileExistsError(args.run)
    args.run.mkdir(parents=True)
    responses = args.run / "responses"
    responses.mkdir()

    warmup = post_json(
        f"{args.server}/v1/chat/completions",
        request_payload(args.served_model, prompts[0][1], args.warmup_tokens, 0),
    )
    (args.run / "warmup-response.json").write_text(
        json.dumps(warmup, indent=2, sort_keys=True) + "\n"
    )

    records = []
    emitted: list[list[int]] = []
    started = time.perf_counter()
    for request_index, (source_index, prompt) in enumerate(prompts):
        request_started = time.perf_counter()
        response = post_json(
            f"{args.server}/v1/chat/completions",
            request_payload(
                args.served_model,
                prompt,
                args.max_tokens,
                20260900 + source_index,
            ),
        )
        elapsed = time.perf_counter() - request_started
        token_ids = response["choices"][0]["token_ids"]
        if len(token_ids) != args.max_tokens:
            raise RuntimeError(
                f"source {source_index} emitted {len(token_ids)} tokens; "
                f"expected {args.max_tokens}"
            )
        emitted.append(token_ids)
        (responses / f"{request_index:03d}-source-{source_index:03d}.json").write_text(
            json.dumps(response, indent=2, sort_keys=True) + "\n"
        )
        records.append(
            {
                "source_index": source_index,
                "elapsed_seconds": elapsed,
                "prompt_tokens": response["usage"]["prompt_tokens"],
                "completion_tokens": response["usage"]["completion_tokens"],
            }
        )
    elapsed = time.perf_counter() - started
    root = token_id_root(emitted)
    if args.expected_token_root_sha256 and root != args.expected_token_root_sha256:
        raise RuntimeError(
            f"output token root mismatch: expected "
            f"{args.expected_token_root_sha256}, got {root}"
        )
    completion_tokens = sum(record["completion_tokens"] for record in records)
    report = {
        "status": "pass",
        "scope": "warm sequential generation throughput; not response quality",
        "source_sha256": args.expected_source_sha256,
        "source_indices": [index for index, _ in prompts],
        "output_token_id_root_sha256": root,
        "expected_output_token_id_root_sha256": args.expected_token_root_sha256,
        "output_token_ids_exact": (
            args.expected_token_root_sha256 is None
            or root == args.expected_token_root_sha256
        ),
        "requests": records,
        "completion_tokens": completion_tokens,
        "elapsed_seconds": elapsed,
        "observed_completion_tokens_per_second": completion_tokens / elapsed,
    }
    (args.run / "generation-result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
