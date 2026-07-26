#!/usr/bin/env python3
"""Regenerate single-turn JSONL prompts with an OpenAI-compatible target server.

This is a bounded data-ingestion adapter for the upstream Speculators training
pipeline. It does not implement DFlash training semantics. Each completed
response is durably published as one indexed JSON file; the final JSONL is
published only when every input has succeeded.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import aiohttp


def load_prompts(
    path: Path, prompt_field: str, source_index_offset: int = 0
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get(prompt_field)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"row {index} has no nonempty {prompt_field!r}")
            rows.append(
                {
                    "source_index": source_index_offset + index,
                    "prompt": prompt,
                    "source": row,
                }
            )
    if not rows:
        raise ValueError("input contains no prompts")
    return rows


def stable_id(source_index: int, prompt: str) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return f"source-{source_index:06d}-{digest}"


def build_sample(
    *,
    source_index: int,
    prompt: str,
    response: dict[str, Any],
    endpoint: str,
    sampling: dict[str, Any],
) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("response must contain exactly one choice")
    choice = choices[0]
    prompt_ids = response.get("prompt_token_ids")
    completion_ids = choice.get("token_ids")
    message = choice.get("message")
    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise ValueError("response is missing prompt_token_ids")
    if not isinstance(completion_ids, list) or not completion_ids:
        raise ValueError("response is missing completion token_ids")
    if not all(isinstance(token, int) for token in prompt_ids + completion_ids):
        raise ValueError("response token IDs must be integers")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("response is missing assistant content")

    primary_id = stable_id(source_index, prompt)
    input_ids = [*prompt_ids, *completion_ids]
    return {
        "id": f"{primary_id}_gen0",
        "primary_id": primary_id,
        "input_ids": input_ids,
        "loss_mask": [0] * len(prompt_ids) + [1] * len(completion_ids),
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": message["content"]},
        ],
        "metadata": {
            "source_index": source_index,
            "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage") or {},
            "endpoint": endpoint,
            "sampling_params": sampling,
        },
    }


def publish_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


async def post_with_retries(
    session: aiohttp.ClientSession,
    endpoint: str,
    payload: dict[str, Any],
    max_retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with session.post(endpoint, json=payload) as response:
                body = await response.text()
                if response.status >= 500 or response.status in {408, 409, 425, 429}:
                    raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
                if response.status >= 400:
                    raise ValueError(f"HTTP {response.status}: {body[:500]}")
                return json.loads(body)
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as error:
            last_error = error
            if attempt == max_retries:
                break
            await asyncio.sleep(2**attempt)
    assert last_error is not None
    raise last_error


async def regenerate(args: argparse.Namespace) -> None:
    rows = load_prompts(args.input, args.prompt_field, args.source_index_offset)
    args.records.mkdir(parents=True, exist_ok=True)
    args.errors.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    for row in rows:
        record_path = args.records / f"{row['source_index']:06d}.json"
        if not record_path.exists():
            queue.put_nowait(row)
    for _ in range(args.concurrency):
        queue.put_nowait(None)

    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    timeout = aiohttp.ClientTimeout(total=args.timeout)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def worker() -> None:
            while True:
                row = await queue.get()
                try:
                    if row is None:
                        return
                    index = row["source_index"]
                    prompt = row["prompt"]
                    payload = {
                        "model": args.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": args.max_tokens,
                        "return_token_ids": True,
                        **sampling,
                    }
                    response = await post_with_retries(
                        session, args.endpoint, payload, args.max_retries
                    )
                    sample = build_sample(
                        source_index=index,
                        prompt=prompt,
                        response=response,
                        endpoint=args.endpoint,
                        sampling=sampling,
                    )
                    publish_json_exclusive(args.records / f"{index:06d}.json", sample)
                    print(json.dumps({"completed": index, "tokens": len(sample["input_ids"])}), flush=True)
                except Exception as error:  # noqa: BLE001
                    if row is not None:
                        error_path = args.errors / f"{row['source_index']:06d}.json"
                        if not error_path.exists():
                            publish_json_exclusive(
                                error_path,
                                {"source_index": row["source_index"], "error": repr(error)},
                            )
                    print(json.dumps({"failed": None if row is None else row["source_index"], "error": repr(error)}), flush=True)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
        await queue.join()
        await asyncio.gather(*workers)

    missing = [
        row["source_index"]
        for row in rows
        if not (args.records / f"{row['source_index']:06d}.json").exists()
    ]
    if missing:
        raise RuntimeError(f"{len(missing)} responses missing; first indices: {missing[:20]}")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as handle:
            for row in rows:
                record = args.records / f"{row['source_index']:06d}.json"
                handle.write(record.read_text(encoding="utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompt-field", default="instruction")
    parser.add_argument("--source-index-offset", type=int, default=0)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8095/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()
    if (
        args.concurrency < 1
        or args.max_tokens < 1
        or args.max_retries < 0
        or args.source_index_offset < 0
    ):
        parser.error(
            "concurrency/max-tokens must be positive; retries/offset nonnegative"
        )
    return args


if __name__ == "__main__":
    asyncio.run(regenerate(parse_args()))
