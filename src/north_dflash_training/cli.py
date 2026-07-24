"""CPU-only dry-run CLI for the implemented scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cache import estimate_feature_cache, format_bytes
from .candidate import DEFAULT_TARGET_CONFIG, DEFAULT_TOKENIZER_CONFIG, derive_north_candidate
from .layout import build_training_batch_layout
from .sampling import sample_anchor_blocks
from .schema import ResponseExample
from .weights import exponential_loss_weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="North DFlash training scaffold (no model/GPU/server work)")
    sub = parser.add_subparsers(dest="command", required=True)
    dry = sub.add_parser("dry-run", help="sample synthetic response blocks and estimate feature storage")
    dry.add_argument("--prompt-length", type=int, default=8)
    dry.add_argument("--response-length", type=int, default=64)
    dry.add_argument("--block-size", type=int, default=16)
    dry.add_argument("--max-anchors", type=int, default=4)
    dry.add_argument("--seed", type=int, default=7)
    dry.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="optional assertion; the actual ID is always derived from tokenizer.json",
    )
    dry.add_argument("--gamma", type=float, default=None, help="override gamma for non-paper block sizes")
    dry.add_argument("--num-sequences", type=int, default=800_000)
    dry.add_argument("--sequence-length", type=int, default=3072)
    dry.add_argument("--ring-buffer-tokens", type=int, default=512)
    dry.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_CONFIG)
    dry.add_argument("--tokenizer-config", type=Path, default=DEFAULT_TOKENIZER_CONFIG)
    dry.add_argument("--tokenizer-json", type=Path, default=None, help="override tokenizer.json used by the mask audit")
    dry.add_argument("--show-blocks", type=int, default=2)
    return parser


def _positive(name: str, value: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def run_dry_run(args: argparse.Namespace) -> dict[str, object]:
    _positive("prompt-length", args.prompt_length)
    _positive("response-length", args.response_length)
    _positive("max-anchors", args.max_anchors)
    example = ResponseExample(
        prompt_tokens=tuple(range(10, 10 + args.prompt_length)),
        response_tokens=tuple(range(1000, 1000 + args.response_length)),
        metadata={"source": "synthetic", "target_generated": False},
    )
    candidate = derive_north_candidate(
        args.target_config,
        args.tokenizer_config,
        getattr(args, "tokenizer_json", None),
    )
    audited_mask_token_id = candidate["derived_draft_candidate"]["mask_token_id"]
    if audited_mask_token_id is None:
        raise ValueError("tokenizer.json does not provide a valid audited <MASK_TOKEN> ID")
    if args.mask_token_id is not None and args.mask_token_id != audited_mask_token_id:
        raise ValueError("--mask-token-id disagrees with the tokenizer.json audit")
    sampled = sample_anchor_blocks(
        example,
        block_size=args.block_size,
        max_anchors=args.max_anchors,
        mask_token_id=audited_mask_token_id,
        seed=args.seed,
    )
    layout = build_training_batch_layout(sampled, gamma=args.gamma)
    cache = estimate_feature_cache(
        num_sequences=args.num_sequences,
        sequence_length=args.sequence_length,
        selected_layers=5,
        hidden_size=2048,
        dtype_bytes=2,
        batch_size=1,
        ring_buffer_tokens=args.ring_buffer_tokens,
    )
    return {
        "status": "dry-run only; no model weights, dataset, GPU, or server touched",
        "synthetic_example": {
            "prompt_length": len(example.prompt_tokens),
            "response_length": len(example.response_tokens),
        },
        "sample": {
            "seed": sampled.seed,
            "block_size": sampled.block_size,
            "eligible_anchors": len(sampled.eligible_anchor_positions),
            "sampled_anchors": list(sampled.anchor_positions),
            "blocks": [
                {
                    "anchor_position": block.anchor_position,
                    "absolute_anchor_position": block.absolute_anchor_position,
                    "input_tokens": list(block.input_tokens),
                    "labels": list(block.labels),
                }
                for block in sampled.blocks[: max(0, args.show_blocks)]
            ],
            "loss_weights": list(exponential_loss_weights(args.block_size, gamma=args.gamma)),
            "mask_token_id": layout.mask_token_id,
            "layout": {
                "num_queries": layout.num_queries,
                "num_blocks": layout.num_blocks,
                "block_ids": list(layout.block_ids),
                "anchor_positions": list(layout.anchor_positions),
                "target_context_positions": [list(value) for value in layout.target_context_positions],
                "visibility_is_cpu_relation": True,
            },
        },
        "feature_cache": {
            **cache.to_dict(),
            "disk_cache": format_bytes(cache.disk_cache_bytes),
            "online_peak": format_bytes(cache.online_peak_bytes),
            "ring_buffer": format_bytes(cache.ring_buffer_bytes),
        },
        "north_candidate": candidate,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_dry_run(args)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"dry-run failed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
