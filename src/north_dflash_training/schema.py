"""A small, JSONL-friendly schema for target-generated response examples."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

SCHEMA_VERSION = 1
IGNORE_INDEX = -100


@dataclass(frozen=True)
class ResponseExample:
    """Tokenized prompt/response pair used by the sampling scaffold.

    ``response_tokens`` should be the clean tokens emitted by the target model.
    The sampler only masks response futures; prompt tokens remain available for
    the eventual target-feature extraction step. Metadata is intentionally
    opaque so dataset provenance and tokenizer identity can be carried without
    adding a training-framework dependency.
    """

    prompt_tokens: tuple[int, ...]
    response_tokens: tuple[int, ...]
    metadata: Mapping[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        _validate_tokens(self.prompt_tokens, "prompt_tokens")
        _validate_tokens(self.response_tokens, "response_tokens")
        if not self.response_tokens:
            raise ValueError("response_tokens must not be empty")
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be an object/mapping")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResponseExample":
        if not isinstance(value, Mapping):
            raise TypeError("response example must be a JSON object")
        required = {"schema_version", "prompt_tokens", "response_tokens"}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"missing required fields: {sorted(missing)}")
        unknown = set(value) - required - {"metadata"}
        if unknown:
            raise ValueError(f"unknown fields: {sorted(unknown)}")
        return cls(
            prompt_tokens=tuple(value["prompt_tokens"]),
            response_tokens=tuple(value["response_tokens"]),
            metadata=value.get("metadata"),
            schema_version=value["schema_version"],
        )

    @classmethod
    def from_json(cls, text: str) -> "ResponseExample":
        return cls.from_mapping(json.loads(text))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "prompt_tokens": list(self.prompt_tokens),
            "response_tokens": list(self.response_tokens),
        }
        if self.metadata is not None:
            result["metadata"] = dict(self.metadata)
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def _validate_tokens(tokens: Any, name: str) -> None:
    if not isinstance(tokens, (tuple, list)):
        raise TypeError(f"{name} must be an array of integer token IDs")
    for index, token in enumerate(tokens):
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError(f"{name}[{index}] must be a non-negative integer")
