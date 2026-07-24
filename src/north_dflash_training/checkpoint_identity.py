"""Exact checkpoint-file identity without loading model tensors.

The hasher reads config, a Hugging Face weight index, and every uniquely named
shard declared by that index in bounded chunks.  It intentionally has no CLI:
calling it for a real multi-gigabyte checkpoint is an explicit review action,
not part of the dry-run or training scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_INDEX_FILENAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
DEFAULT_CHUNK_BYTES = 1024 * 1024


def sha256_file(path: str | Path, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    """Return a SHA-256 digest while reading a file incrementally."""
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointFileDigest:
    """Digest and byte count for one checkpoint file, with a relative name."""

    relative_path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"relative_path": self.relative_path, "size_bytes": self.size_bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class CheckpointIdentityManifest:
    """Exact file-level identity of a config/index/shard checkpoint snapshot."""

    config: CheckpointFileDigest
    index: CheckpointFileDigest
    shards: tuple[CheckpointFileDigest, ...]
    manifest_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported checkpoint identity manifest schema version")
        if not self.shards:
            raise ValueError("checkpoint index must declare at least one shard")
        names = [item.relative_path for item in (self.config, self.index, *self.shards)]
        if len(set(names)) != len(names):
            raise ValueError("config, index, and shard paths must be distinct")
        if tuple(item.relative_path for item in self.shards) != tuple(sorted(item.relative_path for item in self.shards)):
            raise ValueError("shards must be sorted by relative path")
        for item in (self.config, self.index, *self.shards):
            _validate_digest(item)
        expected = _manifest_digest(self.config, self.index, self.shards)
        if self.manifest_sha256 != expected:
            raise ValueError("manifest_sha256 does not match its file digests")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "index": self.index.to_dict(),
            "shards": [item.to_dict() for item in self.shards],
            "manifest_sha256": self.manifest_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointIdentityManifest":
        try:
            def load_file(item: Mapping[str, Any]) -> CheckpointFileDigest:
                return CheckpointFileDigest(
                    relative_path=item["relative_path"], size_bytes=item["size_bytes"], sha256=item["sha256"]
                )
            return cls(
                schema_version=value["schema_version"],
                config=load_file(value["config"]),
                index=load_file(value["index"]),
                shards=tuple(load_file(item) for item in value["shards"]),
                manifest_sha256=value["manifest_sha256"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid checkpoint identity manifest: {exc}") from exc


def _validate_digest(item: CheckpointFileDigest) -> None:
    relative = Path(item.relative_path)
    if relative.is_absolute() or ".." in relative.parts or item.relative_path in {"", "."}:
        raise ValueError("checkpoint manifest paths must be non-empty relative paths")
    if not isinstance(item.size_bytes, int) or isinstance(item.size_bytes, bool) or item.size_bytes < 0:
        raise ValueError("checkpoint file size must be a non-negative integer")
    if not isinstance(item.sha256, str) or len(item.sha256) != 64 or any(c not in "0123456789abcdef" for c in item.sha256):
        raise ValueError("checkpoint file digest must be a lowercase SHA-256 digest")


def _manifest_digest(
    config: CheckpointFileDigest, index: CheckpointFileDigest, shards: tuple[CheckpointFileDigest, ...]
) -> str:
    payload = {
        "schema_version": 1,
        "config": config.to_dict(),
        "index": index.to_dict(),
        "shards": [item.to_dict() for item in shards],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative_existing_file(root: Path, relative_path: str) -> Path:
    candidate = root / relative_path
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"checkpoint file escapes root: {relative_path}") from exc
    if not candidate.is_file():
        raise ValueError(f"checkpoint file is missing or not regular: {relative_path}")
    return candidate


def _digest_file(root: Path, relative_path: str, *, chunk_bytes: int) -> CheckpointFileDigest:
    path = _relative_existing_file(root, relative_path)
    return CheckpointFileDigest(
        relative_path=relative_path,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path, chunk_bytes=chunk_bytes),
    )


def _find_index(root: Path, index_filename: str | None) -> str:
    if index_filename is not None:
        return index_filename
    found = [name for name in DEFAULT_INDEX_FILENAMES if (root / name).is_file()]
    if len(found) != 1:
        raise ValueError("expected exactly one known Hugging Face checkpoint index; pass index_filename explicitly")
    return found[0]


def build_checkpoint_identity_manifest(
    checkpoint_dir: str | Path,
    *,
    config_filename: str = "config.json",
    index_filename: str | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> CheckpointIdentityManifest:
    """Hash config, index, and index-declared shards without loading tensors.

    Only paths listed by ``weight_map`` are hashed.  An index lacking a mapping,
    a missing shard, or a path escaping ``checkpoint_dir`` is rejected.
    """
    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise ValueError("checkpoint_dir must be an existing directory")
    index_name = _find_index(root, index_filename)
    index_path = _relative_existing_file(root, index_name)
    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index_data["weight_map"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid checkpoint index: {exc}") from exc
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("checkpoint index weight_map must be a non-empty object")
    shard_values = tuple(weight_map.values())
    if any(not isinstance(name, str) for name in shard_values):
        raise ValueError("checkpoint index weight_map values must be shard paths")
    shard_names = sorted(set(shard_values))
    config = _digest_file(root, config_filename, chunk_bytes=chunk_bytes)
    index = _digest_file(root, index_name, chunk_bytes=chunk_bytes)
    shards = tuple(_digest_file(root, name, chunk_bytes=chunk_bytes) for name in shard_names)
    return CheckpointIdentityManifest(
        config=config,
        index=index,
        shards=shards,
        manifest_sha256=_manifest_digest(config, index, shards),
    )


def verify_checkpoint_identity(
    checkpoint_dir: str | Path,
    expected: CheckpointIdentityManifest | Mapping[str, Any],
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> None:
    """Rehash exactly the recorded files and raise when any identity differs."""
    manifest = (
        CheckpointIdentityManifest.from_dict(expected) if isinstance(expected, Mapping) else expected
    )
    if not isinstance(manifest, CheckpointIdentityManifest):
        raise TypeError("expected must be a CheckpointIdentityManifest or mapping")
    root = Path(checkpoint_dir)
    actual_config = _digest_file(root, manifest.config.relative_path, chunk_bytes=chunk_bytes)
    actual_index = _digest_file(root, manifest.index.relative_path, chunk_bytes=chunk_bytes)
    actual_shards = tuple(
        _digest_file(root, item.relative_path, chunk_bytes=chunk_bytes) for item in manifest.shards
    )
    actual_digest = _manifest_digest(actual_config, actual_index, actual_shards)
    if actual_digest != manifest.manifest_sha256:
        raise ValueError("checkpoint identity mismatch: config, index, or shard digest changed")
