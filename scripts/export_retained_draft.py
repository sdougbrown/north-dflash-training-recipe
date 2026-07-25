#!/usr/bin/env python3
"""Export a verified retained draft checkpoint into a vLLM-loadable HF artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from safetensors.torch import load as load_safetensors
from safetensors.torch import save_file

from north_dflash_training.feature_stream import TeacherRuntimeIdentity
from north_dflash_training.save_resume import _publish_no_replace, verify_checkpoint_directory


FP8_IDENTITY = TeacherRuntimeIdentity(
    target_name="NorthFP8Target",
    checkpoint_manifest_sha256="35812fdf32f497a558f31bbea43e7d69f8c1cd43c66530c7499de2f293ae2bb6",
    runtime_image_id="sha256:d89ae2666c80ea7d64c903670cd5b8a643f03e878c65aa21b534bc6792dba637",
    backend="TRITON_FP8_MOE",
    selected_layer_ids=(2, 13, 25, 36, 47),
    hidden_size=2048,
    prefix_caching_enabled=False,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retained_bytes(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        chunks = []
        while chunk := os.read(fd, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite export: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = verify_checkpoint_directory(
        args.checkpoint,
        runtime_identity=FP8_IDENTITY,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    weight_record = next(
        item for item in manifest.files if item["relative_path"] == "draft-model.safetensors"
    )
    source_bytes = retained_bytes(args.checkpoint / "draft-model.safetensors")
    if len(source_bytes) != weight_record["size_bytes"]:
        raise RuntimeError("retained draft byte count drifted after verification")
    if sha256_bytes(source_bytes) != weight_record["sha256"]:
        raise RuntimeError("retained draft hash drifted after verification")
    source_state = load_safetensors(source_bytes)
    prefix = "draft_model."
    if not source_state or any(not key.startswith(prefix) for key in source_state):
        raise RuntimeError("retained adapter state has an unexpected key namespace")
    export_state = {key[len(prefix):]: tensor for key, tensor in source_state.items()}
    if len(export_state) != len(source_state) or len(set(export_state)) != len(export_state):
        raise RuntimeError("draft key conversion is not one-to-one")
    prohibited = [key for key in export_state if "embed_tokens" in key or "lm_head" in key]
    if prohibited:
        raise RuntimeError("draft export would duplicate tied target vocabulary")
    required = {"fc.weight", "hidden_norm.weight", "norm.weight"}
    if not required.issubset(export_state):
        raise RuntimeError("draft export lacks required DFlash tensors")

    config = dict(manifest.draft_architecture["config"])
    config.update({
        "_name_or_path": "North-FP8-DFlash-retained-pilot",
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "dtype": "bfloat16",
        "draft_vocab_size": 262144,
        "target_hidden_size": 2048,
        "eagle_aux_hidden_state_layer_ids": [2, 13, 25, 36, 47],
        "tie_word_embeddings": False,
        "rope_theta": 50000.0,
    })
    dflash_config = dict(config["dflash_config"])
    dflash_config.update({
        "target_layer_ids": [1, 12, 24, 35, 46],
        "target_layer_id_convention": "zero_based_transformer_block_index",
        "mask_token_id": 1,
        "sample_from_anchor": False,
        "causal": False,
        "use_aux_hidden_state": True,
    })
    config["dflash_config"] = dflash_config
    if config.get("num_hidden_layers") != 8 or config.get("block_size") != 16:
        raise RuntimeError("retained draft geometry is not the reviewed candidate")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        weights_path = staging / "model.safetensors"
        save_file(export_state, weights_path, metadata={"format": "torch"})
        config_path = staging / "config.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        export_manifest = {
            "artifact_kind": "NORTH_FP8_RETAINED_DFLASH_DRAFT_EXPORT",
            "source_checkpoint": str(args.checkpoint),
            "source_manifest_sha256": manifest.sha256,
            "source_step_count": manifest.step_count,
            "runtime_identity": {
                "target_name": FP8_IDENTITY.target_name,
                "target_checkpoint_manifest_sha256": FP8_IDENTITY.checkpoint_manifest_sha256,
                "runtime_image_id": FP8_IDENTITY.runtime_image_id,
                "backend": FP8_IDENTITY.backend,
                "selected_layer_ids": list(FP8_IDENTITY.selected_layer_ids),
            },
            "weights": {
                "filename": weights_path.name,
                "sha256": sha256_file(weights_path),
                "size_bytes": weights_path.stat().st_size,
                "tensor_count": len(export_state),
                "contains_target_embedding": False,
                "contains_target_lm_head": False,
            },
            "config": {
                "filename": config_path.name,
                "sha256": sha256_file(config_path),
                "size_bytes": config_path.stat().st_size,
            },
            "scope": "candidate acceptance and serving validation; not a selected production geometry",
        }
        artifact_manifest_path = staging / "draft-export-manifest.json"
        artifact_manifest_path.write_text(
            json.dumps(export_manifest, indent=2, sort_keys=True) + "\n"
        )
        for path in staging.iterdir():
            os.chmod(path, 0o644)
            fd = os.open(path, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
        os.chmod(staging, 0o755)
        fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(fd)
        os.close(fd)
        _publish_no_replace(staging, output)
        fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(fd)
        os.close(fd)
    except Exception:
        # Preserve staged bytes for forensic inspection rather than recursively deleting them.
        raise
    print(json.dumps(export_manifest, sort_keys=True))


if __name__ == "__main__":
    main()
