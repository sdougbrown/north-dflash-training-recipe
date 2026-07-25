"""Orchestration-owned lifecycle for one transient connector handoff.

``feature_stream`` deliberately never deletes files. This module is the narrow
orchestration boundary that fingerprints a completed connector artifact, clones
it into the bounded ring, and releases only that exact file after the matching
optimizer result has acknowledged the request. It provides in-process ordering;
a durable crash/restart commit ledger remains a separate pilot requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import stat

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional runtime dependency.
    raise ImportError(
        "Connector lifecycle requires PyTorch; install "
        "north-dflash-training-scaffold[runtime]."
    ) from exc

from .feature_stream import (
    BoundedFeatureRing,
    TeacherRuntimeIdentity,
    load_connector_feature_batch,
)
from .online_step import BoundedOptimizerStepResult


@dataclass(frozen=True)
class ConnectorFileFingerprint:
    """Identity of one immutable regular file at handoff time."""

    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class OwnedConnectorHandoff:
    """Exact artifact retained while its cloned ring request is active."""

    request_id: str
    runtime_identity: TeacherRuntimeIdentity
    feature_path: Path
    feature_fingerprint: ConnectorFileFingerprint
    lock_path: Path
    lock_fingerprint: ConnectorFileFingerprint | None
    feature_bytes_owned: int


@dataclass(frozen=True)
class ConnectorReleaseResult:
    """Auditable release result containing no feature values or model state."""

    request_id: str
    feature_path: str
    feature_file_bytes_released: int
    feature_tensor_bytes_consumed: int
    feature_sha256: str
    lock_file_released: bool


def _fingerprint_regular_file(path: Path) -> ConnectorFileFingerprint:
    if path.is_symlink():
        raise ValueError("connector lifecycle refuses symbolic links")
    try:
        metadata = path.stat()
    except FileNotFoundError as exc:
        raise ValueError("connector artifact must exist") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("connector artifact must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return ConnectorFileFingerprint(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size_bytes=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        sha256=digest.hexdigest(),
    )


def _fingerprint_optional_regular_file(path: Path) -> ConnectorFileFingerprint | None:
    if path.is_symlink():
        raise ValueError("connector lifecycle refuses symbolic links")
    return _fingerprint_regular_file(path) if path.exists() else None


def ingest_connector_handoff(
    path: str | Path,
    *,
    request_id: str,
    runtime_identity: TeacherRuntimeIdentity,
    expected_token_ids: torch.Tensor,
    ring: BoundedFeatureRing,
) -> OwnedConnectorHandoff:
    """Clone one stable connector file into ``ring`` without deleting it.

    Fingerprinting before and after tensor loading rejects a file that changed
    while ownership was being transferred. Ring overflow or validation failure
    leaves both the feature and adjacent ``.lock`` file untouched.
    """
    source = Path(path).absolute()
    lock_path = Path(f"{source}.lock")
    before = _fingerprint_regular_file(source)
    lock_before = _fingerprint_optional_regular_file(lock_path)
    streamed = load_connector_feature_batch(
        source,
        request_id=request_id,
        runtime_identity=runtime_identity,
        expected_token_ids=expected_token_ids,
    )
    after = _fingerprint_regular_file(source)
    lock_after = _fingerprint_optional_regular_file(lock_path)
    if after != before or lock_after != lock_before:
        raise RuntimeError("connector artifact changed while ring ownership was acquired")
    ring.put(streamed)
    return OwnedConnectorHandoff(
        request_id=request_id,
        runtime_identity=runtime_identity,
        feature_path=source,
        feature_fingerprint=after,
        lock_path=lock_path,
        lock_fingerprint=lock_after,
        feature_bytes_owned=streamed.feature_bytes,
    )


def release_connector_after_optimizer_step(
    handoff: OwnedConnectorHandoff,
    *,
    optimizer_result: BoundedOptimizerStepResult,
    ring: BoundedFeatureRing,
) -> ConnectorReleaseResult:
    """Delete exactly one unchanged handoff after its matching optimizer commit.

    ``run_one_bounded_optimizer_step`` acknowledges the ring only after a
    successful update. This function therefore rejects release while the
    request remains active, on request/result drift, or if either on-disk file
    changed after ingestion.
    """
    if optimizer_result.request_id != handoff.request_id:
        raise ValueError("optimizer result does not match connector request_id")
    if ring.runtime_identity != handoff.runtime_identity:
        raise ValueError("ring runtime identity does not match connector handoff")
    if ring.contains_request(handoff.request_id):
        raise RuntimeError("connector request is still active in the feature ring")
    if optimizer_result.feature_bytes_released != handoff.feature_bytes_owned:
        raise ValueError("optimizer result feature-byte count does not match handoff")
    if (
        optimizer_result.updated_parameter_tensors < 1
        or optimizer_result.active_label_count < 1
        or not math.isfinite(optimizer_result.loss)
        or not math.isfinite(optimizer_result.gradient_norm)
    ):
        raise ValueError("optimizer result does not prove a finite committed update")

    current = _fingerprint_regular_file(handoff.feature_path)
    if current != handoff.feature_fingerprint:
        raise RuntimeError("connector feature file changed after ring ingestion")
    current_lock = _fingerprint_optional_regular_file(handoff.lock_path)
    lock_exists = current_lock is not None
    if lock_exists != (handoff.lock_fingerprint is not None):
        raise RuntimeError("connector lock-file presence changed after ring ingestion")
    if current_lock != handoff.lock_fingerprint:
        raise RuntimeError("connector lock file changed after ring ingestion")

    handoff.feature_path.unlink()
    if lock_exists:
        handoff.lock_path.unlink()
    return ConnectorReleaseResult(
        request_id=handoff.request_id,
        feature_path=str(handoff.feature_path),
        feature_file_bytes_released=current.size_bytes,
        feature_tensor_bytes_consumed=handoff.feature_bytes_owned,
        feature_sha256=current.sha256,
        lock_file_released=lock_exists,
    )
