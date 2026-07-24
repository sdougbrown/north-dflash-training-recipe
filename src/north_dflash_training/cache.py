"""Feature-cache size estimates; no feature materialization is performed."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeatureCacheEstimate:
    num_sequences: int
    sequence_length: int
    selected_layers: int
    hidden_size: int
    dtype_bytes: int
    batch_size: int
    ring_buffer_tokens: int
    feature_bytes_per_token: int
    disk_cache_bytes: int
    online_peak_bytes: int
    ring_buffer_bytes: int

    @property
    def disk_to_ring_ratio(self) -> float:
        return self.disk_cache_bytes / self.ring_buffer_bytes if self.ring_buffer_bytes else float("inf")

    def to_dict(self) -> dict[str, int | float]:
        result = asdict(self)
        result["disk_to_ring_ratio"] = self.disk_to_ring_ratio
        return result


def estimate_feature_cache(
    *,
    num_sequences: int,
    sequence_length: int,
    selected_layers: int = 5,
    hidden_size: int = 2048,
    dtype_bytes: int = 2,
    batch_size: int = 1,
    ring_buffer_tokens: int = 512,
) -> FeatureCacheEstimate:
    """Estimate raw selected-layer feature storage.

    ``disk_cache_bytes`` models offline precomputation for every token in the
    dataset. ``online_peak_bytes`` is the working set for a batch if one whole
    sequence's selected features is retained. ``ring_buffer_bytes`` models a
    bounded streaming window. The estimate excludes model weights, optimizer
    state, activations, serialization overhead, and projection storage; it is
    intentionally a lower-bound comparison showing why an unbounded offline
    cache does not scale.
    """
    values = {
        "num_sequences": num_sequences,
        "sequence_length": sequence_length,
        "selected_layers": selected_layers,
        "hidden_size": hidden_size,
        "dtype_bytes": dtype_bytes,
        "batch_size": batch_size,
        "ring_buffer_tokens": ring_buffer_tokens,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if selected_layers == 0 or hidden_size == 0 or dtype_bytes == 0 or batch_size == 0:
        raise ValueError("selected_layers, hidden_size, dtype_bytes, and batch_size must be positive")
    if sequence_length == 0:
        raise ValueError("sequence_length must be positive")

    per_token = selected_layers * hidden_size * dtype_bytes
    return FeatureCacheEstimate(
        **values,
        feature_bytes_per_token=per_token,
        disk_cache_bytes=num_sequences * sequence_length * per_token,
        online_peak_bytes=batch_size * sequence_length * per_token,
        ring_buffer_bytes=batch_size * min(sequence_length, ring_buffer_tokens) * per_token,
    )


def format_bytes(value: int | float) -> str:
    """Human-readable binary units for CLI output."""
    value = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    unit = units[0]
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} {unit}"
