"""Result provenance tagging.

Every benchmark result produced with this plugin is wrapped in a
ProvenanceEnvelope that binds the result to the exact hardware it
was measured on. This prevents the most common benchmarking mistake:
aggregating or comparing numbers from different hardware classes
without realising it.
"""

from __future__ import annotations

import datetime
import json
import platform
import sys
from dataclasses import dataclass, field
from typing import Any

from aiperf_lowvram.gpu import GpuProfile, detect_gpu


@dataclass
class ProvenanceEnvelope:
    """A benchmark result bound to its hardware and software context."""

    result: dict[str, Any]
    gpu: GpuProfile
    timestamp_utc: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    python_version: str = field(
        default_factory=lambda: sys.version.split()[0]
    )
    plugin_version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the envelope to a flat dict ready for JSON export."""
        return {
            "plugin": "aiperf-lowvram",
            "plugin_version": self.plugin_version,
            "timestamp_utc": self.timestamp_utc,
            "environment": {
                "python_version": self.python_version,
                "platform": platform.platform(),
            },
            "hardware": self.gpu.provenance_dict(),
            "result": self.result,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return the envelope as a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        """Write the envelope to a JSON file at path."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())
        print(f"Provenance-tagged result saved to {path}")


def wrap_result(
    result: dict[str, Any],
    device_index: int = 0,
) -> ProvenanceEnvelope:
    """Convenience function: detect GPU and wrap a result dict.

    Args:
        result: Raw benchmark result dictionary.
        device_index: CUDA device index (default 0).

    Returns:
        ProvenanceEnvelope binding the result to detected hardware.

    Example:
        >>> from aiperf_lowvram.provenance import wrap_result
        >>> envelope = wrap_result({"ttft_ms_avg": 142.3, "itl_ms_avg": 18.1})
        >>> print(envelope.to_json())
    """
    gpu = detect_gpu(device_index)
    return ProvenanceEnvelope(result=result, gpu=gpu)
