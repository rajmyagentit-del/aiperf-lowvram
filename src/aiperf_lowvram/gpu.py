"""GPU capability detection for benchmark configuration guards.

Detection is intentionally best-effort and never raises: benchmarking tools
must degrade gracefully to 'unknown hardware' rather than crash the run.
This module encodes the hardware compatibility matrix for modern LLM
inference kernels so that benchmark results carry accurate provenance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Ampere (sm_80) introduced bfloat16 and is the floor for most modern
# inference kernels. Pre-Ampere devices silently fall back to float16,
# which changes what a benchmark number actually means.
BFLOAT16_MIN_CAPABILITY = (8, 0)

# Hopper (sm_90) and Ada (sm_89) have hardware FP8 units.
# Everything below emulates FP8 as weight-only quantization,
# producing numbers that are not comparable to native FP8 results.
FP8_NATIVE_CAPABILITIES = frozenset({(8, 9), (9, 0)})

# Blackwell (sm_100) introduced FP4 native support.
FP4_NATIVE_CAPABILITIES = frozenset({(10, 0)})


@dataclass(frozen=True)
class GpuProfile:
    """Immutable snapshot of the GPU a benchmark is running on.

    frozen=True because hardware does not change mid-run.
    Mutating a profile mid-benchmark would silently corrupt provenance.
    """

    name: str
    compute_capability: tuple[int, int] | None
    total_memory_bytes: int | None

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    @property
    def total_memory_gib(self) -> float | None:
        """Total VRAM in gibibytes, or None if detection failed."""
        if self.total_memory_bytes is None:
            return None
        return self.total_memory_bytes / (1024**3)

    # ------------------------------------------------------------------
    # Precision support flags
    # ------------------------------------------------------------------

    @property
    def supports_bfloat16(self) -> bool:
        """True on Ampere and newer (sm_80+).

        Pre-Ampere GPUs silently cast bf16 tensors to fp16,
        which can cause misleading benchmark numbers for models
        that assume bf16 accumulation.
        """
        if self.compute_capability is None:
            return False
        return self.compute_capability >= BFLOAT16_MIN_CAPABILITY

    @property
    def supports_native_fp8(self) -> bool:
        """True only on Ada (sm_89) and Hopper (sm_90).

        On all other hardware FP8 is weight-only emulation,
        not hardware-accelerated. A T4 FP8 number and an H100
        FP8 number are measuring fundamentally different things.
        """
        return self.compute_capability in FP8_NATIVE_CAPABILITIES

    @property
    def supports_native_fp4(self) -> bool:
        """True only on Blackwell (sm_100+)."""
        return self.compute_capability in FP4_NATIVE_CAPABILITIES

    # ------------------------------------------------------------------
    # Architecture identification
    # ------------------------------------------------------------------

    @property
    def architecture(self) -> str:
        """Human-readable GPU architecture family.

        Used in the provenance block that ships alongside every
        benchmark result, so a T4 number is never mistaken for
        an H100 number by a downstream consumer of the JSON.
        """
        match self.compute_capability:
            case None:
                return "unknown"
            case (7, 0) | (7, 2):
                return "Volta"
            case (7, 5):
                return "Turing"
            case (8, 0) | (8, 6) | (8, 7):
                return "Ampere"
            case (8, 9):
                return "Ada Lovelace"
            case (9, 0):
                return "Hopper"
            case (major, _) if major >= 10:
                return "Blackwell or newer"
            case _:
                return "unknown"

    # ------------------------------------------------------------------
    # Concurrency safety guard
    # ------------------------------------------------------------------

    def safe_max_concurrency(self, model_size_b: float = 7.0) -> int:
        """Estimate a safe maximum concurrency for this GPU and model size.

        This is a conservative heuristic, not a guarantee.
        The real ceiling depends on sequence length and KV cache policy,
        but this prevents the most common OOM crashes on free-tier GPUs.

        Args:
            model_size_b: Model parameter count in billions (default 7B).

        Returns:
            Recommended maximum concurrent requests.
        """
        if self.total_memory_gib is None:
            return 1

        # Rough fp16 weight footprint: ~2 bytes per parameter
        weight_gib = model_size_b * 2.0

        # Reserve 20% for activations, KV cache overhead, CUDA context
        usable_gib = self.total_memory_gib * 0.80

        if usable_gib <= weight_gib:
            return 1

        # Each additional concurrent request needs roughly
        # 0.5 GiB for its KV cache slice at default sequence lengths
        kv_overhead_per_request_gib = 0.5
        remaining = usable_gib - weight_gib
        return max(1, int(remaining / kv_overhead_per_request_gib))

    # ------------------------------------------------------------------
    # Provenance dict — goes into every benchmark result JSON
    # ------------------------------------------------------------------

    def provenance_dict(self) -> dict:
        """Return a dict that uniquely identifies the hardware context.

        Every benchmark result produced by this plugin includes this
        dict so results from different hardware are never silently
        aggregated or compared without context.
        """
        cc = self.compute_capability
        return {
            "gpu_name": self.name,
            "architecture": self.architecture,
            "compute_capability": f"sm_{cc[0]}{cc[1]}" if cc else "unknown",
            "total_memory_gib": round(self.total_memory_gib, 2)
            if self.total_memory_gib
            else None,
            "supports_bfloat16": self.supports_bfloat16,
            "supports_native_fp8": self.supports_native_fp8,
            "supports_native_fp4": self.supports_native_fp4,
        }


# Sentinel used when detection fails entirely
UNKNOWN_GPU = GpuProfile(
    name="unknown",
    compute_capability=None,
    total_memory_bytes=None,
)


def detect_gpu(device_index: int = 0) -> GpuProfile:
    """Detect the GPU at device_index and return its profile.

    Never raises. Returns UNKNOWN_GPU on any failure so the
    benchmark run continues with degraded provenance rather
    than crashing before a single request is sent.

    Args:
        device_index: CUDA device index (0 for the first/only GPU).

    Returns:
        GpuProfile for the detected device, or UNKNOWN_GPU.
    """
    try:
        import torch
    except ImportError:
        logger.debug("torch not installed; GPU detection unavailable")
        return UNKNOWN_GPU

    if not torch.cuda.is_available():
        logger.debug("No CUDA device visible to torch")
        return UNKNOWN_GPU

    try:
        props = torch.cuda.get_device_properties(device_index)
        return GpuProfile(
            name=props.name,
            compute_capability=(props.major, props.minor),
            total_memory_bytes=props.total_memory,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("GPU detection failed for device %d: %s", device_index, exc)
        return UNKNOWN_GPU
