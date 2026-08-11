"""Configuration safety guards for low-VRAM and pre-Ampere GPUs.

Before a benchmark run starts, these guards inspect the requested
configuration and emit warnings or raise errors when the config
would cause OOM crashes, silent metric corruption, or misleading
results on constrained hardware.

Design principle: warn early, explain clearly, never silently proceed
with a configuration that will produce untrustworthy numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from aiperf_lowvram.gpu import GpuProfile

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Subset of benchmark parameters that affect hardware safety."""

    concurrency: int
    input_sequence_length: int
    output_sequence_length: int
    model_size_billions: float
    use_fp8: bool = False
    use_bf16: bool = False


@dataclass
class GuardResult:
    """Result of running all guards against a config + GPU profile."""

    safe: bool
    warnings: list[str]
    errors: list[str]
    adjusted_concurrency: int | None = None

    def log_all(self) -> None:
        """Emit all warnings and errors to the logger."""
        for w in self.warnings:
            logger.warning("[aiperf-lowvram] %s", w)
        for e in self.errors:
            logger.error("[aiperf-lowvram] UNSAFE: %s", e)

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        if self.safe and not self.warnings:
            print("✅ Hardware guard: configuration looks safe for this GPU.")
            return
        print("\n── aiperf-lowvram hardware guard ──────────────────────────")
        for w in self.warnings:
            print(f"  ⚠️  {w}")
        for e in self.errors:
            print(f"  ❌  {e}")
        if self.adjusted_concurrency is not None:
            print(f"  💡 Suggested safe concurrency: {self.adjusted_concurrency}")
        print("────────────────────────────────────────────────────────────\n")


def run_guards(profile: GpuProfile, config: BenchmarkConfig) -> GuardResult:
    """Run all hardware safety guards and return a consolidated result."""
    guard_warnings: list[str] = []
    errors: list[str] = []

    # ── Guard 1: FP8 on non-native hardware ─────────────────────────
    if config.use_fp8 and not profile.supports_native_fp8:
        guard_warnings.append(
            f"FP8 requested but {profile.name} ({profile.architecture}) "
            f"does not have native FP8 units. Falls back to weight-only "
            f"emulation. Numbers are NOT comparable to Hopper/Ada results."
        )

    # ── Guard 2: BF16 on pre-Ampere hardware ────────────────────────
    if config.use_bf16 and not profile.supports_bfloat16:
        guard_warnings.append(
            f"BF16 requested but {profile.name} ({profile.architecture}) "
            f"pre-dates Ampere (sm_80). BF16 tensors will be cast to FP16."
        )

    # ── Guard 3: Concurrency vs VRAM ────────────────────────────────
    safe_concurrency = profile.safe_max_concurrency(config.model_size_billions)
    if config.concurrency > safe_concurrency:
        mem = f"{profile.total_memory_gib:.1f} GiB" if profile.total_memory_gib else "unknown"
        errors.append(
            f"Requested concurrency {config.concurrency} likely exceeds "
            f"available VRAM on {profile.name} ({mem}) "
            f"for a {config.model_size_billions}B model. "
            f"Expected OOM. Recommended max: {safe_concurrency}."
        )

    # ── Guard 4: Sequence length sanity on small VRAM ───────────────
    total_seq = config.input_sequence_length + config.output_sequence_length
    if profile.total_memory_gib is not None and profile.total_memory_gib < 20 and total_seq > 4096:
        guard_warnings.append(
            f"Total sequence length {total_seq} tokens on a GPU with "
            f"{profile.total_memory_gib:.1f} GiB VRAM. "
            f"Consider --isl 512 --osl 256 for initial sweeps."
        )

    # ── Guard 5: Unknown GPU ─────────────────────────────────────────
    if profile.compute_capability is None:
        guard_warnings.append(
            "GPU could not be detected. Running in degraded provenance mode. "
            "Do not compare these numbers with results from identified hardware."
        )

    safe = len(errors) == 0
    adjusted = safe_concurrency if not safe else None
    return GuardResult(
        safe=safe,
        warnings=guard_warnings,
        errors=errors,
        adjusted_concurrency=adjusted,
    )
