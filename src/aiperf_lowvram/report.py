"""Human-readable benchmark report generator.

Takes a ProvenanceEnvelope and renders a clean terminal report
that makes hardware context impossible to miss. Designed for
pasting into GitHub issues, Slack, or a README.
"""

from __future__ import annotations

from aiperf_lowvram.provenance import ProvenanceEnvelope


_SEPARATOR = "═" * 62


def render_report(envelope: ProvenanceEnvelope) -> str:
    """Render a full benchmark report as a plain-text string.

    Args:
        envelope: A ProvenanceEnvelope from provenance.wrap_result().

    Returns:
        Multi-line string ready for print() or file write.
    """
    hw = envelope.gpu
    result = envelope.result
    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────
    lines.append(_SEPARATOR)
    lines.append("  aiperf-lowvram  │  Benchmark Report")
    lines.append(_SEPARATOR)

    # ── Hardware block ───────────────────────────────────────────────
    lines.append("  HARDWARE")
    lines.append(f"    GPU          : {hw.name}")
    lines.append(f"    Architecture : {hw.architecture}")
    cc = hw.compute_capability
    lines.append(
        f"    Compute cap  : sm_{cc[0]}{cc[1]}" if cc else
        "    Compute cap  : unknown"
    )
    lines.append(
        f"    VRAM         : {hw.total_memory_gib:.1f} GiB"
        if hw.total_memory_gib else
        "    VRAM         : unknown"
    )
    lines.append(
        f"    BF16 native  : {'yes' if hw.supports_bfloat16 else 'NO (pre-Ampere)'}"
    )
    lines.append(
        f"    FP8 native   : {'yes' if hw.supports_native_fp8 else 'NO'}"
    )
    lines.append("")

    # ── Metrics block ────────────────────────────────────────────────
    lines.append("  METRICS")
    metric_keys = [
        ("ttft_ms_avg",    "TTFT avg (ms)      "),
        ("ttft_ms_p99",    "TTFT p99 (ms)      "),
        ("itl_ms_avg",     "ITL avg (ms)       "),
        ("itl_ms_p99",     "ITL p99 (ms)       "),
        ("throughput_tps", "Throughput (tok/s) "),
        ("concurrency",    "Concurrency        "),
        ("request_count",  "Request count      "),
    ]
    for key, label in metric_keys:
        val = result.get(key)
        if val is not None:
            lines.append(f"    {label}: {val}")

    # ── Any extra keys the caller put in result ──────────────────────
    known_keys = {k for k, _ in metric_keys}
    extras = {k: v for k, v in result.items() if k not in known_keys}
    if extras:
        lines.append("")
        lines.append("  ADDITIONAL")
        for k, v in extras.items():
            lines.append(f"    {k:<30}: {v}")

    # ── Footer ───────────────────────────────────────────────────────
    lines.append("")
    lines.append(f"  Timestamp : {envelope.timestamp_utc}")
    lines.append(f"  Plugin    : aiperf-lowvram v{envelope.plugin_version}")
    lines.append(_SEPARATOR)

    return "\n".join(lines)


def print_report(envelope: ProvenanceEnvelope) -> None:
    """Print a benchmark report to stdout."""
    print(render_report(envelope))
