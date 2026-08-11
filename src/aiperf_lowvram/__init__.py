"""aiperf-lowvram: Hardware-aware benchmarking guards for AIPerf.

Designed for LLM inference benchmarking on low-VRAM and pre-Ampere GPUs,
particularly the free-tier Tesla T4 available via Google Colab.

Quick start:
    from aiperf_lowvram.gpu import detect_gpu
    from aiperf_lowvram.guards import BenchmarkConfig, run_guards
    from aiperf_lowvram.provenance import wrap_result
    from aiperf_lowvram.report import print_report

    profile = detect_gpu()
    config = BenchmarkConfig(
        concurrency=4,
        input_sequence_length=512,
        output_sequence_length=256,
        model_size_billions=0.5,
    )
    result = run_guards(profile, config)
    result.print_summary()
"""

__version__ = "0.1.0"
__all__ = [
    "BenchmarkConfig",
    "GpuProfile",
    "GuardResult",
    "ProvenanceEnvelope",
    "detect_gpu",
    "print_report",
    "render_report",
    "run_guards",
    "wrap_result",
]
