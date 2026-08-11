# aiperf-lowvram

[![CI](https://github.com/rajmyagentit-del/aiperf-lowvram/actions/workflows/ci.yml/badge.svg)](https://github.com/rajmyagentit-del/aiperf-lowvram/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![AIPerf](https://img.shields.io/badge/plugin%20for-ai--dynamo%2Faiperf-green)](https://github.com/ai-dynamo/aiperf)

Hardware-aware benchmarking guards and provenance tagging for
[AIPerf](https://github.com/ai-dynamo/aiperf) — NVIDIA's open-source
LLM inference benchmarking tool — specifically targeting the
**free-tier Tesla T4** available via Google Colab and other
low-VRAM / pre-Ampere GPU environments.

---

## The Problem

AIPerf is built and tested on H100s and B200s. Run it on a T4 and you hit:

- **OOM crashes** from concurrency configs that assume 80 GiB VRAM
- **Silent metric corruption** when FP8 falls back to weight-only emulation
- **Misleading comparisons** — a T4 result and an H100 result look identical in JSON with no hardware context

This plugin fixes all three before a single benchmark request is sent.

---

## What This Plugin Does

| Feature | Description |
|---|---|
| **GPU detection** | Detects compute capability, VRAM, and architecture family |
| **Safety guards** | Warns before OOM, flags FP8/BF16 fallbacks, caps concurrency |
| **Provenance tagging** | Every result JSON includes the exact hardware it was measured on |
| **Human-readable reports** | Terminal output that makes hardware context impossible to miss |
| **Colab notebook** | One-click reproducible benchmark on the free T4 tier |

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              aiperf-lowvram                 │
│                                             │
│  gpu.py          → detect hardware          │
│  guards.py       → validate config          │
│  provenance.py   → tag results              │
│  report.py       → render output            │
└──────────────┬──────────────────────────────┘
               │ depends on (not a fork)
               ▼
┌─────────────────────────────────────────────┐
│         ai-dynamo/aiperf  (NVIDIA)          │
│         Apache-2.0                          │
└─────────────────────────────────────────────┘
```

---

## Quick Start

```python
from aiperf_lowvram.gpu import detect_gpu
from aiperf_lowvram.guards import BenchmarkConfig, run_guards
from aiperf_lowvram.provenance import wrap_result
from aiperf_lowvram.report import print_report

# 1. Detect your GPU
profile = detect_gpu()
print(f"Running on: {profile.name} ({profile.architecture})")
print(f"VRAM: {profile.total_memory_gib:.1f} GiB")
print(f"BF16 native: {profile.supports_bfloat16}")
print(f"FP8 native:  {profile.supports_native_fp8}")

# 2. Check your config before you run
config = BenchmarkConfig(
    concurrency=4,
    input_sequence_length=512,
    output_sequence_length=256,
    model_size_billions=0.5,
)
guard_result = run_guards(profile, config)
guard_result.print_summary()

# 3. After your benchmark run, wrap the result with hardware provenance
raw_result = {
    "ttft_ms_avg": 142.3,
    "ttft_ms_p99": 198.7,
    "itl_ms_avg": 18.1,
    "itl_ms_p99": 24.6,
    "throughput_tps": 312.4,
    "concurrency": 4,
    "request_count": 100,
}
envelope = wrap_result(raw_result)
print_report(envelope)
envelope.save("result_t4.json")
```

### Example terminal output

```
══════════════════════════════════════════════════════════════
  aiperf-lowvram  │  Benchmark Report
══════════════════════════════════════════════════════════════
  HARDWARE
    GPU          : Tesla T4
    Architecture : Turing
    Compute cap  : sm_75
    VRAM         : 14.1 GiB
    BF16 native  : NO (pre-Ampere)
    FP8 native   : NO

  METRICS
    TTFT avg (ms)      : 142.3
    TTFT p99 (ms)      : 198.7
    ITL avg (ms)       : 18.1
    ITL p99 (ms)       : 24.6
    Throughput (tok/s) : 312.4
    Concurrency        : 4
    Request count      : 100

  Timestamp : 2026-08-11T10:22:31Z
  Plugin    : aiperf-lowvram v0.1.0
══════════════════════════════════════════════════════════════
```

---

## Installation

```bash
pip install git+https://github.com/rajmyagentit-del/aiperf-lowvram.git
```

For development:
```bash
git clone https://github.com/rajmyagentit-del/aiperf-lowvram.git
cd aiperf-lowvram
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Hardware Compatibility Matrix

| Architecture | Examples | sm | BF16 | FP8 native | Notes |
|---|---|---|---|---|---|
| Turing | T4, RTX 2080 | sm_75 | ❌ | ❌ | Free Colab GPU |
| Ampere | A100, RTX 3090 | sm_80/86 | ✅ | ❌ | Common cloud GPU |
| Ada Lovelace | RTX 4090, L40S | sm_89 | ✅ | ✅ | Consumer flagship |
| Hopper | H100, H200 | sm_90 | ✅ | ✅ | Data center |
| Blackwell | B200, GB200 | sm_100 | ✅ | ✅ | FP4 native |

---

## Why Provenance Tagging Matters

Standard benchmark JSON looks like this:

```json
{"ttft_ms_avg": 142.3, "itl_ms_avg": 18.1}
```

This plugin's output looks like this:

```json
{
  "plugin": "aiperf-lowvram",
  "plugin_version": "0.1.0",
  "timestamp_utc": "2026-08-11T10:22:31Z",
  "environment": {
    "python_version": "3.11.9",
    "platform": "Linux-5.15.0-x86_64"
  },
  "hardware": {
    "gpu_name": "Tesla T4",
    "architecture": "Turing",
    "compute_capability": "sm_75",
    "total_memory_gib": 14.07,
    "supports_bfloat16": false,
    "supports_native_fp8": false,
    "supports_native_fp4": false
  },
  "result": {
    "ttft_ms_avg": 142.3,
    "itl_ms_avg": 18.1
  }
}
```

A T4 result and an H100 result are **never silently comparable**.

---

## Running the Full Benchmark on Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Run this in a cell:

```python
!pip install git+https://github.com/rajmyagentit-del/aiperf-lowvram.git
!pip install vllm aiperf

from aiperf_lowvram.gpu import detect_gpu
from aiperf_lowvram.guards import BenchmarkConfig, run_guards

profile = detect_gpu()
print(f"Detected: {profile.name} ({profile.architecture})")

config = BenchmarkConfig(
    concurrency=4,
    input_sequence_length=512,
    output_sequence_length=256,
    model_size_billions=0.5,
)
result = run_guards(profile, config)
result.print_summary()
```

---

## Project Structure

```
aiperf-lowvram/
├── src/aiperf_lowvram/
│   ├── __init__.py       # Public API
│   ├── gpu.py            # GPU detection and capability matrix
│   ├── guards.py         # Pre-run safety validation
│   ├── provenance.py     # Result envelope with hardware context
│   └── report.py         # Human-readable terminal output
├── tests/
│   ├── __init__.py
│   └── test_gpu.py       # Full test suite — no GPU required
├── .github/workflows/
│   └── ci.yml            # CI across Python 3.10, 3.11, 3.12
├── pyproject.toml
├── LICENSE               # Apache-2.0
└── README.md
```

---

## Metric Definitions

| Metric | Definition |
|---|---|
| **TTFT** | Time To First Token — latency from request sent to first token received |
| **ITL** | Inter-Token Latency — average time between consecutive tokens |
| **TPOT** | Time Per Output Token — ITL excluding TTFT |
| **Throughput** | Total tokens generated per second across all concurrent requests |
| **Goodput** | Throughput counting only requests that met a latency SLO |

> **Note:** Different tools measure ITL differently. Some include TTFT inside ITL; this plugin follows AIPerf's convention of keeping them separate. Never aggregate ITL numbers from different tools without checking their definitions.

---

## Roadmap

- [ ] Colab notebook with full concurrency sweep
- [ ] Matplotlib plots of TTFT/ITL vs concurrency
- [ ] SGLang engine support alongside vLLM
- [ ] Automatic model size detection from HuggingFace model card
- [ ] Export to CSV for spreadsheet analysis
- [ ] Integration test against live Ollama endpoint

---

## Contributing

Issues and PRs welcome. Please open an issue before starting large changes.

This project follows the same DCO sign-off convention as
[ai-dynamo/aiperf](https://github.com/ai-dynamo/aiperf) —
add `-s` to your commit command:

```bash
git commit -s -m "your message"
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

This plugin depends on [ai-dynamo/aiperf](https://github.com/ai-dynamo/aiperf)
(Apache-2.0, © NVIDIA Corporation) as an external dependency.
It is not a fork and contains no copied code from that project.

---

## Acknowledgements

Built to address real friction encountered running
[ai-dynamo/aiperf](https://github.com/ai-dynamo/aiperf)
on free-tier Colab hardware. Thanks to the AIPerf maintainers
for building an open benchmarking tool and publishing it under Apache-2.0.
