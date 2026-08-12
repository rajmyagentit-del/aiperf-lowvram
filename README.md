# aiperf-lowvram

[![CI](https://github.com/rajmyagentit-del/aiperf-lowvram/actions/workflows/ci.yml/badge.svg)](https://github.com/rajmyagentit-del/aiperf-lowvram/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Plugin for AIPerf](https://img.shields.io/badge/plugin%20for-ai--dynamo%2Faiperf-76b900)](https://github.com/ai-dynamo/aiperf)
[![Live Demo](https://img.shields.io/badge/demo-live%20interactive-brightgreen)](https://rajmyagentit-del.github.io/aiperf-lowvram/demo.html)

> **A Python plugin for [AIPerf](https://github.com/ai-dynamo/aiperf) — NVIDIA's open-source LLM benchmarking tool —
> that makes it safe and reproducible to run on free-tier hardware.**

🔴 **[Try the live interactive demo →](https://rajmyagentit-del.github.io/aiperf-lowvram/demo.html)**
Select any GPU, configure a benchmark, and see safety guards + provenance output in real time. No install needed.

---

## Why this exists

NVIDIA's AIPerf is an excellent benchmarking tool — but it is built and
tested on H100s and B200s. The largest group of people who actually run it
are students and researchers on free Google Colab T4 GPUs.

When you run AIPerf on a T4, three things go wrong silently:

**1. It crashes with no explanation.**
A concurrency of 10 on a 7B model needs ~5 GiB of KV cache on top of the
model weights. On a 15 GiB T4, that is an OOM. AIPerf hangs indefinitely
with no error message — a known issue documented in their README with no fix.

**2. Your numbers are wrong and you don't know it.**
Request FP8 on a T4 and you get weight-only emulation, not hardware FP8.
Request BF16 and it silently casts to FP16. Nothing warns you. Your
benchmark result looks identical to a real FP8 or BF16 run.

**3. Nobody can reproduce your results.**
A standard benchmark JSON looks like `{"ttft_ms_avg": 142.3}`.
There is no record of whether that came from a T4 or an H100.
Six months later, nobody — including you — can tell.

This plugin fixes all three.

---

## What it does

| Module | What it does | Why it matters |
|---|---|---|
| `gpu.py` | Detects GPU name, compute capability, VRAM, architecture | Foundation for all safety decisions |
| `guards.py` | Validates benchmark config before the run starts | Prevents OOM crashes and misleading numbers |
| `provenance.py` | Wraps every result in a hardware context envelope | Makes results permanently reproducible |
| `report.py` | Renders a human-readable terminal report | Hardware context impossible to miss |
| `prompt_generator.py` | Generates synthetic prompts at exact ISL targets | Matches AIPerf's own benchmarking methodology |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                aiperf-lowvram                   │
│                                                 │
│  gpu.py             → detect hardware           │
│  guards.py          → validate config           │
│  provenance.py      → tag every result          │
│  report.py          → render output             │
│  prompt_generator.py → synthetic ISL prompts    │
└──────────────────┬──────────────────────────────┘
                   │ depends on as a library
                   │ (not a fork, no copied code)
                   ▼
┌─────────────────────────────────────────────────┐
│          ai-dynamo/aiperf  (NVIDIA)             │
│          Apache-2.0                             │
└─────────────────────────────────────────────────┘
```

This is a plugin, not a fork. Every line of code is original.
It depends on AIPerf the same way you depend on `requests`.

---

## Quick start

```python
from aiperf_lowvram.gpu import detect_gpu
from aiperf_lowvram.guards import BenchmarkConfig, run_guards
from aiperf_lowvram.provenance import wrap_result
from aiperf_lowvram.report import print_report

# Step 1 — detect your hardware
profile = detect_gpu()
print(f"GPU: {profile.name} ({profile.architecture})")
print(f"VRAM: {profile.total_memory_gib:.1f} GiB")
print(f"BF16 native: {profile.supports_bfloat16}")
print(f"FP8 native:  {profile.supports_native_fp8}")

# Step 2 — validate your config BEFORE running
config = BenchmarkConfig(
    concurrency=4,
    input_sequence_length=512,
    output_sequence_length=256,
    model_size_billions=0.5,
)
guard = run_guards(profile, config)
guard.print_summary()
# Output example on T4:
# ⚠️  FP8 requested but Tesla T4 has no native FP8 units.
# 💡 Suggested safe concurrency: 8

# Step 3 — wrap your results with hardware provenance
envelope = wrap_result({
    "ttft_ms_avg": 142.3,
    "ttft_ms_p99": 198.7,
    "itl_ms_avg": 18.1,
    "throughput_tps": 312.4,
    "concurrency": 4,
})
print_report(envelope)
envelope.save("result_t4.json")
```

### What the report looks like

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
    Throughput (tok/s) : 312.4
    Concurrency        : 4

  Timestamp : 2026-08-12T10:22:31Z
  Plugin    : aiperf-lowvram v0.1.0
══════════════════════════════════════════════════════════════
```

### What the saved JSON looks like

```json
{
  "plugin": "aiperf-lowvram",
  "hardware": {
    "gpu_name": "Tesla T4",
    "architecture": "Turing",
    "compute_capability": "sm_75",
    "total_memory_gib": 14.07,
    "supports_bfloat16": false,
    "supports_native_fp8": false
  },
  "result": {
    "ttft_ms_avg": 142.3,
    "itl_ms_avg": 18.1
  }
}
```

A T4 result and an H100 result are **never silently comparable.**

---

## Installation

```bash
# Install directly from GitHub
pip install git+https://github.com/rajmyagentit-del/aiperf-lowvram.git
```

For development and contributing:

```bash
git clone https://github.com/rajmyagentit-del/aiperf-lowvram.git
cd aiperf-lowvram
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Run the benchmark on Google Colab (free T4)

No GPU? No problem. Google Colab gives everyone a free T4.

1. Open [Google Colab](https://colab.research.google.com)
2. Runtime → Change runtime type → **T4 GPU**
3. Open `notebooks/t4_benchmark.ipynb` from this repo
4. Run all cells — the notebook handles everything:
   - Detects your hardware automatically
   - Runs safety guards
   - Serves Qwen2.5-0.5B with vLLM
   - Sweeps concurrency 1 → 2 → 4 → 8
   - Measures TTFT and ITL at each level
   - Saves provenance-tagged JSON + plots

```python
# Quick start in Colab — paste this in a cell
!pip install git+https://github.com/rajmyagentit-del/aiperf-lowvram.git vllm

from aiperf_lowvram.gpu import detect_gpu
from aiperf_lowvram.guards import BenchmarkConfig, run_guards

profile = detect_gpu()
print(f"Detected: {profile.name} ({profile.architecture})")
print(f"Safe max concurrency for 0.5B model: {profile.safe_max_concurrency(0.5)}")
```

---

## Hardware compatibility matrix

| Architecture | Examples | Compute Cap | BF16 | FP8 | Notes |
|---|---|---|---|---|---|
| Turing | **T4**, RTX 2080 | sm_75 | ❌ | ❌ | Free Colab GPU — this plugin's focus |
| Ampere | A100, RTX 3090 | sm_80/86 | ✅ | ❌ | Most common cloud GPU |
| Ada Lovelace | RTX 4090, L40S | sm_89 | ✅ | ✅ | Consumer flagship |
| Hopper | H100, H200 | sm_90 | ✅ | ✅ | NVIDIA data center |
| Blackwell | B200, GB200 | sm_100 | ✅ | ✅ | FP4 native |

---

## Metric definitions

Understanding these is essential for interpreting benchmark results correctly.

| Metric | Definition | Common mistake |
|---|---|---|
| **TTFT** | Time To First Token — from request sent to first token received | Measures prefill speed |
| **ITL** | Inter-Token Latency — average time between consecutive tokens | Measures decode speed |
| **TPOT** | Time Per Output Token — ITL excluding TTFT | Some tools conflate with ITL |
| **Throughput** | Total tokens/second across all concurrent requests | Higher concurrency ≠ higher throughput |
| **Goodput** | Throughput counting only requests meeting a latency SLO | The metric that actually matters for production |

> **Important:** Some tools include TTFT inside ITL. This plugin follows
> AIPerf's convention of keeping them separate. Never aggregate ITL numbers
> from different tools without verifying their definitions match.

---

## Test suite

All 20 tests pass without a real GPU. The hardware layer is mocked —
we test the reasoning logic, not the physical device.

```bash
pytest tests/ -v
# test_t4_architecture PASSED
# test_t4_does_not_support_bfloat16 PASSED
# test_h100_supports_fp8 PASSED
# test_excessive_concurrency_produces_error PASSED
# ... 16 more
```

CI runs on Python 3.10, 3.11, and 3.12 on every push.

---

## Project structure

```
aiperf-lowvram/
├── src/aiperf_lowvram/
│   ├── __init__.py          # Public API
│   ├── gpu.py               # GPU detection and capability matrix
│   ├── guards.py            # Pre-run safety validation
│   ├── provenance.py        # Result envelope with hardware context
│   ├── report.py            # Human-readable terminal output
│   └── prompt_generator.py  # Synthetic prompts at exact ISL targets
├── notebooks/
│   └── t4_benchmark.ipynb   # Full concurrency sweep on Colab T4
├── docs/
│   └── demo.html            # Interactive demo (GitHub Pages)
├── tests/
│   └── test_gpu.py          # 20 tests, no GPU required
├── .github/workflows/
│   └── ci.yml               # CI: Python 3.10, 3.11, 3.12
├── pyproject.toml
├── LICENSE                  # Apache-2.0
└── README.md
```

---

## Roadmap

- [x] GPU detection with full capability matrix
- [x] Pre-run safety guards with actionable warnings
- [x] Hardware provenance tagging on every result
- [x] Synthetic prompt generator matching AIPerf ISL methodology
- [x] Interactive demo (GitHub Pages)
- [ ] Real T4 benchmark results committed to `results/`
- [ ] Matplotlib concurrency sweep plots
- [ ] SGLang engine support alongside vLLM
- [ ] Auto-detect model size from HuggingFace model card
- [ ] CSV export for spreadsheet analysis
- [ ] Upstream contribution to ai-dynamo/aiperf

---

## Contributing

Issues and PRs welcome.

This project follows the same DCO sign-off convention as
[ai-dynamo/aiperf](https://github.com/ai-dynamo/aiperf).
Add `-s` to every commit:

```bash
git commit -s -m "your message here"
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

This plugin depends on [ai-dynamo/aiperf](https://github.com/ai-dynamo/aiperf)
(Apache-2.0, © NVIDIA Corporation) as an external library dependency.
It is not a fork and contains no copied code from that project.

---

## Acknowledgements

Built to address real friction encountered running
[ai-dynamo/aiperf](https://github.com/ai-dynamo/aiperf)
on free-tier Colab hardware.

The known issue — *"Startup errors caused by invalid configuration settings
can cause AIPerf to hang indefinitely"* — has no upstream fix for
pre-Ampere hardware. This plugin is that fix.
