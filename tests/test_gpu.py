"""Tests for GPU detection and capability logic.

All tests here run WITHOUT a real GPU — they test the logic,
not the hardware. This means CI passes on any machine, including
GitHub Actions free runners which have no GPU at all.

This is the right way to test hardware-detection code: mock the
hardware layer, test the reasoning layer exhaustively.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from aiperf_lowvram.gpu import (
    BFLOAT16_MIN_CAPABILITY,
    FP8_NATIVE_CAPABILITIES,
    UNKNOWN_GPU,
    GpuProfile,
    detect_gpu,
)
from aiperf_lowvram.guards import BenchmarkConfig, run_guards


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def t4_profile() -> GpuProfile:
    """Tesla T4 — Turing sm_75, 15 GiB VRAM. The free Colab GPU."""
    return GpuProfile(
        name="Tesla T4",
        compute_capability=(7, 5),
        total_memory_bytes=15_109_001_216,
    )


@pytest.fixture
def a100_profile() -> GpuProfile:
    """A100 SXM — Ampere sm_80, 80 GiB VRAM."""
    return GpuProfile(
        name="NVIDIA A100-SXM4-80GB",
        compute_capability=(8, 0),
        total_memory_bytes=85_899_345_920,
    )


@pytest.fixture
def h100_profile() -> GpuProfile:
    """H100 SXM — Hopper sm_90, 80 GiB VRAM."""
    return GpuProfile(
        name="NVIDIA H100 SXM5 80GB",
        compute_capability=(9, 0),
        total_memory_bytes=85_899_345_920,
    )


# ── GpuProfile tests ────────────────────────────────────────────────

class TestGpuProfile:
    def test_t4_architecture(self, t4_profile):
        assert t4_profile.architecture == "Turing"

    def test_a100_architecture(self, a100_profile):
        assert a100_profile.architecture == "Ampere"

    def test_h100_architecture(self, h100_profile):
        assert h100_profile.architecture == "Hopper"

    def test_t4_does_not_support_bfloat16(self, t4_profile):
        assert t4_profile.supports_bfloat16 is False

    def test_a100_supports_bfloat16(self, a100_profile):
        assert a100_profile.supports_bfloat16 is True

    def test_t4_does_not_support_fp8(self, t4_profile):
        assert t4_profile.supports_native_fp8 is False

    def test_h100_supports_fp8(self, h100_profile):
        assert h100_profile.supports_native_fp8 is True

    def test_t4_memory_gib(self, t4_profile):
        assert t4_profile.total_memory_gib == pytest.approx(14.07, abs=0.1)

    def test_unknown_gpu_safe_defaults(self):
        assert UNKNOWN_GPU.architecture == "unknown"
        assert UNKNOWN_GPU.supports_bfloat16 is False
        assert UNKNOWN_GPU.supports_native_fp8 is False
        assert UNKNOWN_GPU.total_memory_gib is None

    def test_provenance_dict_keys(self, t4_profile):
        d = t4_profile.provenance_dict()
        assert "gpu_name" in d
        assert "architecture" in d
        assert "compute_capability" in d
        assert d["compute_capability"] == "sm_75"
        assert d["supports_bfloat16"] is False

    def test_safe_max_concurrency_t4_small_model(self, t4_profile):
        """0.5B model on T4 should allow more than 1 concurrent request."""
        n = t4_profile.safe_max_concurrency(model_size_b=0.5)
        assert n > 1

    def test_safe_max_concurrency_t4_large_model(self, t4_profile):
        """13B model on a 15 GiB T4 should return 1 (barely fits)."""
        n = t4_profile.safe_max_concurrency(model_size_b=13.0)
        assert n == 1


# ── Guard tests ─────────────────────────────────────────────────────

class TestGuards:
    def test_safe_config_produces_no_errors(self, t4_profile):
        config = BenchmarkConfig(
            concurrency=1,
            input_sequence_length=256,
            output_sequence_length=128,
            model_size_billions=0.5,
        )
        result = run_guards(t4_profile, config)
        assert result.safe is True
        assert result.errors == []

    def test_fp8_on_t4_produces_warning(self, t4_profile):
        config = BenchmarkConfig(
            concurrency=1,
            input_sequence_length=256,
            output_sequence_length=128,
            model_size_billions=0.5,
            use_fp8=True,
        )
        result = run_guards(t4_profile, config)
        assert any("FP8" in w for w in result.warnings)

    def test_bf16_on_t4_produces_warning(self, t4_profile):
        config = BenchmarkConfig(
            concurrency=1,
            input_sequence_length=256,
            output_sequence_length=128,
            model_size_billions=0.5,
            use_bf16=True,
        )
        result = run_guards(t4_profile, config)
        assert any("BF16" in w for w in result.warnings)

    def test_excessive_concurrency_produces_error(self, t4_profile):
        config = BenchmarkConfig(
            concurrency=100,
            input_sequence_length=512,
            output_sequence_length=256,
            model_size_billions=7.0,
        )
        result = run_guards(t4_profile, config)
        assert result.safe is False
        assert result.adjusted_concurrency is not None

    def test_long_sequence_on_small_vram_warns(self, t4_profile):
        config = BenchmarkConfig(
            concurrency=1,
            input_sequence_length=4000,
            output_sequence_length=1000,
            model_size_billions=0.5,
        )
        result = run_guards(t4_profile, config)
        assert any("sequence" in w.lower() for w in result.warnings)


# ── detect_gpu tests (mocked) ────────────────────────────────────────

class TestDetectGpu:
    def test_returns_unknown_when_torch_missing(self):
        with patch.dict("sys.modules", {"torch": None}):
            profile = detect_gpu()
        assert profile == UNKNOWN_GPU

    def test_returns_unknown_when_cuda_unavailable(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            profile = detect_gpu()
        assert profile == UNKNOWN_GPU

    def test_detects_gpu_properties(self):
        mock_props = MagicMock()
        mock_props.name = "Tesla T4"
        mock_props.major = 7
        mock_props.minor = 5
        mock_props.total_memory = 15_109_001_216
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = mock_props
        with patch.dict("sys.modules", {"torch": mock_torch}):
            profile = detect_gpu()
        assert profile.name == "Tesla T4"
        assert profile.compute_capability == (7, 5)
        assert profile.architecture == "Turing"
