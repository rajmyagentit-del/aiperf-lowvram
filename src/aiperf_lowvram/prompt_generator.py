"""Synthetic prompt generator matching AIPerf methodology.

AIPerf generates prompts that hit exact input sequence length (ISL)
targets rather than using fixed strings. This module replicates that
approach so benchmark results are methodologically comparable.
"""

from __future__ import annotations

import random

# Word pool producing realistic BPE token distributions.
# Average English technical word tokenizes to ~1.3 tokens.
_WORD_POOL = [
    "the", "model", "inference", "latency", "throughput", "token",
    "generate", "neural", "network", "attention", "transformer",
    "benchmark", "hardware", "memory", "compute", "kernel", "batch",
    "request", "response", "stream", "cache", "prefill", "decode",
    "quantization", "precision", "floating", "point", "tensor",
    "parallel", "distributed", "serving", "deployment", "optimize",
    "performance", "measurement", "metric", "evaluation", "result",
    "system", "architecture", "configuration", "parameter", "weight",
    "gradient", "layer", "embedding", "vocabulary", "context", "window",
    "sequence", "length", "dimension", "head", "query", "key", "value",
]


def generate_prompt(target_tokens: int, seed_offset: int = 0) -> str:
    """Generate a synthetic prompt targeting approximately target_tokens.

    Matches AIPerf methodology: synthetic content at exact ISL targets
    rather than a fixed hardcoded string. This ensures TTFT measurements
    reflect the actual prefill cost of the target sequence length.

    Args:
        target_tokens: Approximate number of tokens desired.
        seed_offset: Added to base seed 42 for variety across prompts.

    Returns:
        A synthetic prompt string.
    """
    random.seed(42 + seed_offset)
    # ~0.75 words per token for typical BPE tokenizer
    target_words = max(1, int(target_tokens * 0.75))
    words = [random.choice(_WORD_POOL) for _ in range(target_words)]
    return (
        "Analyze the following technical description: "
        + " ".join(words)
        + ". Explain the key implications."
    )


def generate_prompt_batch(
    count: int,
    target_tokens: int,
    seed: int = 42,
) -> list[str]:
    """Generate a reproducible batch of synthetic prompts.

    Args:
        count: Number of prompts to generate.
        target_tokens: Target ISL for each prompt.
        seed: Base random seed for reproducibility.

    Returns:
        List of synthetic prompt strings.
    """
    return [generate_prompt(target_tokens, seed_offset=i) for i in range(count)]
