"""Synthetic prompt generator matching AIPerf methodology.

AIPerf generates prompts that hit exact input sequence length (ISL)
targets rather than using fixed strings. This module replicates that
approach so benchmark results are methodologically comparable.
"""

from __future__ import annotations

import random
import string


# Word list that produces realistic token distributions
# Average English word tokenizes to ~1.3 tokens in most BPE tokenizers
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


def generate_prompt(target_tokens: int, words_per_token: float = 0.75) -> str:
    """Generate a synthetic prompt targeting approximately target_tokens.

    Uses a word pool with realistic token distribution rather than
    random characters, so the prompt looks like natural language to
    the tokenizer and produces representative prefill behaviour.

    Args:
        target_tokens: Approximate number of tokens desired.
        words_per_token: Words per token ratio. Default 0.75 means
                         ~1.33 tokens per word, matching typical BPE.

    Returns:
        A synthetic prompt string.
    """
    target_words = max(1, int(target_tokens * words_per_token))
    words = [random.choice(_WORD_POOL) for _ in range(target_words)]

    # Structure as a question to get consistent response behaviour
    prompt = (
        "Please analyze the following technical description and provide "
        "a detailed explanation: " + " ".join(words) + ". "
        "What are the key implications?"
    )
    return prompt


def generate_prompt_batch(
    count: int,
    target_tokens: int,
    seed: int = 42,
) -> list[str]:
    """Generate a reproducible batch of synthetic prompts.

    Args:
        count: Number of prompts to generate.
        target_tokens: Target ISL for each prompt.
        seed: Random seed for reproducibility.

    Returns:
        List of synthetic prompt strings.
    """
    random.seed(seed)
    return [generate_prompt(target_tokens) for _ in range(count)]
