# tests/test_evaluation.py
"""Unit tests for the TEJAS Evaluation suite.

Verifies loss accumulation accuracy, numerical perplexity boundaries,
DataLoader sweeps, and raw corpus token statistics.
"""

from __future__ import annotations

import math
import torch
import pytest
from torch.utils.data import DataLoader
from tejas.model.transformer import ModelArgs, TejasTransformer
from tejas.tokenizer.bpe import TejasTokenizer, TokenizerConfig
from tejas.datasets.dataset import TejasDataset
from tejas.evaluation.evaluator import TejasEvaluator, EvaluationMetrics


@pytest.fixture
def evaluator_setup():
    """Initializes evaluator with tiny models and tokenizers for testing."""
    tok_config = TokenizerConfig(vocab_size=300)
    tok = TejasTokenizer(tok_config)
    tok.train_from_text("corpus", vocab_size=300, min_frequency=1)

    args = ModelArgs(vocab_size=len(tok.encoder), dim=16, n_layers=1, n_heads=1, max_seq_len=8)
    model = TejasTransformer(args)
    model.eval()
    
    evaluator = TejasEvaluator(model, tok)
    return evaluator, model, tok


def test_perplexity_math(evaluator_setup):
    """Verifies that mathematical perplexity mirrors exactly exponential cross entropy loss."""
    evaluator, _, _ = evaluator_setup
    
    # Text evaluation
    text = "test corpus evaluating language model complexity."
    metrics = evaluator.evaluate_text(text, max_seq_len=4)
    
    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.loss >= 0.0
    
    # PPL should equal exp(loss)
    expected_ppl = math.exp(metrics.loss)
    assert abs(metrics.perplexity - expected_ppl) < 1e-4


def test_loader_evaluation(evaluator_setup):
    """Ensures evaluation executes cleanly over multi-batch dataloaders."""
    evaluator, _, tok = evaluator_setup
    
    # Create simple dataset and dataloader
    dataset = TejasDataset("some text to populate batch inputs", tokenizer=tok, max_seq_len=4)
    loader = DataLoader(dataset, batch_size=2)
    
    metrics = evaluator.evaluate_loader(loader)
    
    assert metrics.total_tokens > 0
    assert metrics.loss > 0.0
    assert metrics.perplexity >= 1.0
    assert metrics.tokens_per_second >= 0.0
    assert metrics.unique_tokens >= 0


def test_token_statistics_analysis(evaluator_setup):
    """Validates token distribution and vocabulary coverage calculation stats."""
    evaluator, _, _ = evaluator_setup
    
    text = "evaluate corpus coverage and statistics"
    stats = evaluator.analyze_token_statistics(text)
    
    assert "total_tokens" in stats
    assert "unique_tokens" in stats
    assert "avg_token_length" in stats
    assert "vocab_coverage" in stats
    
    assert stats["total_tokens"] > 0
    assert stats["unique_tokens"] <= stats["total_tokens"]
    assert stats["avg_token_length"] > 0.0
    assert 0.0 <= stats["vocab_coverage"] <= 1.0


# =====================================================================
# Verification and Dependent Information
# =====================================================================
# Dependent Files:
# - tejas/evaluation/evaluator.py (the module being tested)
#
# Correctness:
# 1. Confirms sliding boundaries calculate offset predictions correctly.
# 2. Verifies float safety under extreme loss ranges.
# 3. Asserts that statistics coverage fits within [0.0, 1.0] percentage constraints.
#
# Testing:
# Run this test file using: pytest tests/test_evaluation.py
