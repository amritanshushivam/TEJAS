# tests/test_training.py
"""Unit tests for the TEJAS Training engine.

Verifies random seeding, optimizer weight decay param splits, custom cosine scheduler decay,
gradients propagation in active training loops, and full trainer checkpoint restoration.
"""

from __future__ import annotations

import os
import tempfile
import random
import torch
import pytest
from torch.utils.data import DataLoader
from tejas.model.transformer import ModelArgs, TejasTransformer
from tejas.datasets.dataset import TejasDataset
from tejas.tokenizer.bpe import TejasTokenizer, TokenizerConfig
from tejas.training.trainer import (
    set_seed,
    CosineWarmupScheduler,
    TrainerConfig,
    TejasTrainer
)


def _build_tokenizer_and_model(
    corpus: str,
    vocab_size: int,
    dim: int,
    n_layers: int,
    n_heads: int,
    max_seq_len: int,
) -> tuple[TejasTokenizer, TejasTransformer]:
    tok_config = TokenizerConfig(vocab_size=vocab_size)
    tokenizer = TejasTokenizer(tok_config)
    tokenizer.train_from_text(corpus, vocab_size=vocab_size, min_frequency=1)

    model_args = ModelArgs(
        vocab_size=len(tokenizer.encoder),
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
    )
    model = TejasTransformer(model_args)
    return tokenizer, model


def test_seed_locking():
    """Validates that random seed control yields perfectly deterministic values across calls."""
    set_seed(42)
    a1 = torch.randn(2, 2)
    b1 = [random.random() for _ in range(5)]
    
    set_seed(42)
    a2 = torch.randn(2, 2)
    b2 = [random.random() for _ in range(5)]
    
    assert torch.allclose(a1, a2)
    assert b1 == b2


def test_scheduler_decay_curve():
    """Validates linear warmup and cosine decay calculations at discrete intervals."""
    dummy_model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(dummy_model.parameters(), lr=1.0)
    
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=10,
        total_steps=100,
        base_lr=1.0,
        min_lr=0.1
    )
    
    # Step 0 to Step 5: linear growth (warmup phase)
    # Expected: min_lr + (base_lr - min_lr) * (step / warmup_steps)
    # Step 5: 0.1 + (0.9) * 0.5 = 0.55
    for _ in range(5):
        lr = scheduler.step()
    assert abs(lr - 0.55) < 1e-5
    
    # Step 10: should reach peak base_lr of 1.0
    for _ in range(5):
        lr = scheduler.step()
    assert abs(lr - 1.0) < 1e-5
    
    # Step 55: middle of cosine decay
    # decay_ratio = (55 - 10) / (100 - 10) = 45 / 90 = 0.5
    # cos(pi * 0.5) = 0.0 -> coeff = 0.5 * (1 + 0) = 0.5
    # Expected LR: 0.1 + 0.5 * (0.9) = 0.55
    for _ in range(45):
        lr = scheduler.step()
    assert abs(lr - 0.55) < 1e-3
    
    # Step 100: should match exact min_lr of 0.1
    for _ in range(45):
        lr = scheduler.step()
    assert abs(lr - 0.1) < 1e-3


def test_optimizer_parameter_splits():
    """Verifies that norm scale parameters and embedding layers are exempted from weight decay."""
    tok, model = _build_tokenizer_and_model(
        corpus="corpus",
        vocab_size=300,
        dim=32,
        n_layers=1,
        n_heads=1,
        max_seq_len=4,
    )

    config = TrainerConfig(weight_decay=0.1)

    # Custom collator to extract dataset and mock loaders
    dataset = TejasDataset("some text", tokenizer=tok, max_seq_len=4)
    loader = DataLoader(dataset, batch_size=1)
    
    trainer = TejasTrainer(model, config, loader)
    
    # Verify we split into exactly 2 parameter groups: one with decay, one without decay
    assert len(trainer.optimizer.param_groups) == 2
    assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.1
    assert trainer.optimizer.param_groups[1]["weight_decay"] == 0.0
    
    # Confirm that RMSNorm weight is in the zero-decay group (param_groups[1])
    decay_params = set(trainer.optimizer.param_groups[0]["params"])
    no_decay_params = set(trainer.optimizer.param_groups[1]["params"])
    
    for name, param in model.named_parameters():
        if "norm" in name:
            assert param in no_decay_params
            assert param not in decay_params


def test_fit_gradient_updates():
    """Runs a mini training loop on mock activations to verify that active weights update successfully."""
    # Build tiny models and parameters
    tok, model = _build_tokenizer_and_model(
        corpus="mock corpus to setup token indexing in test datasets.",
        vocab_size=300,
        dim=16,
        n_layers=1,
        n_heads=1,
        max_seq_len=4,
    )
    
    dataset = TejasDataset(
        text_corpus="mock text line one.\nmock line two.",
        tokenizer=tok,
        max_seq_len=4,
        pack_sequences=True
    )
    loader = DataLoader(dataset, batch_size=2)
    
    # Save a reference to a specific weight matrix prior to updates
    prev_weight = model.layers[0].attention.wq.weight.clone()
    
    config = TrainerConfig(
        epochs=1,
        max_steps=2,
        learning_rate=0.1,
        min_lr=0.01,
        warmup_steps=0,
        checkpoint_dir="temp_checkpoints",
        use_amp=False
    )
    
    trainer = TejasTrainer(model, config, loader)
    trainer.fit()
    
    # Ensure parameter weights have actively drifted, indicating valid gradient updates
    new_weight = model.layers[0].attention.wq.weight
    assert not torch.allclose(prev_weight, new_weight)


def test_trainer_checkpoint_load_resume():
    """Tests saving a trainer checkpoint to a temporary file and restoring state seamlessly."""
    tok, model = _build_tokenizer_and_model(
        corpus="mock tokens list.",
        vocab_size=300,
        dim=16,
        n_layers=1,
        n_heads=1,
        max_seq_len=4,
    )
    
    dataset = TejasDataset("test string corpus data", tokenizer=tok, max_seq_len=4)
    loader = DataLoader(dataset, batch_size=1)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = TrainerConfig(
            epochs=1,
            checkpoint_dir=tmp_dir,
            save_every_steps=100
        )
        trainer = TejasTrainer(model, config, loader)
        
        # Advance state artificially
        trainer.global_step = 25
        trainer.scheduler.current_step = 25
        trainer.history["train_loss"].append(4.25)
        
        # Save
        trainer.save_checkpoint("test_checkpoint.pt")
        
        # Build fresh model/trainer and restore
        fresh_model = TejasTransformer(model.args)
        fresh_trainer = TejasTrainer(fresh_model, config, loader)
        
        fresh_trainer.load_checkpoint(os.path.join(tmp_dir, "test_checkpoint.pt"))
        
        assert fresh_trainer.global_step == 25
        assert fresh_trainer.scheduler.current_step == 25
        assert fresh_trainer.history["train_loss"] == [4.25]


def test_fit_flushes_partial_accumulation():
    """Ensures the trainer steps once at epoch end even when accumulation is incomplete."""
    tok, model = _build_tokenizer_and_model(
        corpus="short corpus for accumulation flush testing.",
        vocab_size=300,
        dim=16,
        n_layers=1,
        n_heads=1,
        max_seq_len=4,
    )

    dataset = TejasDataset(
        text_corpus="single training line for tail flush.",
        tokenizer=tok,
        max_seq_len=4,
        pack_sequences=True,
    )
    loader = DataLoader(dataset, batch_size=1)

    config = TrainerConfig(
        epochs=1,
        accumulate_grad_batches=2,
        checkpoint_dir="temp_checkpoints_tail_flush",
        use_amp=False,
    )

    trainer = TejasTrainer(model, config, loader)
    trainer.fit()

    assert trainer.global_step == 1


# =====================================================================
# Verification and Dependent Information
# =====================================================================
# Dependent Files:
# - tejas/training/trainer.py (the module being tested)
#
# Correctness:
# 1. Validates scheduling equations math correctness at multiple pivot steps.
# 2. Ensures zero weight decay applies safely to layer-norm scale vectors.
# 3. Confirms end-to-end backpropagation operates successfully on real tensors.
# 4. Validates deep serialization mapping correctness under temporary folder scopes.
#
# Testing:
# Run this test file using: pytest tests/test_training.py
