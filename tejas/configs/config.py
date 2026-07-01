# tejas/configs/config.py
"""Architectural presets and configurations for TEJAS.

This module provides standard, production-ready hyperparameter configs:
- 'tejas-mini' (for local CPU testing or prototyping)
- 'tejas-small' (highly suited for consumer GPUs)
- 'tejas-medium' (full standard research model size)
"""

from __future__ import annotations

from typing import Dict, Any
from tejas.model.transformer import ModelArgs
from tejas.training.trainer import TrainerConfig


def get_model_args_preset(preset_name: str, vocab_size: int = 50257) -> ModelArgs:
    """Returns a pre-configured ModelArgs class instance based on predefined sizes.

    Args:
        preset_name: Key selector ('tejas-mini', 'tejas-small', 'tejas-medium').
        vocab_size: Target vocabulary size.
    """
    if preset_name == "tejas-mini":
        # Fast, low-memory footprints for unittests and CPU debugging
        return ModelArgs(
            vocab_size=vocab_size,
            dim=256,
            n_layers=4,
            n_heads=4,
            max_seq_len=256,
            dropout=0.0
        )
    elif preset_name == "tejas-small":
        # Great size for standard single-GPU environments (like Google Colab)
        return ModelArgs(
            vocab_size=vocab_size,
            dim=512,
            n_layers=8,
            n_heads=8,
            max_seq_len=1024,
            dropout=0.1
        )
    elif preset_name == "tejas-medium":
        # Standard research scale modeling comparable to GPT-2 Small (124M params)
        return ModelArgs(
            vocab_size=vocab_size,
            dim=768,
            n_layers=12,
            n_heads=12,
            max_seq_len=2048,
            dropout=0.1
        )
    else:
        raise ValueError(f"Unknown architectural preset '{preset_name}' requested.")


def get_trainer_config_preset(preset_name: str) -> TrainerConfig:
    """Returns training parameters tailored to architectural scales.

    Args:
        preset_name: Key selector ('tejas-mini', 'tejas-small', 'tejas-medium').
    """
    if preset_name == "tejas-mini":
        return TrainerConfig(
            epochs=2,
            learning_rate=5e-4,
            min_lr=5e-5,
            warmup_steps=10,
            accumulate_grad_batches=1,
            val_every_steps=10,
            save_every_steps=50,
            use_amp=False
        )
    elif preset_name == "tejas-small":
        return TrainerConfig(
            epochs=3,
            learning_rate=3e-4,
            min_lr=3e-5,
            warmup_steps=100,
            accumulate_grad_batches=2,
            val_every_steps=100,
            save_every_steps=200,
            use_amp=True
        )
    elif preset_name == "tejas-medium":
        return TrainerConfig(
            epochs=5,
            learning_rate=2.5e-4,
            min_lr=2.5e-5,
            warmup_steps=500,
            accumulate_grad_batches=4,
            val_every_steps=250,
            save_every_steps=500,
            use_amp=True
        )
    else:
        raise ValueError(f"Unknown training preset '{preset_name}' requested.")


# =====================================================================
# Verification and Dependent Information
# =====================================================================
# Dependent Files:
# - main.py (consumes these presets to initialize model and training configurations)
#
# Correctness:
# 1. Preset ratios maintain attention-head compatibility constraints (dim divisible by n_heads).
# 2. Accumulation values correspond appropriately to model weights to prevent memory leaks on consumer cards.
#
# Testing:
# Handled end-to-end via model instantiation and training integration loops.
