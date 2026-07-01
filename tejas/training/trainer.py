# tejas/training/trainer.py
"""Training orchestration and optimization systems for TEJAS.

This module implements a production-grade trainer:
- Random Seed control for determinism
- Custom Cosine Learning Rate Scheduler with Warmup
- Mixed-Precision training (AMP) with GradScaler
- Gradient clipping and accumulation
- Cross Entropy evaluation (ignoring PAD indices)
- Validation sweeps
- Serialization/Resume-checkpoint workflows
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tejas.model.transformer import TejasTransformer

logger = logging.getLogger("tejas.training")


def set_seed(seed: int) -> None:
    """Sets random seeds across standard Python, NumPy, and PyTorch frameworks.

    Guarantees deterministic replication of training weights and dataloader shuffling.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure reproducible CUDA algorithms if using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed locked to %d for deterministic training.", seed)


@dataclass
class TrainerConfig:
    """Configuration hyper-parameters for training runs.

    Attributes:
        epochs: Max training epochs.
        max_steps: Optional absolute step limit. If reached, overrides epoch limit.
        learning_rate: Initial peak learning rate.
        min_lr: Minimum decayed learning rate.
        weight_decay: Weight decay factor for AdamW.
        beta1: AdamW beta1 coefficient.
        beta2: AdamW beta2 coefficient.
        warmup_steps: Linear lr warmup step count.
        grad_clip: Gradient norm clipping threshold.
        accumulate_grad_batches: Number of steps to accumulate gradients over.
        use_amp: Enable automatic mixed-precision training.
        checkpoint_dir: Target directory to save training checkpoints.
        save_every_steps: Step interval to save intermediate checkpoints.
        val_every_steps: Step interval to run validation sweeps.
    """
    epochs: int = 1
    max_steps: Optional[int] = None
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 100
    grad_clip: float = 1.0
    accumulate_grad_batches: int = 1
    use_amp: bool = False
    checkpoint_dir: str = "checkpoints"
    save_every_steps: int = 500
    val_every_steps: int = 100


class CosineWarmupScheduler:
    """Custom Cosine Learning Rate Scheduler with Linear Warmup.

    Adheres to standard GPT-style decay paths.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float,
    ) -> None:
        """Initializes scheduler variables."""
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self) -> float:
        """Computes learning rate for the current step and applies it to the optimizer."""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # 1. Linear Warmup Phase
            lr = self.min_lr + (self.base_lr - self.min_lr) * (self.current_step / max(1, self.warmup_steps))
        elif self.current_step > self.total_steps:
            # 2. Decay cap
            lr = self.min_lr
        else:
            # 3. Cosine Decay Phase
            decay_ratio = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.min_lr + coeff * (self.base_lr - self.min_lr)
            
        # Update learning rates inside optimizer param groups
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            
        return lr


class TejasTrainer:
    """Comprehensive Trainer Engine for optimizing the TEJAS Language Model."""

    def __init__(
        self,
        model: TejasTransformer,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """Initializes the optimization models, loaders, and scheduling metrics."""
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.device = next(model.parameters()).device
        logger.info("Initializing TEJAS Trainer on target device: %s.", self.device)
        
        # 1. Initialize optimizer
        self.optimizer = self._build_optimizer()
        
        # 2. Determine step counts and initialize custom scheduler
        num_batches = len(self.train_loader)
        total_training_steps = num_batches * self.config.epochs
        if self.config.max_steps is not None:
            total_training_steps = min(total_training_steps, self.config.max_steps)
            
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=total_training_steps,
            base_lr=self.config.learning_rate,
            min_lr=self.config.min_lr,
        )
        
        # 3. AMP Loss Scaler setup
        # Use bfloat16 or float16 based on CUDA availability
        scaler_device = "cuda" if "cuda" in str(self.device) else "cpu"
        self.scaler = torch.amp.GradScaler(device=scaler_device, enabled=self.config.use_amp)
        
        # 4. Standard cross-entropy loss function (ignoring pad token index)
        # Extract pad_id from dataset's tokenizer for consistency with training data
        # Fallback to 0 if not available, but dataset's tokenizer.pad_id should be used
        if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'tokenizer'):
            self.pad_id = train_loader.dataset.tokenizer.pad_id
        else:
            self.pad_id = 0  # Fallback if dataset doesn't have tokenizer
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        
        # State tracking metrics
        self.global_step = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "perplexity": [],
        }

    def _build_optimizer(self) -> AdamW:
        """Splits model weights into decay and no-decay groups, returning AdamW."""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Weight decay should not be applied to biases or normalization scales
            if "norm" in name or "bias" in name or "embeddings" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )

    def train_step(self, batch: Dict[str, torch.Tensor], step_idx: int) -> float:
        """Runs a single forward and backward pass, scaling and accumulating gradients."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Enable Autocast if AMP is active
        # PyTorch uses "cuda" or "cpu" as device_type for autocasting
        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        autocast_ctx = torch.amp.autocast(device_type=device_type, enabled=self.config.use_amp)
        
        with autocast_ctx:
            # logits: (B, SeqLen, VocabSize)
            logits = self.model(input_ids)
            
            # Reshape for cross entropy: (B * SeqLen, VocabSize) vs (B * SeqLen)
            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.view(-1)
            
            loss = self.loss_fn(flat_logits, flat_labels)
            # Scale loss proportionally to gradient accumulation batches
            loss = loss / self.config.accumulate_grad_batches
            
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        return loss.item() * self.config.accumulate_grad_batches

    def optimize_step(self) -> float:
        """Executes clipping, parameter updates, learning rate scheduler steps, and resets."""
        # Unscale gradients prior to clipping
        self.scaler.unscale_(self.optimizer)
        
        # Gradient norm clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Flush accumulated gradients
        self.optimizer.zero_grad(set_to_none=True)
        
        # Update decayed learning rate step
        lr = self.scheduler.step()
        
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

    def validate(self) -> float:
        """Runs validation pass over validation dataloader, calculating mean loss."""
        if self.val_loader is None:
            return 0.0
            
        self.model.eval()
        total_loss = 0.0
        count = 0
        
        logger.info("Running validation evaluation sweep...")
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                logits = self.model(input_ids)
                flat_logits = logits.view(-1, logits.size(-1))
                flat_labels = labels.view(-1)
                
                loss = self.loss_fn(flat_logits, flat_labels)
                total_loss += loss.item()
                count += 1
                
        self.model.train()
        mean_loss = total_loss / max(1, count)
        logger.info("Validation completed. Mean Loss: %.4f, Perplexity: %.4f", mean_loss, math.exp(mean_loss))
        return mean_loss

    def fit(self) -> None:
        """Begins the core training loop for epochs and steps specified."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        logger.info("Starting training loop...")
        
        step_loss_accumulator = 0.0
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            logger.info("Starting Epoch %d / %d...", epoch + 1, self.config.epochs)
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Execute individual forward-backward training step
                loss_val = self.train_step(batch, batch_idx)
                step_loss_accumulator += loss_val
                
                # If accumulation interval reached or this is the final batch, step optimizer and schedule learning rate
                is_last_batch = batch_idx + 1 == len(self.train_loader)
                if (batch_idx + 1) % self.config.accumulate_grad_batches == 0 or is_last_batch:
                    grad_norm = self.optimize_step()
                    self.global_step += 1
                    
                    # Compute mean training loss for this step
                    step_loss = step_loss_accumulator
                    step_loss_accumulator = 0.0
                    
                    # Record tracking histories
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.history["train_loss"].append(step_loss)
                    self.history["lr"].append(current_lr)
                    
                    # Periodic console logging
                    if self.global_step % 10 == 0 or self.global_step == 1:
                        elapsed = time.time() - start_time
                        logger.info(
                            "Step %d | Loss: %.4f | LR: %.3e | GradNorm: %.3f | Time/Step: %.2fs",
                            self.global_step, step_loss, current_lr, grad_norm, elapsed / 10
                        )
                        start_time = time.time()
                        
                    # Periodic validation sweeps
                    if self.global_step % self.config.val_every_steps == 0:
                        val_loss = self.validate()
                        self.history["val_loss"].append(val_loss)
                        self.history["perplexity"].append(math.exp(val_loss))
                        self.model.train()
                        
                    # Periodic checkpoint serialization
                    if self.global_step % self.config.save_every_steps == 0:
                        self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
                        
                    # Step limits checks
                    if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                        logger.info("Step limit of %d steps reached. Ending training early.", self.config.max_steps)
                        break
                        
            if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                break
                
        # Final checkpoint save at training completion
        self.save_checkpoint("checkpoint_final.pt")
        logger.info("Training fully completed successfully!")

    def save_checkpoint(self, file_name: str) -> None:
        """Saves a complete training checkpoint to disk to allow resuming training."""
        file_path = os.path.join(self.config.checkpoint_dir, file_name)
        
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state": {
                "current_step": self.scheduler.current_step,
                "warmup_steps": self.scheduler.warmup_steps,
                "total_steps": self.scheduler.total_steps,
                "base_lr": self.scheduler.base_lr,
                "min_lr": self.scheduler.min_lr,
            },
            "history": self.history,
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config,
        }
        
        torch.save(checkpoint, file_path)
        logger.info("Training checkpoint successfully saved to %s.", file_path)

    def load_checkpoint(self, file_path: str) -> None:
        """Loads a training checkpoint and restores all model, optimizer, scheduler, and loss histories."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Checkpoint file not found at {file_path}")

        # Trusted local training checkpoints contain optimizer/config objects,
        # so they must be loaded with weights_only disabled on PyTorch 2.6+.
        checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
        
        # Load states
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        # Restore scheduler states
        sched_state = checkpoint["scheduler_state"]
        self.scheduler.current_step = sched_state["current_step"]
        self.scheduler.warmup_steps = sched_state["warmup_steps"]
        self.scheduler.total_steps = sched_state["total_steps"]
        self.scheduler.base_lr = sched_state["base_lr"]
        self.scheduler.min_lr = sched_state["min_lr"]
        
        self.global_step = checkpoint["global_step"]
        self.history = checkpoint["history"]
        
        logger.info("Successfully loaded checkpoint from %s. Resuming training at step %d.", file_path, self.global_step)


# =====================================================================
# Verification and Dependent Information
# =====================================================================
# Dependent Files:
# - tests/test_training.py (unit tests validating optimization gradients and scheduler decay)
# - main.py (orchestrates full workflow: config -> data -> training -> inference)
#
# Correctness:
# 1. Splitting parameters ensures normalization scales are not impacted by L2 regularization.
# 2. Gradient accumulation scales backward passes correctly to prevent clipping skew.
# 3. Custom cosine scheduler mirrors the Exact decay math used in commercial transformer suites.
# 4. Checking ignores PAD indexes in cross entropy, preventing padding from inflating accuracy scores.
#
# Testing:
# Run trainer unit tests using: pytest tests/test_training.py
