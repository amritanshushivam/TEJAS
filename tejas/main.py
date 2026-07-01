# tejas/main.py
"""Master execution script orchestrating the complete TEJAS lifecycle.

Demonstrates:
1. Environment and GPU checks.
2. Building a BPE Tokenizer from a local raw corpus.
3. Constructing an offline Map-style training Dataset and PyTorch DataLoader.
4. Instantiating a TEJAS decoder-only Transformer under the 'tejas-mini' preset.
5. Training the model via TejasTrainer with Cosine Warmup scheduling.
6. Evaluating loss and Perplexity using TejasEvaluator.
7. Serializing and loading checkpoints from disk.
8. Autoregressive inference with temperature, top-k, top-p, and repetitive penaltes.
"""

from __future__ import annotations

import math
import os
import shutil
import torch
from tejas.configs.config import get_model_args_preset, get_trainer_config_preset
from tejas.datasets.dataset import build_dataloader, TejasDataset
from tejas.evaluation.evaluator import TejasEvaluator
from tejas.inference.generator import GenerationConfig, TejasGenerator
from tejas.model.transformer import TejasTransformer
from tejas.tokenizer.bpe import TejasTokenizer, TokenizerConfig
from tejas.training.trainer import set_seed, TejasTrainer
from tejas.utils.helpers import check_gpu_availability, count_parameters, setup_logger

# 1. Setup Logging
logger = setup_logger("tejas.main")


def run_pipeline() -> None:
    """Runs the end-to-end TEJAS workflow."""
    logger.info("=========================================================")
    logger.info("             TEJAS DECODER-ONLY LLM LIFECYCLE            ")
    logger.info("=========================================================")

    # Setup determinism
    set_seed(42)

    # 2. Check GPU Environment
    env_info = check_gpu_availability()
    logger.info("Operating Device: %s (CUDA Available: %s)", env_info["device_name"], env_info["cuda_available"])
    device = torch.device(env_info["device_type"])

    # 3. Prepare Dummy Corpus for Model Pre-training
    dummy_corpus = (
        "tejas is a production-quality large language model. "
        "the architecture is a decoder-only transformer built with pytorch from scratch. "
        "tejas incorporates rotary positional embeddings for relative position encoding. "
        "it implements root mean square normalization to simplify layer activations. "
        "swiglu activation gates are used inside the feed-forward mlp structures. "
        "causal self-attention is accelerated using key-value sequence caching. "
        "weight tying couples the token embeddings directly with the output head. "
        "we train tejas using the adamw optimizer and gradient accumulation. "
        "cosine warmup schedules learning rates beautifully throughout epochs. "
        "the bpe tokenizer handles utf8 encoding deterministically."
    )
    # Multiply corpus to simulate multiple training steps
    full_corpus = "\n".join([dummy_corpus] * 50)

    # 4. Train Tokenizer
    logger.info("--- Step 1: Tokenizer Training ---")
    tok_config = TokenizerConfig(
        vocab_size=128,  # Tiny vocab size for micro mock runs
        special_tokens=["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"]
    )
    tokenizer = TejasTokenizer(tok_config)
    tokenizer.train_from_text(full_corpus, vocab_size=128, min_frequency=1)
    
    # Save tokenizer state
    tokenizer_file = "checkpoints/tokenizer.json"
    os.makedirs("checkpoints", exist_ok=True)
    tokenizer.save(tokenizer_file)

    # 5. Build Training Dataset & DataLoader
    logger.info("--- Step 2: Dataset & DataLoader Construction ---")
    # Context length of 16 tokens for mini tests
    max_seq_len = 16
    dataset = TejasDataset(
        text_corpus=full_corpus,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        pack_sequences=True
    )
    
    train_loader = build_dataloader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        pad_id=tokenizer.pad_id
    )
    logger.info("Loaded DataLoader with %d training batches.", len(train_loader))

    # 6. Initialize Model
    logger.info("--- Step 3: Model Initialization ---")
    # Retrieve 'tejas-mini' preset
    model_args = get_model_args_preset("tejas-mini", vocab_size=len(tokenizer.encoder))
    model_args.max_seq_len = max_seq_len * 2  # Support position capacity
    
    model = TejasTransformer(model_args).to(device)
    
    # Verify Parameter statistics
    param_stats = count_parameters(model)
    logger.info(
        "Total Params: %d | Trainable Params: %d | Memory Footprint: %.2f MB",
        param_stats["total_parameters"],
        param_stats["trainable_parameters"],
        param_stats["size_megabytes"]
    )

    # 7. Start Training Loops
    logger.info("--- Step 4: Model Optimization (Training) ---")
    # Retrieve tailor-made mini trainer configurations
    trainer_config = get_trainer_config_preset("tejas-mini")
    trainer_config.checkpoint_dir = "checkpoints"
    trainer_config.epochs = 1
    trainer_config.max_steps = 15  # Limit step cycles for rapid demo verification
    trainer_config.save_every_steps = 10
    
    trainer = TejasTrainer(
        model=model,
        config=trainer_config,
        train_loader=train_loader
    )
    trainer.fit()

    # 8. Evaluate Trained Model
    logger.info("--- Step 5: Metric Evaluation ---")
    evaluator = TejasEvaluator(model, tokenizer)
    metrics = evaluator.evaluate_loader(train_loader)
    logger.info(
        "Final Validation metrics: Mean Loss: %.4f | Perplexity: %.4f | Throughput: %.1f tokens/sec",
        metrics.loss, metrics.perplexity, metrics.tokens_per_second
    )

    # 9. Serialization Restore Sweep
    logger.info("--- Step 6: Checkpoint Resumption Audit ---")
    checkpoint_file = "checkpoints/checkpoint_step_10.pt"
    if os.path.exists(checkpoint_file):
        fresh_model = TejasTransformer(model_args).to(device)
        fresh_trainer = TejasTrainer(fresh_model, trainer_config, train_loader)
        
        fresh_trainer.load_checkpoint(checkpoint_file)
        logger.info("Model successfully resumed from global step %d.", fresh_trainer.global_step)
        
        # Override active model with resumed state for inference
        model = fresh_model

    # 10. Generate Text (Inference decoding strategies)
    logger.info("--- Step 7: Autoregressive Text Generation ---")
    generator = TejasGenerator(model, tokenizer)
    
    prompt = "tejas is a production-quality large language model."
    logger.info("Prompt Seed: '%s'", prompt)

    # Decode Strategy 1: Greedy Decoding (temperature = 0.0)
    greedy_config = GenerationConfig(
        max_new_tokens=15,
        temperature=0.0,
        eos_id=tokenizer.eos_id
    )
    greedy_out = generator.generate(prompt, config=greedy_config)
    logger.info("Greedy Generation Out: '%s'", prompt + greedy_out)

    # Decode Strategy 2: Top-p and Temperature sampling with Repetition Penalty
    sampling_config = GenerationConfig(
        max_new_tokens=15,
        temperature=0.8,
        top_k=25,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_id=tokenizer.eos_id
    )
    sampling_out = generator.generate(prompt, config=sampling_config)
    logger.info("Probabilistic Sampling Out: '%s'", prompt + sampling_out)

    logger.info("=========================================================")
    logger.info("      TEJAS SYSTEM LIFECYCLE EXECUTED SUCCESSFULLY       ")
    logger.info("=========================================================")


if __name__ == "__main__":
    # Clear preexisting checkpoints
    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints")
    run_pipeline()
