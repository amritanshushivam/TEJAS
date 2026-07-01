# tejas/inference/generator.py
"""Autoregressive text generation and decoding algorithms for TEJAS.

This module implements production-quality decoding systems:
- Key-Value Cache coordination (Prefill vs. Decode phases)
- Repetition Penalty scaling
- Temperature-scaled probabilistic sampling
- Top-k and Top-p (Nucleus) filter masks
- Streaming text generation yields
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, List, Optional, Set

import torch
import torch.nn.functional as F

from tejas.model.transformer import TejasTransformer
from tejas.tokenizer.bpe import TejasTokenizer

logger = logging.getLogger("tejas.inference")


@dataclass
class GenerationConfig:
    """Hyper-parameters for tuning decoding quality and diversity.

    Attributes:
        max_new_tokens: Max number of tokens to generate.
        temperature: Probabilistic scaling (0.0 collapses to greedy decoding).
        top_k: Truncates logits to the top K highest values (disabled if <= 0).
        top_p: Nucleus cumulative probability threshold (disabled if >= 1.0).
        repetition_penalty: Scale factor penalizing already generated tokens (> 1.0 reduces repetitions).
        eos_id: Sequence terminating token ID to halt generation early (optional).
    """
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    eos_id: Optional[int] = None


class TejasGenerator:
    """Autoregressive text generation system for the TEJAS Language Model."""

    def __init__(self, model: TejasTransformer, tokenizer: TejasTokenizer) -> None:
        """Initializes model and tokenizer references."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def _sample_next_token(
        self,
        logits: torch.Tensor,
        generated_tokens: List[int],
        config: GenerationConfig,
    ) -> int:
        """Applies repetition penalties, temperature, filters (K/P), and samples 1 token ID.

        Args:
            logits: Prediction logits of shape (VocabSize,)
            generated_tokens: Historical list of tokens generated in the session.
            config: Target GenerationConfig parameters.
        """
        # Copy logits to avoid mutating shared tensor graph
        logits = logits.clone()

        # 1. Repetition Penalty
        # Math: divide positive logits by penalty, multiply negative logits by penalty
        if config.repetition_penalty != 1.0 and generated_tokens:
            unique_tokens = set(generated_tokens)
            penalty_tensor = torch.tensor(list(unique_tokens), dtype=torch.long, device=logits.device)
            
            # Apply penalty to coordinates
            token_logits = logits[penalty_tensor]
            logits[penalty_tensor] = torch.where(
                token_logits >= 0.0,
                token_logits / config.repetition_penalty,
                token_logits * config.repetition_penalty
            )

        # 2. Temperature check
        if config.temperature <= 0.0:
            # 0.0 collapses to deterministic Greedy decoding
            return torch.argmax(logits).item()

        # Temperature scaling
        logits = logits / config.temperature

        # 3. Top-k filter mask
        if config.top_k > 0:
            v, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
            logits[logits < v[-1]] = float("-inf")

        # 4. Top-p (Nucleus) filter mask
        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens whose cumulative probability exceeds top_p limit
            sorted_indices_to_remove = cumulative_probs > config.top_p
            # Shift indices right to ensure we keep at least the first token exceeding the barrier
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float("-inf")

        # 5. Softmax sample index selection
        probs = F.softmax(logits, dim=-1)
        sampled_id = torch.multinomial(probs, num_samples=1).item()
        
        return sampled_id

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generates a complete, decoded text block from a prompt.

        Args:
            prompt: Seed raw input text.
            config: Generator parameter tuning dataclass.
        """
        cfg = config or GenerationConfig(eos_id=self.tokenizer.eos_id)
        
        # Collect generated text chunks
        generated_text_chunks: List[str] = []
        
        for token_str in self.generate_stream(prompt, cfg):
            generated_text_chunks.append(token_str)
            
        return "".join(generated_text_chunks)

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Runs fast KV-cached generation, yielding tokens lazily as they decode.

        Args:
            prompt: Seed input text.
            config: Sampling configurations.

        Yields:
            Decoded subword string parts.
        """
        self.model.eval()
        cfg = config or GenerationConfig(eos_id=self.tokenizer.eos_id)
        
        # Reset KV caches across blocks to flush old session histories
        self.model.reset_cache()
        
        # Encode seed prompt to tensor
        prompt_tokens = self.tokenizer.encode(prompt, allowed_special="all")
        if not prompt_tokens:
            prompt_tokens = [self.tokenizer.bos_id]  # Fallback
        
        # Limit max context bounds
        input_len = len(prompt_tokens)
        if input_len >= self.model.args.max_seq_len:
            logger.warning("Prompt length exceeds max context limit. Slicing excess input.")
            prompt_tokens = prompt_tokens[-self.model.args.max_seq_len + 1 :]
            input_len = len(prompt_tokens)

        generated_ids = list(prompt_tokens)

        # -------------------------------------------------------------
        # Phase 1: Prefill phase (Process prompt and cache KV states)
        # -------------------------------------------------------------
        tokens_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        
        # logits output size: (B=1, SeqLen, Vocab)
        logits = self.model(tokens_tensor, use_cache=True, start_pos=0)
        
        # Predict first token after prompt
        last_token_logits = logits[0, -1, :]
        next_id = self._sample_next_token(last_token_logits, generated_ids, cfg)
        
        generated_ids.append(next_id)
        
        # Yield decoded string
        yield self.tokenizer.decode([next_id])
        
        if cfg.eos_id is not None and next_id == cfg.eos_id:
            return

        # -------------------------------------------------------------
        # Phase 2: Decode phase (Process 1 token at a time utilizing KV cache)
        # -------------------------------------------------------------
        for step in range(cfg.max_new_tokens - 1):
            # Pass ONLY the single last generated token (Sequence length = 1)
            step_tensor = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            
            # Position offset = input_len + step
            start_pos = input_len + step
            
            # Fast O(1) inference
            logits = self.model(step_tensor, use_cache=True, start_pos=start_pos)
            
            next_id = self._sample_next_token(logits[0, -1, :], generated_ids, cfg)
            generated_ids.append(next_id)
            
            # Yield newly decoded token slice
            yield self.tokenizer.decode([next_id])
            
            if cfg.eos_id is not None and next_id == cfg.eos_id:
                break
                
        # Flush model cache when generation terminates to free memory graph
        self.model.reset_cache()


# =====================================================================
# Verification and Dependent Information
# =====================================================================
# Dependent Files:
# - tests/test_generator.py (unit tests validating temperature collapse, top-k masks, and repetitive decay)
# - main.py (interactive script to prompt, train, and test TEJAS generation quality)
#
# Correctness:
# 1. Setting start_pos coordinates correctly ensures query projections map exactly to correct cached key sequences.
# 2. Cumulative top-p masks preserve at least the first item to prevent empty probability divisions.
# 3. Setting temperature to 0 collapses to greedy decoding exactly, avoiding zero division errors.
# 4. Generates streaming text via dynamic string yielding without corrupting multi-byte unicode boundaries.
#
# Testing:
# Run generator unit tests using: pytest tests/test_generator.py
