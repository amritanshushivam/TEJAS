# tejas/model/transformer.py
"""Tejas Decoder-Only Transformer Model Implementation.

This module provides a research-grade, highly optimized decoder-only transformer,
featuring modern architectural choices:
- Root Mean Square Normalization (RMSNorm)
- Rotary Position Embeddings (RoPE)
- SwiGLU Gated FeedForward Network (FFN)
- Multi-Head Self-Attention with Causal Mask and KV Caching
- Weight Tying between Input Embeddings and Language Modeling Head
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    """Hyperparameters and configuration for the TEJAS Transformer.

    Attributes:
        vocab_size: Size of the vocabulary.
        dim: Dimensionality of the model embeddings.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key-value heads for Grouped-Query Attention (optional).
                    Defaults to n_heads (Multi-Head Attention).
        multiple_of: FFN dimension multiplier constraint.
        ffn_dim_multiplier: Optional scaler for hidden FFN size.
        max_seq_len: Maximum supported sequence length for positional embeddings.
        dropout: Dropout probability.
        norm_eps: Epsilon value for numerical stability in RMSNorm.
    """
    vocab_size: int = 50257
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    max_seq_len: int = 2048
    dropout: float = 0.0
    norm_eps: float = 1e-5


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    Simplified Normalization that skips mean subtraction, saving computation
    and parameters while achieving identical convergence performance.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """Initializes weight and scale parameters."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the root-mean-square normalization."""
        # Math: x * rsqrt(mean(x^2, dim=-1) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies normalization and scaling."""
        return self._norm(x.float()).type_as(x) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precomputes rotation complex values (frequencies) for Rotary Positional Embeddings.

    Args:
        dim: Head dimension (must be even).
        end: Maximum sequence length.
        theta: Inverse frequency base.

    Returns:
        Tensor of complex frequencies with shape (end, dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    # Convert polar coordinates (r=1, theta) to complex numbers: cos(theta) + i*sin(theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshapes frequency tensor for correct elementwise broadcasting with input tensors."""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Positional Embeddings (RoPE) to query and key projection tensors.

    Args:
        xq: Query projections of shape (B, SeqLen, H_q, Dim_head)
        xk: Key projections of shape (B, SeqLen, H_k, Dim_head)
        freqs_cis: Precomputed complex frequencies of shape (SeqLen, Dim_head // 2)

    Returns:
        Tuple of rotated query and key tensors matching original shape.
    """
    # Group real dimensions as complex values
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast frequencies
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # Perform complex multiplication
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


class SwiGLUFeedForward(nn.Module):
    """SwiGLU (Swish Gated Linear Unit) Feed-Forward Network.

    Uses gating framework that outperforms traditional ReLU-based MLPs.
    Math: FFN(x) = (SiLU(xW_gate) * xW_up) * W_down
    """

    def __init__(self, args: ModelArgs) -> None:
        """Initializes layers and determines correct hidden dimension size."""
        super().__init__()
        
        # Determine standard hidden dimension size (4 * d)
        hidden_dim = 4 * args.dim
        # Scale intermediate projection size if requested
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round up to closest multiple of parameter constraint
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)  # Gate weight
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)  # Down-projection weight
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)  # Up-projection weight
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying SwiGLU gating computation."""
        # Math: silu(w1(x)) * w3(x) -> downproject -> dropout
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CausalSelfAttention(nn.Module):
    """Causal Multi-Head / Grouped-Query Self-Attention.

    Supports Multi-Head (MHA) or Grouped-Query (GQA) attention patterns,
    including causal triangular masking and key-value (KV) sequence caching.
    """

    def __init__(self, args: ModelArgs) -> None:
        """Initializes projections, parameters, and cache states."""
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        # Calculate key-value projection size ratios (for grouped attention support)
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

        # Local KV cache state container for inference
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def reset_cache(self) -> None:
        """Flushes cached attention sequence histories."""
        self.cache_k = None
        self.cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Runs the self-attention mechanism with causal masking.

        Args:
            x: Input activations (B, SeqLen, Dim)
            freqs_cis: Precomputed position frequencies
            use_cache: Flag to enable KV cache read/write for generation.
            start_pos: Generation sequence step offset index for caching.
        """
        bsz, seqlen, _ = x.shape
        
        # Project inputs to Query, Key, and Value states
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape to multi-head partitions: (B, SeqLen, Heads, HeadDim)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply Rotary positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # KV cache updates during autoregressive text generation
        if use_cache:
            if self.cache_k is None or start_pos == 0:
                # Initialize new cache vectors matching max capacity
                self.cache_k = torch.zeros((bsz, seqlen + 512, self.n_kv_heads, self.head_dim), device=x.device, dtype=x.dtype)
                self.cache_v = torch.zeros((bsz, seqlen + 512, self.n_kv_heads, self.head_dim), device=x.device, dtype=x.dtype)
            
            # Check for cache capacity overflow, and dynamically resize if needed
            if start_pos + seqlen > self.cache_k.shape[1]:
                extra_space = max(512, seqlen)
                self.cache_k = torch.cat([self.cache_k, torch.zeros((bsz, extra_space, self.n_kv_heads, self.head_dim), device=x.device, dtype=x.dtype)], dim=1)
                self.cache_v = torch.cat([self.cache_v, torch.zeros((bsz, extra_space, self.n_kv_heads, self.head_dim), device=x.device, dtype=x.dtype)], dim=1)

            # Store current projections
            self.cache_k[:, start_pos : start_pos + seqlen] = xk
            self.cache_v[:, start_pos : start_pos + seqlen] = xv

            # Retrieve complete historical sequences
            keys = self.cache_k[:, : start_pos + seqlen]
            values = self.cache_v[:, : start_pos + seqlen]
        else:
            keys = xk
            values = xv

        # Match Key-Value sizes with Grouped-Query Attention constraints if needed
        if self.n_rep > 1:
            keys = torch.repeat_interleave(keys, self.n_rep, dim=2)
            values = torch.repeat_interleave(values, self.n_rep, dim=2)

        # Align shapes for dot product attention computation:
        # (Batch, Heads, SeqLen, HeadDim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute dot-product raw attention scores
        # (B, H, SeqLen, Key_SeqLen)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply causal masking only during training or pre-fill phase
        if not use_cache or seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1)
            # Pad mask to match sequence dimensions
            if keys.shape[2] > seqlen:
                mask = torch.cat([torch.zeros((seqlen, keys.shape[2] - seqlen), device=x.device), mask], dim=1)
            scores = scores + mask

        # Softmax normalize scores to obtain weight coefficients
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)

        # Weighted summation of value vectors
        output = torch.matmul(scores, values)  # Shape: (B, H, SeqLen, HeadDim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # Project output and apply residual dropout
        return self.resid_dropout(self.wo(output))


class TransformerBlock(nn.Module):
    """A single complete Transformer block Layer.

    Maintains independent residual pipelines and applies pre-normalization.
    """

    def __init__(self, layer_id: int, args: ModelArgs) -> None:
        """Initializes internal Norm, Attention, and Feed-Forward modules."""
        super().__init__()
        self.layer_id = layer_id
        self.attention = CausalSelfAttention(args)
        self.feed_forward = SwiGLUFeedForward(args)
        
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Runs pre-norm block passes with skip connections."""
        # Attention pass + residual addition
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cis,
            use_cache=use_cache,
            start_pos=start_pos
        )
        # MLP / SwiGLU feed-forward pass + residual addition
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class TejasTransformer(nn.Module):
    """The central TEJAS Large Language Model.

    A high-precision decoder-only autoregressive transformer model.
    """

    def __init__(self, args: ModelArgs) -> None:
        """Initializes structural embeddings, blocks, and output systems."""
        super().__init__()
        self.args = args
        
        # Token embedding lookup layer
        self.token_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.embeddings_dropout = nn.Dropout(args.dropout)
        
        # Sequential list of transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(layer_id=idx, args=args) for idx in range(args.n_layers)
        ])
        
        # Final layers
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Weight tying (tying token embeddings to language modeling projections)
        self.output_head.weight = self.token_embeddings.weight

        # Precompute rotary frequencies matrix
        self.freqs_cis = precompute_freqs_cis(
            dim=args.dim // args.n_heads,
            end=args.max_seq_len * 2
        )

    def reset_cache(self) -> None:
        """Iterates through structural layers flushing all local attention histories."""
        for layer in self.layers:
            layer.attention.reset_cache()

    def forward(
        self,
        tokens: torch.Tensor,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Forward pass of the TEJAS transformer model.

        Args:
            tokens: Sequence token integer IDs of shape (B, SeqLen)
            use_cache: Flag to enable/disable KV caching for generation.
            start_pos: Position offset when retrieving elements from cache.

        Returns:
            Output prediction logits of shape (B, SeqLen, VocabSize)
        """
        _, seqlen = tokens.shape
        
        # Shape check & device synchronization
        h = self.token_embeddings(tokens)
        h = self.embeddings_dropout(h)
        
        # Slice correct subset of rotary position embeddings
        device = tokens.device
        if self.freqs_cis.device != device:
            self.freqs_cis = self.freqs_cis.to(device)
            
        freqs_cis_sub = self.freqs_cis[start_pos : start_pos + seqlen]

        # Feed activations sequentially through transformer blocks
        for layer in self.layers:
            h = layer(
                h,
                freqs_cis=freqs_cis_sub,
                use_cache=use_cache,
                start_pos=start_pos
            )

        # Normalize final activations and project to vocabulary distribution
        h = self.norm(h)
        logits = self.output_head(h)
        
        return logits


# =====================================================================
# Verification and Dependent Information
# =====================================================================
# Dependent Files:
# - tests/test_model.py (unit tests verifying layer shape preservation and RoPE consistency)
# - training/trainer.py (calculates CrossEntropyLoss and coordinates weights optimizations)
# - inference/decode.py (decodes sequential output logits during autoregressive text generation)
#
# Correctness:
# 1. Verification of weight tying guarantees parameter memory reduction and vocabulary projection alignment.
# 2. Rotary Positional Embeddings are calculated with high mathematical precision (using polar-complex transformations).
# 3. Dynamic resizing of KV cache prevents sequence overflow errors during long text generation.
# 4. Correct multi-head calculations ensure causal integrity during causal attention masking.
#
# Testing:
# Run this model code's structural tests using: pytest tests/test_model.py
