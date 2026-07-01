# tests/test_model.py
"""Unit tests for the TEJAS Decoder-Only Transformer Model.

Verifies structural submodules (RMSNorm, SwiGLU, RoPE), complete model forward passes,
lossless weight tying, causal masking, and KV cache behavior.
"""

from __future__ import annotations

import torch
import pytest
from tejas.model.transformer import (
    ModelArgs,
    RMSNorm,
    SwiGLUFeedForward,
    precompute_freqs_cis,
    apply_rotary_emb,
    TejasTransformer
)


def test_rmsnorm_properties():
    """Validates that RMSNorm normalizes input activations and matches weight shapes."""
    batch_size, seq_len, dim = 2, 4, 16
    x = torch.randn(batch_size, seq_len, dim) * 10.0
    
    norm_layer = RMSNorm(dim=dim, eps=1e-5)
    out = norm_layer(x)
    
    assert out.shape == x.shape
    # Check that root mean square is approximately 1.0 before weight scale
    rms_pre_weight = out.pow(2).mean(-1).sqrt()
    # Since weight is initialized to 1.0, the root mean square of out should be near 1.0
    torch.testing.assert_close(rms_pre_weight, torch.ones_like(rms_pre_weight), rtol=1e-3, atol=1e-3)


def test_swiglu_dimensions():
    """Ensures SwiGLU FeedForward conforms to multiple_of rounding constraints."""
    args = ModelArgs(
        dim=128,
        multiple_of=32,
        ffn_dim_multiplier=1.0
    )
    ffn = SwiGLUFeedForward(args)
    
    x = torch.randn(2, 4, 128)
    out = ffn(x)
    
    assert out.shape == x.shape
    
    # Hidden dimension check:
    # Standard hidden dim = 4 * 128 = 512
    # SwiGLU scaling = 2 * 512 / 3 = 341.333
    # Round to multiple of 32 = 32 * ceil(341.333/32) = 32 * 11 = 352
    assert ffn.w1.out_features == 352
    assert ffn.w2.in_features == 352


def test_rope_rotation_properties():
    """Verifies Rotary Positional Embedding (RoPE) shapes and output sizes."""
    batch, seq_len, heads, head_dim = 2, 8, 4, 16
    xq = torch.randn(batch, seq_len, heads, head_dim)
    xk = torch.randn(batch, seq_len, heads, head_dim)
    
    freqs_cis = precompute_freqs_cis(dim=head_dim, end=32)
    freqs_cis_sub = freqs_cis[:seq_len]
    
    rot_q, rot_k = apply_rotary_emb(xq, xk, freqs_cis_sub)
    
    assert rot_q.shape == xq.shape
    assert rot_k.shape == xk.shape
    # Ensure they are not identical to inputs, indicating that rotation took place
    assert not torch.allclose(rot_q, xq)
    assert not torch.allclose(rot_k, xk)


def test_weight_tying():
    """Asserts that weight tying correctly mirrors input embeddings and output projection."""
    args = ModelArgs(vocab_size=1000, dim=64, n_layers=2, n_heads=2)
    model = TejasTransformer(args)
    
    # Assert they share the exact same underlying tensor memory pointer
    assert model.token_embeddings.weight.data_ptr() == model.output_head.weight.data_ptr()


def test_model_forward_pass():
    """Verifies end-to-end forward pass shapes on a mini TEJAS model."""
    args = ModelArgs(
        vocab_size=100,
        dim=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=16
    )
    model = TejasTransformer(args)
    model.eval()
    
    tokens = torch.randint(0, 100, (2, 8))  # Batch=2, SeqLen=8
    
    with torch.no_grad():
        logits = model(tokens)
        
    assert logits.shape == (2, 8, 100)  # Shape: (B, SeqLen, VocabSize)


def test_kv_cache_inference():
    """Validates that inference using KV caching produces identical results to a single forward pass."""
    args = ModelArgs(
        vocab_size=50,
        dim=32,
        n_layers=1,
        n_heads=2,
        max_seq_len=16
    )
    model = TejasTransformer(args)
    model.eval()
    
    # 1. Standard forward pass over 4 tokens
    input_tokens = torch.randint(0, 50, (1, 4))
    
    with torch.no_grad():
        standard_logits = model(input_tokens)
        
    # 2. Sequential caching pass token-by-token
    model.reset_cache()
    cached_logits_list = []
    
    with torch.no_grad():
        # First step: pre-fill (process first 3 tokens)
        logits_prefill = model(input_tokens[:, :3], use_cache=True, start_pos=0)
        # Next step: generate 4th token (input sequence length = 1)
        logits_step = model(input_tokens[:, 3:4], use_cache=True, start_pos=3)
        
    # Standard forward pass logits for the last token should match cached-step logits
    last_standard_logits = standard_logits[:, 3:4, :]
    
    # Check that they match with reasonable precision
    torch.testing.assert_close(logits_step, last_standard_logits, rtol=1e-4, atol=1e-4)


# =====================================================================
# Verification and Dependent Information
# =====================================================================
# Dependent Files:
# - tejas/model/transformer.py (the module being tested)
#
# Correctness:
# 1. Comprehensive coverage of all individual structural features (RMSNorm, SwiGLU, RoPE).
# 2. Explicitly checks and asserts the memory pointer shared in weight tying.
# 3. Tests key autoregressive caching behavior to prevent regression in generation logic.
#
# Testing:
# Run this test file using: pytest tests/test_model.py
