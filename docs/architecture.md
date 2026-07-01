# TEJAS Architectural Specifications

This document outlines the detailed architectural blueprints, layer structures, and mathematical formulas of the TEJAS decoder-only Large Language Model.

---

## 1. Network Topology Overview

TEJAS strictly implements a pre-normalized transformer architecture. Normalization layers are placed immediately before the self-attention and feed-forward sub-layers. A residual skip connection adds the normalized layer output back to the original input token representation, preventing gradient dispersion across high depth ranges.

```text
       Input Tokens (Ids)
               │
      [Weight-Tied Embeddings]
               │
        Embedding Dropout
               │
   ┌───────────┴───────────┐
   │                       ▼
   │                 [RMSNorm]
   │                       │
   │               Causal Attention ─── [RoPE Positional Rotations]
   │                       │            [KV Caching Storage]
   │                Attention Dropout
   │                       │
   ▼                       ▼
( + ) ◄─────────────── Residual
   │
   ├───────────────────────┐
   │                       ▼
   │                 [RMSNorm]
   │                       │
   │                [SwiGLU FFN]
   │                       │
   │                 FFN Dropout
   │                       │
   ▼                       ▼
( + ) ◄─────────────── Residual
   │
   ▼
[RMSNorm]
   │
[Weight-Tied Projection LM Head]
   │
Output Logits (Softmax Predictions)
```

---

## 2. Mathematical Components

### Rotary Positional Embeddings (RoPE)
Instead of adding static positional vectors, TEJAS uses relative position rotary rotations. Given a d-dimensional embedding slice, it is split into 2D chunks, and rotated by position indices $m$:

$$\begin{pmatrix} q'_1 \\ q'_2 \end{pmatrix} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_1 \\ q_2 \end{pmatrix}$$

where the frequency parameters are defined as:
$$\theta_i = 10000^{-2(i-1)/d}$$

This transformation is implemented using PyTorch complex views in `precompute_freqs_cis` and applied inside `CausalSelfAttention`.

### SwiGLU Gated FeedForward Blocks
The intermediate FeedForward Network consists of three linear transformations. Two matrices project the input into a high-dimensional space where element-wise SiLU activation gating is applied, and the third projects it back:

$$\text{SwiGLU}(x) = \left( \text{SiLU}(xW_1) \cdot xW_3 \right) W_2$$

The hidden layer size is scaled using `multiple_of` parameters inside `SwiGLUFeedForward` to align allocation segments with standard tensor cores.

### Pre-RMSNorm
Traditional LayerNorm computes mean-centering statistics which is redundant on deep networks. TEJAS replaces LayerNorm with RMSNorm:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x^2_i + \epsilon}} \cdot \gamma$$

This speeds up convergence while saving floating-point compute registers during the forward and backward sweeps.
