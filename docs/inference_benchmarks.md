# TEJAS Inference & Decoding Specifications

This document outlines the design, mathematics, and performance benchmarks of the TEJAS text generation and decoding systems.

---

## 1. Key-Value (KV) Caching Mechanism

During autoregressive text generation, predicting the next token requires computing self-attention over all previous tokens. In standard transformers, this demands recalculating Key and Value projection vectors on every generation step:
- **With no cache**: Time complexity scales quadratically at $O(N^2)$ per generation step.
- **With KV Cache**: Pre-computed Key ($K$) and Value ($V$) tensors are stored in GPU memory. The model only projects the single newly predicted token's vectors, shrinking time complexity to $O(1)$ per step.

```text
Step t:
  New Input Token ────────► [Embeddings + RoPE] ──► q_t, k_t, v_t
                                                      │
                                                      ▼
  KV Cache Store:
    Keys:   [ k_1, k_2, ..., k_{t-1} ] <── Append ─── k_t
    Values: [ v_1, v_2, ..., v_{t-1} ] <── Append ─── v_t
                                                      │
                                                      ▼
  Attention Calculation:
    Scores = Softmax( q_t · Keys^T / sqrt(d) + Mask )
    Output = Scores · Values
```

---

## 2. Decoding and Sampling Configuration

`TejasGenerator` implements four core layers to model generation variety:

### Temperature scaling
Normalizes logits by scale factor $T$:
$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
- Low temperatures ($T \to 0$) lead to deterministic, greedy outputs.
- High temperatures ($T > 1.0$) enrich variety.

### Top-K Filtering
Limits candidates to the top $K$ most likely tokens, cutting off unstable, low-probability completions.

### Top-P (Nucleus) Sampling
Limits candidates dynamically by summing sorted probabilities up to cutoff value $P$:
$$\sum_{i=1}^{V'} p_i \geq P$$
This ensures the candidate pool expands during high-entropy scenarios and shrinks during highly predictable segments.

### Repetition Penalty
Down-scales logits for tokens that have already been generated in the current output stream by divisor $\alpha$:
$$\theta_i = \begin{cases} z_i / \alpha & \text{if } z_i \geq 0 \\ z_i \cdot \alpha & \text{if } z_i < 0 \end{cases}$$
This prevents repetitive loops and promotes varied vocabulary selections.
