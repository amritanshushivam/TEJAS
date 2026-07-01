# TEJAS Training Recipe & Hyperparameters

This document defines standard pretraining setups, warmup decay limits, and learning rate schedules for training TEJAS model scales from scratch.

---

## 1. Hyperparameter Configurations

| Parameter | TEJAS Mini | TEJAS Small | TEJAS Medium |
| :--- | :--- | :--- | :--- |
| **Embedding Dimension ($d$)** | 256 | 512 | 768 |
| **Layers ($L$)** | 4 | 8 | 12 |
| **Attention Heads ($H$)** | 4 | 8 | 12 |
| **FeedForward Hidden Dim** | 682 | 1365 | 2048 |
| **Weight Decay** | 0.1 | 0.1 | 0.1 |
| **Max Learning Rate** | $3.0 \times 10^{-4}$ | $3.0 \times 10^{-4}$ | $3.0 \times 10^{-4}$ |
| **Min Learning Rate** | $3.0 \times 10^{-5}$ | $3.0 \times 10^{-5}$ | $3.0 \times 10^{-5}$ |
| **Warmup Steps** | 100 | 2000 | 2000 |
| **AdamW $\beta_1$** | 0.9 | 0.9 | 0.9 |
| **AdamW $\beta_2$** | 0.95 | 0.95 | 0.95 |
| **Gradient Clipping ($L_2$)** | 1.0 | 1.0 | 1.0 |

---

## 2. Optimization Pipeline

### Cosine Warmup Schedules
TEJAS relies on `CosineWarmupScheduler` to scale learning rates. For step $t$:
1. **Warmup Phase ($t < T_{\text{warmup}}$)**:
   $$\eta_t = \eta_{\text{min}} + \frac{t}{T_{\text{warmup}}} (\eta_{\text{max}} - \eta_{\text{min}})$$
2. **Cosine Decay Phase ($T_{\text{warmup}} \leq t \leq T_{\text{total}}$)**:
   $$\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\pi \frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}}\right)\right)$$

### Mixed Precision Training (AMP)
To save memory and accelerate training steps, TEJAS supports Automatic Mixed Precision:
- Activations and weights are mapped to `torch.float16` or `torch.bfloat16` during matrix multiplication.
- A `torch.cuda.amp.GradScaler` tracks gradients to scale losses dynamically, preventing underflow issues in lower-precision representations.

### Gradient Accumulation
When training on limited physical hardware, large effective batch sizes are simulated using gradient accumulation:
- Multiple forward and backward passes are run before invoking `optimizer.step()`.
- This replicates massive cluster batches without triggering Out-Of-Memory (OOM) errors.
