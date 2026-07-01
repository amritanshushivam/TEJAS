# TEJAS: Decoder-Only LLM Lab

TEJAS is a decoder-only Transformer language model implementation built from scratch with Python 3.14 and PyTorch 2.x.

## What It Includes

- RMSNorm
- Rotary Positional Embeddings (RoPE)
- Causal self-attention
- SwiGLU feed-forward blocks
- Weight tying
- Residual connections
- AdamW training
- AMP support
- Checkpointing
- Streaming generation

## Project Layout

- `tejas/tokenizer/` - BPE tokenizer
- `tejas/model/` - Transformer architecture
- `tejas/datasets/` - Dataset and dataloader utilities
- `tejas/training/` - Training loop and optimization
- `tejas/inference/` - Text generation utilities
- `tejas/evaluation/` - Evaluation metrics
- `tejas/configs/` - Model and training presets
- `tejas/utils/` - Shared helpers
- `tejas/main.py` - End-to-end pipeline entry point

## Quick Start

Install backend dependencies:

```bash
pip install -r tejas/requirements.txt
```

Run the pipeline:

```bash
python -m tejas.main
```

Run tests:

```bash
pytest tests/
```

## Notes

- The repository is now Python-only.
- The old web GUI has been removed.
- Keep changes backward compatible unless a real bug requires otherwise.
