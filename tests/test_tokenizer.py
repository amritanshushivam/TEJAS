# tests/test_tokenizer.py
"""Unit tests for the TEJAS BPE Tokenizer.

Verifies vocabulary initialization, training correctness, encoding, decoding,
special tokens handling, and JSON serialization.
"""

from __future__ import annotations

import os
import tempfile
import pytest
from tejas.tokenizer.bpe import TejasTokenizer, TokenizerConfig


def test_special_tokens_initialization():
    """Verifies that special tokens are correctly registered and have the correct IDs."""
    config = TokenizerConfig(
        vocab_size=1000,
        special_tokens=["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"]
    )
    tokenizer = TejasTokenizer(config)

    assert tokenizer.bos_id == 0
    assert tokenizer.eos_id == 1
    assert tokenizer.pad_id == 2
    assert tokenizer.unk_id == 3

    assert tokenizer.decode([0]) == "<|bos|>"
    assert tokenizer.decode([1]) == "<|eos|>"
    assert tokenizer.decode([2]) == "<|pad|>"
    assert tokenizer.decode([3]) == "<|unk|>"


def test_bpe_training():
    """Verifies that the tokenizer trains correctly on a sample corpus."""
    corpus = "tejas is a model. tejas tokenizer is modular and clean. tejas is clean."
    
    config = TokenizerConfig(vocab_size=300)
    tokenizer = TejasTokenizer(config)
    
    # Train
    tokenizer.train_from_text(corpus, vocab_size=300, min_frequency=1)
    
    # Vocab size should be greater than the baseline (4 special tokens + 256 bytes = 260)
    assert len(tokenizer.encoder) > 260
    assert len(tokenizer.encoder) <= 300


def test_encode_decode_roundtrip():
    """Ensures that encoding and decoding is a lossless operation (roundtrip)."""
    corpus = "The quick brown fox jumps over the lazy dog. Programming in PyTorch is awesome! ⚡🤖"
    
    config = TokenizerConfig(vocab_size=320)
    tokenizer = TejasTokenizer(config)
    tokenizer.train_from_text(corpus, vocab_size=320, min_frequency=1)
    
    # Text to encode and decode
    input_text = "Programming in PyTorch over the lazy dog."
    token_ids = tokenizer.encode(input_text)
    decoded_text = tokenizer.decode(token_ids)
    
    assert decoded_text == input_text


def test_special_tokens_boundary():
    """Validates that special tokens are preserved and handled during encoding and decoding."""
    corpus = "Hello world!"
    config = TokenizerConfig(vocab_size=300)
    tokenizer = TejasTokenizer(config)
    tokenizer.train_from_text(corpus, vocab_size=300, min_frequency=1)
    
    input_text = "<|bos|>Hello world!<|eos|>"
    
    # Test with default allowed_special='all'
    token_ids = tokenizer.encode(input_text, allowed_special="all")
    assert tokenizer.bos_id in token_ids
    assert tokenizer.eos_id in token_ids
    
    decoded = tokenizer.decode(token_ids)
    assert decoded == input_text


def test_save_load_roundtrip():
    """Verifies tokenizer configuration serialization and load integrity."""
    corpus = "Machine learning and deep neural networks are transforming standard computing systems."
    config = TokenizerConfig(vocab_size=350)
    tokenizer = TejasTokenizer(config)
    tokenizer.train_from_text(corpus, vocab_size=350, min_frequency=1)
    
    original_text = "machine neural computing"
    original_ids = tokenizer.encode(original_text)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "tokenizer.json")
        tokenizer.save(save_path)
        
        # Create a new blank tokenizer and load
        loaded_tokenizer = TejasTokenizer()
        loaded_tokenizer.load(save_path)
        
        assert loaded_tokenizer.config.vocab_size == tokenizer.config.vocab_size
        assert len(loaded_tokenizer.encoder) == len(tokenizer.encoder)
        
        # Check if codes match
        loaded_ids = loaded_tokenizer.encode(original_text)
        assert loaded_ids == original_ids
        
        decoded_text = loaded_tokenizer.decode(loaded_ids)
        assert decoded_text == original_text


def test_save_to_current_directory_roundtrip():
    """Verifies tokenizer save/load works when the target path has no parent directory."""
    corpus = "Saving to the current directory should work cleanly."
    tokenizer = TejasTokenizer(TokenizerConfig(vocab_size=320))
    tokenizer.train_from_text(corpus, vocab_size=320, min_frequency=1)

    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            os.chdir(tmp_dir)
            tokenizer.save("tokenizer_root.json")

            reloaded = TejasTokenizer()
            reloaded.load("tokenizer_root.json")

            sample_text = "current directory save"
            assert reloaded.encode(sample_text) == tokenizer.encode(sample_text)
        finally:
            os.chdir(original_cwd)


# =====================================================================
# Verification and Dependent Information
# =====================================================================
# Dependent Files:
# - tejas/tokenizer/bpe.py (the module being tested)
#
# Correctness:
# 1. Coverage of all major tokenizer lifecycles (init, train, encode, decode, save, load).
# 2. Utilizes temporary directories to isolate filesystem changes during testing.
# 3. Explicitly asserts lossless Unicode string mapping (roundtrip).
#
# Testing:
# Run this test file using: pytest tests/test_tokenizer.py
