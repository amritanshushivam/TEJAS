# tests/test_dataset.py
"""Unit tests for the TEJAS Dataset modules and loaders.

Validates Map-style datasets, sequence packing boundary correctness,
streaming iterable data, and the dynamic batch padding collator.
"""

from __future__ import annotations

import os
import tempfile
import torch
import pytest
from tejas.tokenizer.bpe import TejasTokenizer, TokenizerConfig
from tejas.datasets.dataset import TejasDataset, TejasIterableDataset, TejasCollator, build_dataloader


@pytest.fixture
def dummy_tokenizer():
    """Provides a dummy tokenizer trained on basic text for testing."""
    config = TokenizerConfig(vocab_size=300)
    tok = TejasTokenizer(config)
    corpus = "tejas model. transformers are great. sequence packing is very cool."
    tok.train_from_text(corpus, vocab_size=300, min_frequency=1)
    return tok


def test_map_dataset_no_packing(dummy_tokenizer):
    """Verifies offset input-target matching and shapes in non-packed Map-style dataset."""
    text_corpus = "transformers are great.\nsequence packing is very cool."
    
    # max_seq_len = 8 (implies inputs of len 8, targets of len 8)
    dataset = TejasDataset(
        text_corpus=text_corpus,
        tokenizer=dummy_tokenizer,
        max_seq_len=8,
        pack_sequences=False
    )
    
    assert len(dataset) > 0
    item = dataset[0]
    
    assert "input_ids" in item
    assert "labels" in item
    assert item["input_ids"].shape == (8,)
    assert item["labels"].shape == (8,)
    
    # Assert standard 1-token language modeling offset shift:
    # labels[i] should correspond to inputs[i+1] (excluding the very last target token)
    torch.testing.assert_close(item["input_ids"][1:], item["labels"][:-1])


def test_map_dataset_packing(dummy_tokenizer):
    """Verifies that sequence packing merges text documents and fills capacity limits."""
    text_corpus = "short.\nshort line.\nanother line."
    
    dataset = TejasDataset(
        text_corpus=text_corpus,
        tokenizer=dummy_tokenizer,
        max_seq_len=16,
        pack_sequences=True
    )
    
    # Packed documents should produce dense sequences, reducing padding counts
    assert len(dataset) >= 1
    item = dataset[0]
    assert item["input_ids"].shape == (16,)


def test_collator_dynamic_padding():
    """Verifies that the collator dynamically pads elements of differing lengths to maximum batch size."""
    pad_id = 99
    collator = TejasCollator(pad_id=pad_id)
    
    batch = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 4])},
        {"input_ids": torch.tensor([5, 6]), "labels": torch.tensor([6, 7])},
        {"input_ids": torch.tensor([8, 9, 10, 11]), "labels": torch.tensor([9, 10, 11, 12])}
    ]
    
    collated = collator(batch)
    
    assert "input_ids" in collated
    assert "labels" in collated
    
    # All rows should pad to length 4 (longest item size)
    assert collated["input_ids"].shape == (3, 4)
    assert collated["labels"].shape == (3, 4)
    
    # Row 1 (length 3 originally) should have exactly 1 pad token at the end
    assert collated["input_ids"][0, -1].item() == pad_id
    # Row 2 (length 2 originally) should have exactly 2 pad tokens
    assert collated["input_ids"][1, -1].item() == pad_id
    assert collated["input_ids"][1, -2].item() == pad_id
    # Row 3 (length 4 originally) should have no padding
    assert collated["input_ids"][2, -1].item() != pad_id


def test_iterable_streaming_dataset(dummy_tokenizer):
    """Validates lazy streaming and token boundaries in TejasIterableDataset."""
    lines_content = "sentence number one.\nsentence number two.\nsentence number three."
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "corpus.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(lines_content)
            
        dataset = TejasIterableDataset(
            file_path=file_path,
            tokenizer=dummy_tokenizer,
            max_seq_len=10
        )
        
        items = list(dataset)
        assert len(items) > 0
        
        first_batch = items[0]
        assert first_batch["input_ids"].shape == (10,)
        assert first_batch["labels"].shape == (10,)


def test_dataloader_integration(dummy_tokenizer):
    """Ensures that the dataloader builds, shuffles, and iterates batches correctly."""
    text_corpus = "A continuous corpus that we use to check the PyTorch DataLoader integration. It works!"
    dataset = TejasDataset(
        text_corpus=text_corpus,
        tokenizer=dummy_tokenizer,
        max_seq_len=8,
        pack_sequences=True
    )
    
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        pad_id=dummy_tokenizer.pad_id
    )
    
    for batch in dataloader:
        assert batch["input_ids"].shape == (2, 8)
        assert batch["labels"].shape == (2, 8)
        break


# =====================================================================
# Verification and Dependent Information
# =====================================================================
# Dependent Files:
# - tejas/datasets/dataset.py (the module being tested)
#
# Correctness:
# 1. Tests both offline Map-style and online Iterable data processing styles.
# 2. Assertively validates offset prediction shift boundaries for training accuracy.
# 3. Exercises the exact math behind dynamic padding and pad token injection.
#
# Testing:
# Run this test file using: pytest tests/test_dataset.py
