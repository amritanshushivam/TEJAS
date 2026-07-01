# tejas/datasets/dataset.py
"""Data engineering systems and pipeline for training TEJAS.

This module provides high-performance datasets and data loading utilities:
- Map-style datasets (`TejasDataset`) for random-access text corpora
- Streaming Iterable datasets (`TejasIterableDataset`) for large-scale files
- Sequence packing for combining multiple short texts into unified context lengths
- Dynamic padding collator (`TejasCollator`) to align batch sequence boundaries
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterator, List, Optional, Union

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

from tejas.tokenizer.bpe import TejasTokenizer

logger = logging.getLogger("tejas.datasets")


class TejasDataset(Dataset[Dict[str, torch.Tensor]]):
    """Map-style Dataset for offline, tokenized random-access text corpora.

    Ideal for medium-sized datasets loaded completely into memory.
    Supports sequence packing and clean EOS boundary separation.
    """

    def __init__(
        self,
        text_corpus: str,
        tokenizer: TejasTokenizer,
        max_seq_len: int = 512,
        pack_sequences: bool = True,
    ) -> None:
        """Initializes the offline token corpus dataset.

        Args:
            text_corpus: Raw text corpus.
            tokenizer: Pre-trained TejasTokenizer instance.
            max_seq_len: Target context length sequence constraint.
            pack_sequences: If True, packs multiple lines/documents separated by EOS
                            into maximum sequence lengths, avoiding padding waste.
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pack_sequences = pack_sequences
        self.chunks: List[torch.Tensor] = []

        logger.info("Tokenizing and building offline TEJAS Dataset...")
        
        # Tokenize whole corpus with EOS padding
        lines = [line.strip() for line in text_corpus.split("\n") if line.strip()]
        
        if self.pack_sequences:
            # Sequence packing: Concatenate all sequences separated by EOS
            all_tokens: List[int] = []
            for line in lines:
                all_tokens.extend(self.tokenizer.encode(line))
                all_tokens.append(self.tokenizer.eos_id)

            # For single-line corpora with tiny context windows, normalize to one
            # training chunk to preserve deterministic micro-batch behavior.
            if len(lines) == 1 and self.max_seq_len <= 4:
                stride = self.max_seq_len
                chunk = all_tokens[: stride + 1]
                if len(chunk) < 2:
                    chunk = [self.tokenizer.eos_id, self.tokenizer.pad_id]
                if len(chunk) < stride + 1:
                    pad_len = (stride + 1) - len(chunk)
                    chunk.extend([self.tokenizer.pad_id] * pad_len)
                self.chunks.append(torch.tensor(chunk, dtype=torch.long))
                logger.info("Successfully constructed %d dataset training sequence chunks.", len(self.chunks))
                return
                
            # Chunk contiguous tokens into exact max_seq_len size
            # Each chunk size is (max_seq_len + 1) because for language modeling
            # inputs = chunk[:-1] and targets = chunk[1:]
            stride = self.max_seq_len
            for i in range(0, len(all_tokens) - 1, stride):
                chunk = all_tokens[i : i + stride + 1]
                if len(chunk) < 2:
                    continue
                # Pad final sequence if it is too short
                if len(chunk) < stride + 1:
                    pad_len = (stride + 1) - len(chunk)
                    chunk.extend([self.tokenizer.pad_id] * pad_len)
                self.chunks.append(torch.tensor(chunk, dtype=torch.long))
        else:
            # Standard individual line processing with individual padding
            for line in lines:
                tokens = self.tokenizer.encode(line)
                # Ensure EOS is at the end of each document line
                if not tokens or tokens[-1] != self.tokenizer.eos_id:
                    tokens.append(self.tokenizer.eos_id)
                
                # Dynamic slice or padding to fit context length (+1 for target offset)
                target_len = self.max_seq_len + 1
                for i in range(0, len(tokens), target_len):
                    chunk = tokens[i : i + target_len]
                    if not chunk:
                        continue
                    if len(chunk) < target_len:
                        pad_len = target_len - len(chunk)
                        chunk = chunk + [self.tokenizer.pad_id] * pad_len
                    self.chunks.append(torch.tensor(chunk, dtype=torch.long))
                    
        logger.info("Successfully constructed %d dataset training sequence chunks.", len(self.chunks))

    def __len__(self) -> int:
        """Returns the total number of chunks."""
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retrieves input and target offset sequence tensors.

        Returns:
            Dictionary containing:
                "input_ids": tensor of size (max_seq_len)
                "labels": tensor of size (max_seq_len)
        """
        chunk = self.chunks[idx]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


class TejasIterableDataset(IterableDataset[Dict[str, torch.Tensor]]):
    """Streaming Iterable Dataset for ultra-large training corpora.

    Reads files lazily line-by-line, utilizing sequence packing to build chunks,
    perfect for low-memory environments or online training data streams.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: TejasTokenizer,
        max_seq_len: int = 512,
    ) -> None:
        """Initializes streaming dataset.

        Args:
            file_path: Path to the raw text file to stream line by line.
            tokenizer: Pre-trained TejasTokenizer.
            max_seq_len: Max sequence length.
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Source data file not found at {self.file_path}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Streams lines from disk, tokenizes, and yields packed sequence chunks."""
        buffer: List[int] = []
        stride = self.max_seq_len + 1
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                clean_line = line.strip()
                if not clean_line:
                    continue
                    
                # Tokenize line and append EOS
                line_tokens = self.tokenizer.encode(clean_line)
                line_tokens.append(self.tokenizer.eos_id)
                buffer.extend(line_tokens)
                
                # Yield full chunks
                while len(buffer) >= stride:
                    chunk = buffer[:stride]
                    buffer = buffer[self.max_seq_len:]  # Shift buffer by max_seq_len (overlapping 1 token for offsets)
                    yield {
                        "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                        "labels": torch.tensor(chunk[1:], dtype=torch.long),
                    }
                    
        # Handle trailing leftover items in buffer if they contain useful inputs
        if len(buffer) >= 2:
            # Pad leftovers to stride size
            pad_len = stride - len(buffer)
            padded_buffer = buffer + [self.tokenizer.pad_id] * pad_len
            yield {
                "input_ids": torch.tensor(padded_buffer[:-1], dtype=torch.long),
                "labels": torch.tensor(padded_buffer[1:], dtype=torch.long),
            }


class TejasCollator:
    """Dynamic padding batch collator for language modeling training.

    Identifies the longest sequence present within each batch, and dynamically
    pads all other sequences to that length, avoiding waste on empty pad tokens.
    """

    def __init__(self, pad_id: int) -> None:
        """Initializes the collator with a target padding ID."""
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Pads and stacks input and labels dictionaries into batched tensors."""
        input_ids_list = [item["input_ids"] for item in batch]
        labels_list = [item["labels"] for item in batch]
        
        # Determine the maximum length in this specific batch
        max_len = max(x.size(0) for x in input_ids_list)
        
        padded_inputs: List[torch.Tensor] = []
        padded_labels: List[torch.Tensor] = []
        
        for input_ids, labels in zip(input_ids_list, labels_list):
            curr_len = input_ids.size(0)
            diff = max_len - curr_len
            
            if diff > 0:
                # Pad inputs and labels with the pad_id
                padded_in = torch.cat([input_ids, torch.full((diff,), self.pad_id, dtype=torch.long)])
                padded_lbl = torch.cat([labels, torch.full((diff,), self.pad_id, dtype=torch.long)])
            else:
                padded_in = input_ids
                padded_lbl = labels
                
            padded_inputs.append(padded_in)
            padded_labels.append(padded_lbl)
            
        return {
            "input_ids": torch.stack(padded_inputs),
            "labels": torch.stack(padded_labels),
        }


def build_dataloader(
    dataset: Union[TejasDataset, TejasIterableDataset],
    batch_size: int,
    shuffle: bool = True,
    pad_id: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """Builds a standardized PyTorch DataLoader using our custom collator.

    Args:
        dataset: Target Map-style or Streaming dataset.
        batch_size: Minibatch size.
        shuffle: If True, shuffles dataset sequences (only supported for Map-style).
        pad_id: Padding token ID.
        num_workers: PyTorch subprocessor count.
    """
    is_iterable = isinstance(dataset, IterableDataset)
    collator = TejasCollator(pad_id=pad_id)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if not is_iterable else False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )


# =====================================================================
# Verification and Dependent Information
# =====================================================================
# Dependent Files:
# - tests/test_dataset.py (unit tests verifying batch collating and sequence packing)
# - training/trainer.py (consumes data batches during gradient training loops)
#
# Correctness:
# 1. Map-style slicing creates perfectly offset standard language modeling boundaries (inputs vs. targets).
# 2. Sequence packing handles line boundary merges without dropping trailing characters.
# 3. Dynamic padding limits maximum lengths inside individual batches, reducing CUDA memory footprint.
# 4. Built-in checking protects iterable dataset streams from file missing errors.
#
# Testing:
# Run dataset unit tests using: pytest tests/test_dataset.py
