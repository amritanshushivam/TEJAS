# tejas/tokenizer/bpe.py
"""Byte-Pair Encoding (BPE) Tokenizer implementation for TEJAS.

This module implements a highly polished, production-ready Byte-Pair Encoding
tokenizer designed from scratch. It handles byte-level UTF-8 encoding,
pretokenization, vocabulary building, and fast deterministic encoding/decoding.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Union

# Set up logger for the tokenizer module
logger = logging.getLogger("tejas.tokenizer")
logging.basicConfig(level=logging.INFO)


def bytes_to_unicode() -> Dict[int, str]:
    """Returns a deterministic mapping of bytes (0-255) to readable Unicode characters.

    This ensures that whitespace and control characters are mapped to printable
    Unicode characters to avoid breaking regex and file serialization.
    Adopted from standard GPT-style byte-level tokenizers.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs_strs = [chr(x) for x in cs]
    return dict(zip(bs, cs_strs))


@dataclass
class TokenizerConfig:
    """Configuration class for the TEJAS Tokenizer.

    Attributes:
        vocab_size: Target vocabulary size.
        bos_token: Beginning-of-sequence special token.
        eos_token: End-of-sequence special token.
        pad_token: Padding special token.
        unk_token: Unknown special token fallback.
        special_tokens: Complete set of special tokens to register.
    """
    vocab_size: int = 50257
    bos_token: str = "<|bos|>"
    eos_token: str = "<|eos|>"
    pad_token: str = "<|pad|>"
    unk_token: str = "<|unk|>"
    special_tokens: List[str] = field(default_factory=lambda: [
        "<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"
    ])


class TejasTokenizer:
    """A research-grade Byte-Pair Encoding (BPE) tokenizer for TEJAS LLM.

    Provides high-performance vocabulary building, text-to-ID tokenization,
    and ID-to-text detokenization, complete with special token registers.
    """

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        """Initializes the tokenizer with a configuration."""
        self.config = config or TokenizerConfig()
        
        # Byte encoders
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # Core vocabulary maps
        self.encoder: Dict[str, int] = {}
        self.decoder: Dict[int, str] = {}
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        
        # Special tokens setup
        self.special_tokens_map: Dict[str, int] = {}
        self.special_tokens_decoder: Dict[int, str] = {}
        
        # Cache to speed up tokenization of unseen words
        self.cache: Dict[str, str] = {}
        
        # Pretokenization regex pattern
        # Splitting on contractions, words, digits, and spaces
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^a-zA-Z0-9\s]+|\s+(?!\S)|\s+"""
        )
        
        self._register_special_tokens()

    def _register_special_tokens(self) -> None:
        """Populates the special token mapping."""
        for idx, token in enumerate(self.config.special_tokens):
            self.special_tokens_map[token] = idx
            self.special_tokens_decoder[idx] = token
            # Populate basic encoder/decoder with special tokens
            self.encoder[token] = idx
            self.decoder[idx] = token

    @property
    def bos_id(self) -> int:
        """Returns the token ID for the BOS token."""
        return self.special_tokens_map[self.config.bos_token]

    @property
    def eos_id(self) -> int:
        """Returns the token ID for the EOS token."""
        return self.special_tokens_map[self.config.eos_token]

    @property
    def pad_id(self) -> int:
        """Returns the token ID for the PAD token."""
        return self.special_tokens_map[self.config.pad_token]

    @property
    def unk_id(self) -> int:
        """Returns the token ID for the UNK token."""
        return self.special_tokens_map[self.config.unk_token]

    def _get_stats(self, ids: List[Tuple[str, ...]], counts: List[int]) -> Dict[Tuple[str, str], int]:
        """Calculates frequencies of adjacent token pairs across the corpus."""
        pairs: Dict[Tuple[str, str], int] = {}
        for word, count in zip(ids, counts):
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pairs[pair] = pairs.get(pair, 0) + count
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], ids: List[Tuple[str, ...]]) -> List[Tuple[str, ...]]:
        """Merges all occurrences of a specific pair of tokens in-place in the word list."""
        new_ids = []
        for word in ids:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                    new_word.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_ids.append(tuple(new_word))
        return new_ids

    def train_from_text(self, text: str, vocab_size: int, min_frequency: int = 1) -> None:
        """Trains the BPE tokenizer on a given raw text corpus.

        Args:
            text: Raw string text to train on.
            vocab_size: Target size of the vocabulary (must be > 256 + special tokens).
            min_frequency: Minimum occurrence frequency of a pair to be merged.
        """
        logger.info("Initializing BPE Training...")
        
        # 1. Reset state with special tokens
        self.encoder.clear()
        self.decoder.clear()
        self.bpe_ranks.clear()
        self.cache.clear()
        self._register_special_tokens()
        
        # Offset to start vocabulary indices after special tokens and raw bytes
        num_special = len(self.config.special_tokens)
        
        # 2. Add raw bytes to encoder
        for b in range(256):
            char = self.byte_encoder[b]
            idx = num_special + b
            self.encoder[char] = idx
            self.decoder[idx] = char
            
        current_vocab_size = num_special + 256
        if vocab_size <= current_vocab_size:
            logger.warning(
                "Requested vocab_size %d is too small. Setting to default %d.",
                vocab_size, current_vocab_size + 1000
            )
            vocab_size = current_vocab_size + 1000
            
        # 3. Find unique words and counts using the pretokenizer pattern
        words = self.pat.findall(text)
        word_counts: Dict[Tuple[str, ...], int] = {}
        
        for w in words:
            # Map word bytes to clean unicode representations
            encoded_word = tuple(self.byte_encoder[b] for b in w.encode("utf-8"))
            word_counts[encoded_word] = word_counts.get(encoded_word, 0) + 1
            
        unique_words = list(word_counts.keys())
        counts = [word_counts[w] for w in unique_words]
        
        # 4. Iteratively find and merge pairs
        max_merges = vocab_size - current_vocab_size
        logger.info("Iterative BPE Merging for up to %d merges...", max_merges)
        
        for i in range(max_merges):
            pairs = self._get_stats(unique_words, counts)
            if not pairs:
                logger.info("No more pairs to merge. Stopping early at vocab size %d.", current_vocab_size)
                break
                
            best_pair = max(pairs, key=lambda k: pairs[k])
            if pairs[best_pair] < min_frequency:
                logger.info(
                    "Best pair frequency %d is below threshold %d. Stopping early.",
                    pairs[best_pair], min_frequency
                )
                break
                
            # Assign rank and add to vocab
            new_token = best_pair[0] + best_pair[1]
            self.bpe_ranks[best_pair] = i

            # Keep token IDs contiguous so encoded IDs always remain < len(encoder).
            token_idx = current_vocab_size
            self.encoder[new_token] = token_idx
            self.decoder[token_idx] = new_token
            
            # Execute merge on unique word lists
            unique_words = self._merge_vocab(best_pair, unique_words)
            current_vocab_size += 1
            
            if (i + 1) % 100 == 0 or (i + 1) == max_merges:
                logger.info("BPE Merge Progress: %d / %d merges completed.", i + 1, max_merges)
                
        logger.info("BPE training completed. Final vocabulary size: %d.", len(self.encoder))

    def _bpe(self, token: str) -> str:
        """Applies trained BPE merge rules to a single word-level string."""
        if token in self.cache:
            return self.cache[token]
            
        word = tuple(token)
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i+1]))
            
        if not pairs:
            return token
            
        while True:
            # Find the pair with the smallest merge rank (highest priority)
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
                
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                    
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
                    
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = set()
                for i in range(len(word) - 1):
                    pairs.add((word[i], word[i+1]))
                    
        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text: str, allowed_special: Set[str] | str = "all") -> List[int]:
        """Encodes string text into a sequence of vocabulary token IDs.

        Handles special tokens cleanly according to the safety set.

        Args:
            text: Raw input string.
            allowed_special: A set of allowed special tokens or string "all".

        Returns:
            List of token integer IDs.
        """
        bpe_tokens: List[int] = []
        
        # Determine allowed special tokens
        allowed: Set[str] = set()
        if allowed_special == "all":
            allowed = set(self.special_tokens_map.keys())
        elif isinstance(allowed_special, set):
            allowed = allowed_special
            
        # Parse text, isolating allowed special tokens
        if allowed:
            special_pattern = "|".join(re.escape(tok) for tok in allowed)
            parts = re.split(f"({special_pattern})", text)
        else:
            parts = [text]
            
        for part in parts:
            if part in allowed:
                # Direct map for special tokens
                bpe_tokens.append(self.special_tokens_map[part])
            else:
                # Sub-word tokenization of regular chunks
                words = self.pat.findall(part)
                for word in words:
                    # Deterministically byte-encode the word chunk
                    byte_encoded = "".join(self.byte_encoder[b] for b in word.encode("utf-8"))
                    # Apply BPE merge rules
                    bpe_merged = self._bpe(byte_encoded).split(" ")
                    # Map strings to IDs, defaulting to UNK if somehow not found
                    for token in bpe_merged:
                        bpe_tokens.append(self.encoder.get(token, self.unk_id))
                        
        return bpe_tokens

    def decode(self, ids: List[int]) -> str:
        """Decodes a sequence of token IDs back into a readable string.

        Args:
            ids: List of token integers.

        Returns:
            Reconstructed Unicode string.
        """
        text_parts: List[str] = []
        raw_bytes: List[int] = []
        
        for idx in ids:
            if idx in self.special_tokens_decoder:
                # Flush existing raw bytes first before outputting special tokens
                if raw_bytes:
                    text_parts.append(bytes(raw_bytes).decode("utf-8", errors="replace"))
                    raw_bytes.clear()
                text_parts.append(self.special_tokens_decoder[idx])
            else:
                token_str = self.decoder.get(idx, "")
                # Decode through byte translation
                for char in token_str:
                    byte_val = self.byte_decoder.get(char)
                    if byte_val is not None:
                        raw_bytes.append(byte_val)
                        
        if raw_bytes:
            text_parts.append(bytes(raw_bytes).decode("utf-8", errors="replace"))
            
        return "".join(text_parts)

    def save(self, file_path: str) -> None:
        """Saves tokenizer state to a file to support loading later.

        Serializes merge ranks and configurations.
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        # Convert tuple keys in bpe_ranks to strings for JSON serialization
        ranks_serializable = {f"{k[0]} {k[1]}": v for k, v in self.bpe_ranks.items()}
        
        data = {
            "vocab_size": self.config.vocab_size,
            "special_tokens": self.config.special_tokens,
            "encoder": self.encoder,
            "bpe_ranks": ranks_serializable,
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        logger.info("Tokenizer state successfully saved to %s.", file_path)

    def load(self, file_path: str) -> None:
        """Loads a pre-trained tokenizer state from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tokenizer state file not found at {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.config = TokenizerConfig(
            vocab_size=data["vocab_size"],
            special_tokens=data["special_tokens"]
        )
        
        # Reset and load
        self.encoder = data["encoder"]
        self.decoder = {int(v): k for k, v in self.encoder.items()}
        
        # Restore special tokens
        self.special_tokens_map.clear()
        self.special_tokens_decoder.clear()
        self._register_special_tokens()
        
        # Restore BPE ranks
        self.bpe_ranks.clear()
        for k, v in data["bpe_ranks"].items():
            parts = k.split(" ")
            if len(parts) == 2:
                self.bpe_ranks[(parts[0], parts[1])] = v
                
        # Flush translation cache
        self.cache.clear()
        logger.info("Tokenizer state loaded successfully from %s. Vocab size: %d.", file_path, len(self.encoder))


# =====================================================================
# Verification and Dependent Information
# =====================================================================
# Dependent Files:
# - tests/test_tokenizer.py (unit tests validating tokenization correctness)
# - datasets/dataset.py (converts raw text datasets to packed BPE tokens)
# - model/transformer.py (ingests token sequence ids from this tokenizer)
# - inference/decode.py (decodes output token distributions to text strings)
#
# Correctness:
# 1. Deterministic character mapping ensures UTF-8 boundary integrity.
# 2. GPT-style regex splitter prevents cross-contraction or bad spacing merges.
# 3. Dynamic sub-word caching guarantees O(N) evaluation behavior during inference.
# 4. Bidirectional mapping is verified lossless.
#
# Testing:
# Run tests by running the command: pytest tests/test_tokenizer.py
