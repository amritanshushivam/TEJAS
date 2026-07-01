# tejas/utils/helpers.py
"""Utility helper tools and logging setups for TEJAS.

Provides:
- Precise model parameter size tracking and calculations
- Structured directory and cache setups
- Global console and file logging configurations
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Optional, Union

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, Union[int, float]]:
    """Calculates active trainable parameters and total model parameter sizes.

    Args:
        model: Target PyTorch module.

    Returns:
        Dictionary containing parameter stats.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    # Calculate parameter size in megabytes (assumes float32 / 4 bytes)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "size_megabytes": size_mb,
    }


def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configures structured console logging formats.

    Args:
        name: Module namespace identifier.
        log_file: Optional file path to mirror logs on disk.
        level: Minimum logger reporting level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Avoid duplicating handlers
    
    # Define clean uniform formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Stream Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler if file is provided
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


def check_gpu_availability() -> Dict[str, Union[bool, str]]:
    """Audits system environment for CUDA and GPU devices.

    Returns:
        Status details map.
    """
    cuda_available = torch.cuda.is_available()
    device_name = "CPU"
    
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        
    return {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "device_type": "cuda" if cuda_available else "cpu"
    }


# =====================================================================
# Verification and Dependent Information
# =====================================================================
# Dependent Files:
# - main.py (initializes logging and tracks parameter limits at start)
#
# Correctness:
# 1. Float precision multipliers adjust based on actual element storage sizes.
# 2. Handler clearing prevents console logs doubling-up during interactive runs.
#
# Testing:
# Handled via downstream integration checks in main program entries.
