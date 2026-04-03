#!/usr/bin/env python3
"""
Shared defaults for paired-edit training, inference, and validation.
"""

from __future__ import annotations

from pathlib import Path


PAIRED_EDIT_PROMPT = (
    "professional manicure photo retouch, clean cuticles, clean sidewalls, "
    "refined nail shape, preserve original nail design, natural skin texture, realistic hand photo"
)

DEFAULT_CORE_DATASET_DIR = Path("dataset/paired_edit_core_v1")
DEFAULT_BASELINE_PAIR_IDS = ["pair_0005", "pair_0015", "pair_0009", "pair_0040"]
DEFAULT_BASELINE_INPUT = Path("inputs/paired_edit/IMG_7588.JPG")
