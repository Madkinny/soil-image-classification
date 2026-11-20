"""Single source of truth for image loading.

Training, evaluation, the CLI, and the Streamlit app all call `load_rgb`, so there is no
train/serve skew in color order or resizing. Model-specific scaling lives inside the model.
"""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps


def to_rgb_array(img: Image.Image, size: int) -> np.ndarray:
    """Return an RGB uint8 array of shape (size, size, 3)."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_rgb(source: str | Path | BinaryIO, size: int) -> np.ndarray:
    with Image.open(source) as img:
        return to_rgb_array(img, size)


def load_batch(paths, size: int) -> np.ndarray:
    return np.stack([load_rgb(p, size) for p in paths])
