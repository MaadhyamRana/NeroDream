"""image-array utilities for the DeepDream pipeline. helper functions.
""" 

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

ArrayHWC = np.ndarray  # shape (H, W, 3), float32, values in [0, 1]


def load_image(path: str, max_size: Optional[int] = None) -> ArrayHWC:
    """Load an image from disk as an HWC float32 array in [0, 1].
    downscale image if its longest side exceeds max_size if given.
    """
    img = Image.open(path).convert("RGB")
    if max_size is not None:
        w, h = img.size
        longest = max(w, h)
        if longest > max_size:
            scale = max_size / longest
            
            # Lanczos filter, reduces aliasing and preserves edges upon downscaling
            img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def save_image(array: ArrayHWC, path: str) -> None:
    """Save the image file after clipping out-of-range values."""
    clipped = np.clip(array, 0.0, 1.0)
    Image.fromarray((clipped * 255.0).astype(np.uint8)).save(path)


def pil_resize(array: ArrayHWC, size_hw: Tuple[int, int]) -> ArrayHWC:
    """Resize an HWC float32 array to size_hw = (height, width) using Lanczos."""
    h, w = size_hw
    img = Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8))
    img = img.resize((w, h), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def build_octave_sizes(
    base_size_hw: Tuple[int, int], num_octaves: int, octave_scale: float
) -> List[Tuple[int, int]]:
    """Return `num_octaves` (height, width) pairs, ascending, smallest to largest.

    The last entry is always exactly base_size_hw -- DeepDream should finish
    at the caller's original resolution, not a rounding-drifted approximation
    of it, so we force that rather than trust the geometric series.
    """
    if num_octaves < 1:
        raise ValueError("num_octaves must be >= 1")
    base_h, base_w = base_size_hw
    sizes = []
    for i in range(num_octaves):
        power = num_octaves - 1 - i
        factor = octave_scale**power
        sizes.append((max(1, round(base_h / factor)), max(1, round(base_w / factor))))
    sizes[-1] = (base_h, base_w)
    
    """
    at this point, sizes looks something like:
    [(91, 91), (118, 118), (154, 154), (200, 200)]
    when given base_size_hw=(200, 200), num_octaves=4, octave_scale=1.3
    so:
    [(200/1.3^3, 200/1.3^3), (200/1.3^2, 200/1.3^2), (200/1.3, 200/1.3), (200, 200)]
    """
    
    return sizes


def compute_lost_detail(
    original: ArrayHWC, small_size_hw: Tuple[int, int], target_size_hw: Tuple[int, int]
) -> ArrayHWC:
    """High-frequency detail present in `original` at target_size_hw that a
    naive downscale-to-small_size_hw-then-upscale-back would destroy.

    Adding this back after each octave upscale is what keeps DeepDream output
    from looking progressively blurrier at each larger octave
    """
    downscaled = pil_resize(original, small_size_hw)
    reconstructed = pil_resize(downscaled, target_size_hw)
    true_at_target = pil_resize(original, target_size_hw)
    return true_at_target - reconstructed
