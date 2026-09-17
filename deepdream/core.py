"""The actual gradient-ascent / octave loop
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from .image_utils import ArrayHWC, compute_lost_detail, pil_resize

GradFn = Callable[[ArrayHWC], ArrayHWC]


def gradient_ascent_step(
    image: ArrayHWC,
    compute_grad_fn: GradFn,
    step_size: float,
    jitter_px: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> ArrayHWC:
    """One step: (optionally jitter) -> gradient -> normalized step -> (un-jitter).

    Jitter (randomly rolling the image before the forward pass, then rolling
    back after) stops the output from locking onto a repeating grid pattern.
    """
    rng = rng or np.random.default_rng()
    ox, oy = 0, 0   # offsets
    if jitter_px > 0:
        ox, oy = rng.integers(-jitter_px, jitter_px + 1, size=2)
        image = np.roll(image, (oy, ox), axis=(0, 1))

    
    """
    Gradient Normalization.
    This is important because raw gradient magnitudes can be in extremes
    making step sizes unstable. Dividing by a scalar standard deviation
    scales the update to a manageable magnitude, and the tiny
    1e-8 prevents division by zero if the gradient is 
    constant or extremely tiny.
    """
    grad = compute_grad_fn(image)
    grad_std = float(np.std(grad)) + 1e-8
    image = image + step_size * (grad / grad_std)

    if jitter_px > 0:
        image = np.roll(image, (-oy, -ox), axis=(0, 1))
        
    return image


def run_octaves(
    original: ArrayHWC,
    octave_sizes: List[Tuple[int, int]],
    iterations_per_octave: int,
    compute_grad_fn: GradFn,
    step_size: float,
    jitter_px: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> ArrayHWC:
    """Run the full DeepDream loop across ascending octave sizes.
    octave_sizes must be ascending and end at the resolution you want the
    output in.
    """
    rng = rng or np.random.default_rng()
    img = pil_resize(original, octave_sizes[0])

    for i, size in enumerate(octave_sizes):
        print(f'\nOn Octave level {i}:')
        if i > 0:
            img = pil_resize(img, size) + compute_lost_detail(original, octave_sizes[i - 1], size)
        for _ in range(iterations_per_octave):
            print('. ', end='')
            img = gradient_ascent_step(img, compute_grad_fn, step_size, jitter_px, rng)
        
        print()
        
    return img
