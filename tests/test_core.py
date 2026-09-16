import numpy as np

from deepdream.core import gradient_ascent_step, run_octaves
from deepdream.image_utils import build_octave_sizes


def test_gradient_ascent_step_moves_image_in_gradient_direction():
    image = np.zeros((10, 10, 3), dtype=np.float32)
    grad_fn = lambda img: np.ones_like(img)  # constant "increase everything" gradient
    stepped = gradient_ascent_step(image, grad_fn, step_size=0.1, jitter_px=0)
    assert np.all(stepped > image)


def test_gradient_ascent_step_preserves_shape_with_jitter():
    image = np.random.rand(12, 16, 3).astype(np.float32)
    grad_fn = lambda img: np.random.rand(*img.shape).astype(np.float32)
    stepped = gradient_ascent_step(image, grad_fn, step_size=0.05, jitter_px=4)
    assert stepped.shape == image.shape


def test_zero_gradient_leaves_image_unchanged():
    image = np.random.rand(20, 20, 3).astype(np.float32)
    zero_grad_fn = lambda img: np.zeros_like(img)
    stepped = gradient_ascent_step(image, zero_grad_fn, step_size=1.0, jitter_px=0)
    assert np.allclose(stepped, image, atol=1e-6)


def test_run_octaves_output_matches_final_octave_resolution():
    original = np.random.rand(64, 64, 3).astype(np.float32)
    sizes = build_octave_sizes((64, 64), num_octaves=3, octave_scale=1.4)
    grad_fn = lambda img: np.zeros_like(img)  # no-op dreaming isolates plumbing/shape bugs
    result = run_octaves(original, sizes, iterations_per_octave=2, compute_grad_fn=grad_fn, step_size=0.1)
    assert result.shape == (64, 64, 3)


def test_run_octaves_with_zero_gradient_stays_close_to_original():
    original = np.random.rand(32, 32, 3).astype(np.float32)
    sizes = build_octave_sizes((32, 32), num_octaves=2, octave_scale=1.4)
    grad_fn = lambda img: np.zeros_like(img)
    result = run_octaves(original, sizes, iterations_per_octave=1, compute_grad_fn=grad_fn, step_size=0.1)
    # with no gradient push, output should resemble the original (modulo resize blur)
    assert np.mean(np.abs(result - original)) < 0.05


def test_run_octaves_with_constant_gradient_increases_brightness():
    original = np.full((32, 32, 3), 0.3, dtype=np.float32)
    sizes = build_octave_sizes((32, 32), num_octaves=2, octave_scale=1.4)
    grad_fn = lambda img: np.ones_like(img)
    result = run_octaves(original, sizes, iterations_per_octave=3, compute_grad_fn=grad_fn, step_size=0.05)
    assert np.mean(result) > np.mean(original)
