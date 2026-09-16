import numpy as np

from deepdream.image_utils import build_octave_sizes, compute_lost_detail, pil_resize


def test_build_octave_sizes_ends_exactly_at_base_size():
    sizes = build_octave_sizes((300, 450), num_octaves=4, octave_scale=1.4)
    assert sizes[-1] == (300, 450)


def test_build_octave_sizes_is_ascending():
    sizes = build_octave_sizes((300, 450), num_octaves=5, octave_scale=1.4)
    heights = [h for h, _ in sizes]
    widths = [w for _, w in sizes]
    assert heights == sorted(heights)
    assert widths == sorted(widths)


def test_build_octave_sizes_returns_requested_count():
    sizes = build_octave_sizes((200, 200), num_octaves=6, octave_scale=1.3)
    assert len(sizes) == 6


def test_build_octave_sizes_rejects_zero_octaves():
    try:
        build_octave_sizes((100, 100), num_octaves=0, octave_scale=1.4)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_pil_resize_changes_shape():
    img = np.random.rand(50, 80, 3).astype(np.float32)
    resized = pil_resize(img, (25, 40))
    assert resized.shape == (25, 40, 3)


def test_pil_resize_roundtrip_preserves_rough_content():
    img = np.zeros((40, 40, 3), dtype=np.float32)
    img[:20, :, :] = 1.0  # top half white, bottom half black
    up = pil_resize(img, (80, 80))
    down = pil_resize(up, (40, 40))
    assert np.mean(down[:20]) > np.mean(down[20:])


def test_compute_lost_detail_is_near_zero_when_no_downscale_happens():
    img = np.random.rand(60, 60, 3).astype(np.float32)
    detail = compute_lost_detail(img, small_size_hw=(60, 60), target_size_hw=(60, 60))
    assert np.allclose(detail, 0.0, atol=1e-5)


def test_compute_lost_detail_is_nonzero_when_downscaling_loses_information():
    rng = np.random.default_rng(0)
    img = rng.random((80, 80, 3)).astype(np.float32)  # high-frequency noise
    detail = compute_lost_detail(img, small_size_hw=(10, 10), target_size_hw=(80, 80))
    assert np.mean(np.abs(detail)) > 0.01
