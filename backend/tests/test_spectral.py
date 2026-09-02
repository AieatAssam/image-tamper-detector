from io import BytesIO

import numpy as np
from PIL import Image

from backend.app.analysis.base import ImageContext
from backend.app.analysis import spectral
from backend.app.analysis.spectral import SpectralPeakDetector


def _context(gray: np.ndarray) -> ImageContext:
    output = BytesIO()
    Image.fromarray(gray.astype(np.uint8), "L").convert("RGB").save(output, format="PNG")
    return ImageContext(output.getvalue())


def test_nearest_neighbour_upsampling_has_periodic_peaks():
    rng = np.random.default_rng(11)
    small = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    upsampled = np.repeat(np.repeat(small, 4, axis=0), 4, axis=1)
    peak_sigma, peak_count, visualization = SpectralPeakDetector().measure(upsampled)
    assert peak_sigma > 4.0
    assert peak_count > 0
    assert visualization.shape == (512, 512)


def test_spectral_is_stable_across_png_encoding():
    rng = np.random.default_rng(13)
    image = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    detector = SpectralPeakDetector()
    first = detector.measure(image)[0]
    second = detector.measure(np.asarray(Image.fromarray(image).convert("L")))[0]
    assert abs(first - second) < 0.05


def test_spectral_does_not_apply_jpeg_grid_mask(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("JPEG lattice masking is not part of this variant")

    monkeypatch.setattr(spectral, "_jpeg_grid_mask", fail, raising=False)
    image = np.random.default_rng(19).integers(0, 256, (64, 64), dtype=np.uint8)
    SpectralPeakDetector().measure(image)


def test_jpeg_grid_does_not_change_spectral_score():
    rng = np.random.default_rng(17)
    image = Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8), "RGB")
    encoded = []
    for quality in (60, 95):
        output = BytesIO()
        image.save(output, format="JPEG", quality=quality)
        encoded.append(np.asarray(Image.open(BytesIO(output.getvalue())).convert("L")))
    detector = SpectralPeakDetector()
    assert abs(detector.measure(encoded[0])[0] - detector.measure(encoded[1])[0]) < 0.1


def test_small_image_is_not_applicable():
    result = SpectralPeakDetector().run(_context(np.zeros((16, 16), dtype=np.uint8)))
    assert result.state.value == "not_applicable"
