from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.resampling import ResamplingDetector, _absolute_residual, _contrast_spectrum, _cumulative_periodogram, _p_map


def _jpeg(array: np.ndarray, quality: int = 95) -> bytes:
    output = BytesIO()
    Image.fromarray(array).save(output, format="JPEG", quality=quality)
    return output.getvalue()


def _textured_image(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    cv2.rectangle(image, (80, 80), (430, 430), (30, 190, 100), 3)
    return image


def test_uses_kirchner_fixed_predictor_and_absolute_residual():
    gray = np.arange(25, dtype=np.uint8).reshape(5, 5)
    expected = np.abs(
        gray.astype(np.float32)
        - cv2.filter2D(
            gray.astype(np.float32), cv2.CV_32F, np.array(
                [[-.25, .5, -.25], [.5, 0, .5], [-.25, .5, -.25]], dtype=np.float32
            ), borderType=cv2.BORDER_REFLECT101
        )
    )
    np.testing.assert_allclose(_absolute_residual(gray), expected)


def test_global_resize_is_not_tampering_signal():
    original = cv2.resize(_textured_image(5), (768, 768), interpolation=cv2.INTER_CUBIC)
    resized = cv2.resize(original, (512, 512), interpolation=cv2.INTER_CUBIC)
    result = ResamplingDetector().run(ImageContext(_jpeg(resized)))

    assert result.state is DetectorState.APPLICABLE
    assert result.score is not None and np.isfinite(result.score)
    assert np.isfinite(result.metrics["periodogram_delta"])
    assert result.visualization is not None
    assert result.visualization.shape == (512, 512)


def test_resampling_reports_cumulative_periodogram_statistic():
    host = _textured_image(7)
    source = host[40:200, 40:200]
    transformed = cv2.resize(source, (230, 230), interpolation=cv2.INTER_CUBIC)
    tampered = host.copy()
    tampered[160:390, 160:390] = transformed
    result = ResamplingDetector().run(ImageContext(_jpeg(tampered)))

    assert result.state is DetectorState.APPLICABLE
    assert result.score is not None and np.isfinite(result.score)
    assert result.metrics["periodogram_height"] == 257.0
    assert result.metrics["periodogram_width"] == 257.0


def test_small_image_is_not_applicable():
    result = ResamplingDetector().run(ImageContext(_jpeg(np.zeros((128, 128, 3), dtype=np.uint8))))

    assert result.state is DetectorState.NOT_APPLICABLE
    assert result.score is None
    assert result.flagged is None


def test_kirchner_p_map_uses_fixed_controls():
    gray = np.arange(25, dtype=np.uint8).reshape(5, 5)
    residual = _absolute_residual(gray) / 255.0

    np.testing.assert_allclose(_p_map(gray), np.exp(-np.square(residual)))


def test_cumulative_periodogram_uses_first_quadrant_and_sobel_gradient():
    p_map = np.random.default_rng(16).random((256, 256), dtype=np.float32)

    delta, cumulative = _cumulative_periodogram(p_map)

    assert np.isfinite(delta) and delta >= 0.0
    assert cumulative.shape == (129, 129)
    assert np.all(cumulative >= 0.0) and np.all(cumulative <= 1.0)
    assert cumulative[-1, -1] == 1.0


def test_resampling_contrast_applies_paper_window_before_fft():
    p_map = np.random.default_rng(17).random((8, 8), dtype=np.float32)
    spatial_y = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    spatial_x = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    radius = np.hypot(spatial_y[:, None], spatial_x[None, :])
    window = np.ones_like(radius, dtype=np.float32)
    transition = (radius >= 0.75) & (radius <= np.sqrt(2.0))
    window[radius > np.sqrt(2.0)] = 0.0
    window[transition] = 0.5 + 0.5 * np.cos(
        np.pi * (radius[transition] - 0.75) / (np.sqrt(2.0) - 0.75)
    )
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(p_map * window)))
    frequencies = np.fft.fftshift(np.fft.fftfreq(8))
    frequency_radius = 2.0 * np.hypot(frequencies[:, None], frequencies[None, :])
    highpass = 0.5 - 0.5 * np.cos(np.pi * np.minimum(frequency_radius, np.sqrt(2.0)) / np.sqrt(2.0))
    contrasted = spectrum * highpass
    maximum = float(contrasted.max())
    expected = ((contrasted / maximum) ** 4 * maximum).astype(np.float32)

    np.testing.assert_allclose(_contrast_spectrum(p_map), expected, rtol=1e-5, atol=1e-6)
