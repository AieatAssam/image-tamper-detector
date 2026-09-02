"""Fast resampling detector reimplemented from Kirchner (2008).

The implementation uses Kirchner's fixed 3x3 linear predictor, p-map, and
cumulative-periodogram decision. The source implementation is not used here.
"""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

import cv2
import numpy as np

from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext, to_probability


_MAX_ANALYSIS_SIDE = 1024
_MIN_IMAGE_SIDE = 256
_PREDICTOR = np.array(
    [
        [-0.25, 0.50, -0.25],
        [0.50, 0.00, 0.50],
        [-0.25, 0.50, -0.25],
    ],
    dtype=np.float32,
)

# These retain the existing calibration interface until the integrating agent
# refits the changed raw statistic. The paper determines this threshold
# empirically from a false-acceptance target.
_SCORE_THRESHOLD = 0.115
_SCORE_SCALE = 0.04


def _analysis_gray(ctx: ImageContext) -> np.ndarray:
    image = cv2.cvtColor(ctx.downscaled_rgb_uint8, cv2.COLOR_RGB2GRAY)
    longest = max(image.shape[:2])
    if longest > _MAX_ANALYSIS_SIDE:
        ratio = _MAX_ANALYSIS_SIDE / longest
        image = cv2.resize(
            image,
            (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _absolute_residual(gray: np.ndarray) -> np.ndarray:
    image = np.asarray(gray, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("resampling analysis requires a grayscale image")
    prediction = cv2.filter2D(image, cv2.CV_32F, _PREDICTOR, borderType=cv2.BORDER_REFLECT101)
    return np.abs(image - prediction)


def _p_map(gray: np.ndarray) -> np.ndarray:
    image = np.asarray(gray, dtype=np.float32)
    residual = _absolute_residual(image)
    if float(np.max(image)) > 1.0:
        residual /= 255.0
    return np.exp(-np.square(residual)).astype(np.float32)


def _contrast_spectrum(p_map: np.ndarray) -> np.ndarray:
    image = np.asarray(p_map, dtype=np.float32)
    height, width = image.shape
    spatial_y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    spatial_x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    spatial_radius = np.hypot(spatial_y[:, None], spatial_x[None, :])
    radial_window = np.ones_like(spatial_radius, dtype=np.float32)
    transition = (spatial_radius >= 0.75) & (spatial_radius <= np.sqrt(2.0))
    radial_window[spatial_radius > np.sqrt(2.0)] = 0.0
    radial_window[transition] = 0.5 + 0.5 * np.cos(
        np.pi * (spatial_radius[transition] - 0.75) / (np.sqrt(2.0) - 0.75)
    )
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image * radial_window)))
    frequencies_y = np.fft.fftshift(np.fft.fftfreq(height))
    frequencies_x = np.fft.fftshift(np.fft.fftfreq(width))
    radius = 2.0 * np.hypot(frequencies_y[:, None], frequencies_x[None, :])

    highpass = 0.5 - 0.5 * np.cos(np.pi * np.minimum(radius, np.sqrt(2.0)) / np.sqrt(2.0))
    contrasted = spectrum * highpass
    maximum = float(np.max(contrasted))
    if maximum <= 1e-12:
        return np.zeros_like(contrasted, dtype=np.float32)
    return ((contrasted / maximum) ** 4 * maximum).astype(np.float32)


def _cumulative_periodogram(p_map: np.ndarray) -> tuple[float, np.ndarray]:
    spectrum = _contrast_spectrum(p_map)
    height, width = spectrum.shape
    unshifted = np.fft.ifftshift(spectrum)
    first_quadrant = np.square(unshifted[: height // 2 + 1, : width // 2 + 1])
    first_quadrant[0, 0] = 0.0
    cumulative = np.cumsum(np.cumsum(first_quadrant, axis=0), axis=1)
    total = float(cumulative[-1, -1])
    if total <= 1e-12:
        return 0.0, np.zeros_like(cumulative, dtype=np.float32)
    cumulative = (cumulative / total).astype(np.float32)
    gradient_x = cv2.Sobel(cumulative, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(cumulative, cv2.CV_32F, 0, 1, ksize=3)
    delta = float(np.max(np.hypot(gradient_x, gradient_y)))
    return delta, cumulative


def _measure(gray: np.ndarray) -> tuple[float, np.ndarray, dict[str, float]]:
    image = np.asarray(gray)
    if image.ndim != 2:
        raise ValueError("resampling analysis requires a grayscale image")
    if min(image.shape) < _MIN_IMAGE_SIDE:
        raise ValueError("resampling analysis requires at least 256x256 pixels")

    delta, cumulative = _cumulative_periodogram(_p_map(image))
    metrics = {
        "periodogram_delta": delta,
        "periodogram_max": float(np.max(cumulative)),
        "periodogram_height": float(cumulative.shape[0]),
        "periodogram_width": float(cumulative.shape[1]),
    }
    return delta, cumulative, metrics


def _visualization(periodogram: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    if float(periodogram.max()) <= float(periodogram.min()):
        heatmap = np.zeros_like(periodogram, dtype=np.uint8)
    else:
        heatmap = cv2.normalize(periodogram, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.resize(heatmap, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_NEAREST)


class ResamplingDetector:
    id = "resampling"
    name = "Resampling Spectral Periodicity"
    family = "geometric"
    applicable_formats = frozenset({"JPEG", "PNG", "WEBP", "TIFF"})
    produces_map = True
    description = "Measures cumulative-periodogram gradients in a fixed-predictor resampling p-map."
    limitations = [
        "Small, heavily compressed, or smoothly textured regions may not leave a measurable periodic signal.",
        "This is a resampling cue and cannot establish who edited an image or distinguish every interpolation method.",
    ]

    def __init__(self, settings: Mapping[str, float | bool] | None = None) -> None:
        config = settings or {}
        self.threshold = float(config.get("threshold", _SCORE_THRESHOLD))
        self.scale = float(config.get("scale", _SCORE_SCALE))
        self.higher_is_worse = bool(config.get("higher_is_worse", True))
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        if ctx.format and ctx.format not in self.applicable_formats:
            return False, f"resampling does not support decoded format {ctx.format}"
        image_shape = ctx.downscaled_rgb_uint8.shape[:2]
        longest = max(image_shape)
        if longest > _MAX_ANALYSIS_SIDE:
            ratio = _MAX_ANALYSIS_SIDE / longest
            image_shape = tuple(max(1, round(dimension * ratio)) for dimension in image_shape)
        if min(image_shape) < _MIN_IMAGE_SIDE:
            return False, "resampling requires both analysis dimensions to be at least 256px for spectral analysis"
        return True, "image is large enough for bounded resampling spectral analysis"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        applicable, reason = self.applicable(ctx)
        if not applicable:
            return DetectorResult(
                self.id,
                DetectorState.NOT_APPLICABLE,
                None,
                None,
                self.threshold,
                reason,
                {},
                None,
                _duration(started),
            )

        gray = _analysis_gray(ctx)
        raw, cumulative, metrics = _measure(gray)
        score = to_probability(raw, self.threshold, self.scale, self.higher_is_worse)
        flagged = score >= 0.5
        visualization = _visualization(cumulative, (ctx.height, ctx.width))
        return DetectorResult(
            self.id,
            DetectorState.APPLICABLE,
            score,
            flagged,
            self.threshold,
            f"cumulative-periodogram gradient {raw:.3f} ({'exceeds' if flagged else 'is below'} {self.threshold:.3f})",
            {**metrics, "analysis_width": float(gray.shape[1]), "analysis_height": float(gray.shape[0])},
            visualization,
            _duration(started),
        )


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
