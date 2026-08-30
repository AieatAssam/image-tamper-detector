"""Local resampling inconsistency detector, reimplemented from Kirchner (2008).

The implementation uses Kirchner's fixed 3x3 linear predictor.  It measures
periodic structure in the two-dimensional DFT of the absolute prediction
residual, then scores disagreement between bounded overlapping image blocks.
The source implementation is not used here.
"""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

import cv2
import numpy as np

from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext, to_probability


_MAX_ANALYSIS_SIDE = 1024
_BLOCK_SIZE = 128
_BLOCK_STRIDE = 64
_MIN_IMAGE_SIDE = _BLOCK_SIZE * 4
_PREDICTOR = np.array(
    [
        [-0.25, 0.50, -0.25],
        [0.50, 0.00, 0.50],
        [-0.25, 0.50, -0.25],
    ],
    dtype=np.float32,
)

# These are provisional until the integrating agent adds this detector to the
# shared calibration file.  The raw value is dimensionless block disagreement.
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


def _positions(length: int) -> list[int]:
    last = max(0, length - _BLOCK_SIZE)
    values = list(range(0, last + 1, _BLOCK_STRIDE))
    if not values or values[-1] != last:
        values.append(last)
    return values


def _absolute_residual(gray: np.ndarray) -> np.ndarray:
    image = np.asarray(gray, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("resampling analysis requires a grayscale image")
    prediction = cv2.filter2D(image, cv2.CV_32F, _PREDICTOR, borderType=cv2.BORDER_REFLECT101)
    return np.abs(image - prediction)


def _peak_to_background(block: np.ndarray) -> float:
    centered = block.astype(np.float32) - float(np.mean(block))
    window = np.outer(np.hanning(block.shape[0]), np.hanning(block.shape[1])).astype(np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered * window)))
    height, width = spectrum.shape
    yy, xx = np.indices((height, width))
    radius = np.hypot(yy - height // 2, xx - width // 2)
    valid = (radius > 4) & (radius <= min(height, width) * 0.45)
    background = float(np.median(spectrum[valid]))
    if background <= 1e-8:
        return 0.0
    # The upper-tail percentile is a stable peak estimate for finite noisy blocks.
    return float(np.percentile(spectrum[valid], 99.5) / background)


def _measure(gray: np.ndarray) -> tuple[float, np.ndarray, dict[str, float]]:
    image = np.asarray(gray)
    if image.ndim != 2:
        raise ValueError("resampling analysis requires a grayscale image")
    if min(image.shape) < _MIN_IMAGE_SIDE:
        raise ValueError("resampling analysis requires at least 256x256 pixels")

    absolute_residual = _absolute_residual(image)
    rows = _positions(image.shape[0])
    columns = _positions(image.shape[1])
    block_peaks = np.asarray(
        [
            _peak_to_background(absolute_residual[row : row + _BLOCK_SIZE, column : column + _BLOCK_SIZE])
            for row in rows
            for column in columns
        ],
        dtype=np.float32,
    ).reshape(len(rows), len(columns))
    median = float(np.median(block_peaks))
    deviations = np.abs(block_peaks - median)
    # Robust disagreement is intentionally used instead of the median level:
    # a uniform web resize is not tampering, while a resized pasted block is.
    disagreement = float(np.percentile(deviations, 75) / max(abs(median), 1e-6))
    metrics = {
        "local_inconsistency": disagreement,
        "block_peak_median": median,
        "block_peak_p90": float(np.percentile(block_peaks, 90)),
        "block_peak_max": float(np.max(block_peaks)),
        "block_count": float(block_peaks.size),
    }
    return disagreement, block_peaks, metrics


def _visualization(block_peaks: np.ndarray, image_shape: tuple[int, int], output_shape: tuple[int, int]) -> np.ndarray:
    median = float(np.median(block_peaks))
    deviations = np.abs(block_peaks - median).astype(np.float32)
    if float(deviations.max()) <= 1e-8:
        heatmap = np.zeros_like(deviations, dtype=np.uint8)
    else:
        heatmap = cv2.normalize(deviations, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(heatmap, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_NEAREST)


class ResamplingDetector:
    id = "resampling"
    name = "Local Resampling Inconsistency"
    family = "geometric"
    applicable_formats = frozenset({"JPEG", "PNG", "WEBP", "TIFF"})
    produces_map = True
    description = "Measures block-to-block disagreement in fixed-predictor resampling spectra."
    limitations = [
        "A globally resized image is not tampering and is intentionally not flagged by this score.",
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
            return False, "resampling requires both analysis dimensions to be at least 512px for stable block disagreement"
        return True, "image is large enough for bounded local resampling analysis"

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
        raw, block_peaks, metrics = _measure(gray)
        score = to_probability(raw, self.threshold, self.scale, self.higher_is_worse)
        flagged = score >= 0.5
        visualization = _visualization(block_peaks, gray.shape, (ctx.height, ctx.width))
        return DetectorResult(
            self.id,
            DetectorState.APPLICABLE,
            score,
            flagged,
            self.threshold,
            f"local resampling inconsistency {raw:.3f} ({'exceeds' if flagged else 'is below'} {self.threshold:.3f})",
            {**metrics, "analysis_width": float(gray.shape[1]), "analysis_height": float(gray.shape[0])},
            visualization,
            _duration(started),
        )


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
