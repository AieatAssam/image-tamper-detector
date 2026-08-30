"""Paper-only Splicebuster Mahalanobis reimplementation.

The method follows Cozzolino, Poggi, and Verdoliva, "Splicebuster: A new
blind image splicing detector", IEEE WIFS 2015, DOI 10.1109/WIFS.2015.7368565.
No reference implementation is used: third-order residual co-occurrences are
computed independently and scored with the paper's single-Gaussian variant.
"""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

import cv2
import numpy as np

from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext, to_probability
from backend.app.analysis.qtable import jpeg_quality_proxy


BLOCK_SIZE = 128
BLOCK_STRIDE = 32
MAX_ANALYSIS_SIDE = 1024
QUANTIZATION_STEP = 2.0
TRUNCATION = 1
DEFAULT_THRESHOLD = 5.0
DEFAULT_SCALE = 2.0
MIN_ESTIMATED_JPEG_QUALITY = 80.0
_ALPHABET_SIZE = 2 * TRUNCATION + 1
_FEATURE_DIMENSION = _ALPHABET_SIZE**4


def _analysis_image(ctx: ImageContext) -> np.ndarray:
    """Use the shared bounded image, with a second cap for this feature pass."""
    image = ctx.downscaled_rgb_uint8
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= MAX_ANALYSIS_SIDE:
        return image
    ratio = MAX_ANALYSIS_SIDE / float(longest)
    return cv2.resize(
        image,
        (max(1, int(round(width * ratio))), max(1, int(round(height * ratio)))),
        interpolation=cv2.INTER_AREA,
    )


def _third_order_residuals(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return horizontal and vertical third-order derivative residuals."""
    kernel = np.array([[1.0, -3.0, 3.0, -1.0]], dtype=np.float32)
    horizontal = cv2.filter2D(gray, cv2.CV_32F, kernel, anchor=(1, 0), borderType=cv2.BORDER_REFLECT101)
    vertical = cv2.filter2D(gray, cv2.CV_32F, kernel.T, anchor=(0, 1), borderType=cv2.BORDER_REFLECT101)
    return horizontal, vertical


def _quantize(residual: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(residual / QUANTIZATION_STEP), -TRUNCATION, TRUNCATION).astype(np.int8)


def _cooccurrence_codes(quantized: np.ndarray, axis: int) -> np.ndarray:
    """Encode four consecutive quantised residuals without pixel-wise loops."""
    windows = np.lib.stride_tricks.sliding_window_view(quantized, 4, axis=axis)
    digits = windows.astype(np.int16) + TRUNCATION
    return (
        ((digits[..., 0] * _ALPHABET_SIZE + digits[..., 1]) * _ALPHABET_SIZE + digits[..., 2])
        * _ALPHABET_SIZE
        + digits[..., 3]
    ).astype(np.int16)


def _block_starts(length: int) -> np.ndarray:
    return np.arange(0, length - BLOCK_SIZE + 1, BLOCK_STRIDE, dtype=np.int32)


def _integral_histograms(
    codes: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    block_height: int,
    block_width: int,
) -> np.ndarray:
    """Aggregate every code bin over every block using integral images."""
    features = np.zeros((len(rows), len(columns), _FEATURE_DIMENSION), dtype=np.float32)
    row0 = rows[:, None]
    column0 = columns[None, :]
    for code in range(_FEATURE_DIMENSION):
        integral = cv2.integral((codes == code).astype(np.uint8), sdepth=cv2.CV_32S)
        features[..., code] = (
            integral[row0 + block_height, column0 + block_width]
            - integral[row0, column0 + block_width]
            - integral[row0 + block_height, column0]
            + integral[row0, column0]
        )
    return features


def _block_features(gray: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    horizontal, vertical = _third_order_residuals(gray)
    horizontal_codes = _cooccurrence_codes(_quantize(horizontal), axis=1)
    vertical_codes = _cooccurrence_codes(_quantize(vertical), axis=0)
    rows = _block_starts(gray.shape[0])
    columns = _block_starts(gray.shape[1])

    horizontal_histogram = _integral_histograms(
        horizontal_codes,
        rows,
        columns,
        BLOCK_SIZE,
        BLOCK_SIZE - 3,
    )
    vertical_histogram = _integral_histograms(
        vertical_codes,
        rows,
        columns,
        BLOCK_SIZE - 3,
        BLOCK_SIZE,
    )
    features = horizontal_histogram + vertical_histogram
    features /= np.maximum(features.sum(axis=2, keepdims=True), 1.0)
    return features.reshape(-1, _FEATURE_DIMENSION), (len(rows), len(columns))


def _mahalanobis(features: np.ndarray) -> np.ndarray:
    """Fit one regularised Gaussian and return each block's distance."""
    mean = features.mean(axis=0)
    centered = features - mean
    if len(features) < 2:
        return np.zeros(len(features), dtype=np.float64)

    covariance = centered.T @ centered / float(max(1, len(features) - 1))
    diagonal = np.diag(covariance)
    floor = max(float(np.median(diagonal)) * 1e-3, 1e-8)
    covariance.flat[:: covariance.shape[0] + 1] += floor
    inverse = np.linalg.pinv(covariance, rcond=1e-6)
    squared = np.einsum("ni,ij,nj->n", centered, inverse, centered, optimize=True)
    return np.sqrt(np.maximum(squared, 0.0))


def _visualization(distances: np.ndarray, block_shape: tuple[int, int], image_shape: tuple[int, int]) -> np.ndarray:
    block_map = distances.reshape(block_shape).astype(np.float32)
    if float(block_map.max()) <= float(block_map.min()):
        block_map.fill(0.0)
    else:
        block_map = cv2.normalize(block_map, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.resize(
        block_map.astype(np.uint8),
        (image_shape[1], image_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


class SpliceBusterDetector:
    id = "splicebuster"
    name = "Splicebuster Residual Co-occurrence"
    family = "processing_chain"
    applicable_formats = frozenset({"JPEG"})
    produces_map = True
    description = "Measures local processing-chain differences with residual co-occurrence statistics."
    limitations = [
        "Requires a JPEG quantization-table quality estimate of at least 80.",
        "Requires at least 256 pixels on both axes and a sufficiently heterogeneous image.",
        "A single Gaussian is a cheaper approximation to the paper's two-component EM model.",
        "Content boundaries, strong texture changes, and multiple camera pipelines can resemble a splice.",
    ]

    def __init__(self, settings: Mapping[str, float | bool] | None = None) -> None:
        settings = settings or {}
        self.threshold = float(settings.get("threshold", DEFAULT_THRESHOLD))
        self.scale = float(settings.get("scale", DEFAULT_SCALE))
        self.minimum_quality = float(settings.get("minimum_quality", MIN_ESTIMATED_JPEG_QUALITY))
        self.higher_is_worse = bool(settings.get("higher_is_worse", True))
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        if ctx.format not in self.applicable_formats:
            return False, "Splicebuster requires JPEG input to estimate recompression strength"
        quality = jpeg_quality_proxy(ctx)
        if quality is None:
            return False, "Splicebuster requires JPEG quantization tables to estimate recompression strength"
        if quality < self.minimum_quality:
            return False, f"estimated JPEG quality {quality:.0f} is below the Splicebuster minimum {self.minimum_quality:.0f}"
        image = ctx.downscaled_rgb_uint8
        longest = max(image.shape[:2])
        ratio = min(1.0, MAX_ANALYSIS_SIDE / float(longest))
        height = max(1, int(round(image.shape[0] * ratio)))
        width = max(1, int(round(image.shape[1] * ratio)))
        if min(height, width) < 2 * BLOCK_SIZE:
            return False, "Splicebuster requires both analysis dimensions to be at least 256px"
        return True, f"estimated JPEG quality {quality:.0f} meets the Splicebuster minimum and image is large enough"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        applicable, reason = self.applicable(ctx)
        if not applicable:
            return DetectorResult(
                detector_id=self.id,
                state=DetectorState.NOT_APPLICABLE,
                score=None,
                flagged=None,
                threshold=self.threshold,
                reason=reason,
                metrics={},
                visualization=None,
                duration_ms=_duration(started),
            )

        image = _analysis_image(ctx)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        features, block_shape = _block_features(gray)
        distances = _mahalanobis(features)
        raw = float(np.max(distances))
        score = to_probability(raw, self.threshold, self.scale, self.higher_is_worse)
        flagged = score >= 0.5
        return DetectorResult(
            detector_id=self.id,
            state=DetectorState.APPLICABLE,
            score=score,
            flagged=flagged,
            threshold=self.threshold,
            reason=f"maximum block Mahalanobis distance {raw:.3f} {'exceeds' if flagged else 'is below'} the {self.threshold:.3f} threshold",
            metrics={
                "mahalanobis_max": raw,
                "mahalanobis_mean": float(np.mean(distances)),
                "mahalanobis_median": float(np.median(distances)),
                "block_count": float(len(distances)),
                "feature_dimension": float(features.shape[1]),
                "analysis_width": float(image.shape[1]),
                "analysis_height": float(image.shape[0]),
                "block_size": float(BLOCK_SIZE),
                "block_stride": float(BLOCK_STRIDE),
            },
            visualization=_visualization(distances, block_shape, gray.shape),
            duration_ms=_duration(started),
        )


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
