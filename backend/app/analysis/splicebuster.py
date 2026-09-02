"""Paper-only Splicebuster residual co-occurrence implementation.

The method follows Cozzolino, Poggi, and Verdoliva, "Splicebuster: A new
blind image splicing detector", IEEE WIFS 2015, DOI 10.1109/WIFS.2015.7368565.
No reference implementation is used: third-order residual co-occurrences,
symmetry pooling, PCA, and the paper's two-Gaussian EM variant are computed
independently.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from time import perf_counter

import cv2
import numpy as np
from scipy.special import logsumexp

from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext, to_probability
from backend.app.analysis.qtable import jpeg_quality_proxy


BLOCK_SIZE = 128
BLOCK_STRIDE = 1
MAX_ANALYSIS_SIDE = 1024
QUANTIZATION_STEP = 2.0
TRUNCATION = 1
DEFAULT_THRESHOLD = 5.0
DEFAULT_SCALE = 2.0
MIN_ESTIMATED_JPEG_QUALITY = 80.0
_ALPHABET_SIZE = 2 * TRUNCATION + 1
_HISTOGRAM_DIMENSION = _ALPHABET_SIZE**4
_PCA_DIMENSION = 25
_EM_RUNS = 30
_EM_ITERATIONS = 40
_EM_TOLERANCE = 1e-5
_EM_MAX_SAMPLES = 4096
_DARK_MEAN_THRESHOLD = 16.0
_MAX_SATURATED_FRACTION = 0.01


def _code(values: tuple[int, ...]) -> int:
    code = 0
    for value in values:
        code = code * _ALPHABET_SIZE + value + TRUNCATION
    return code


def _symmetry_groups() -> tuple[tuple[int, ...], ...]:
    remaining = set(product(range(-TRUNCATION, TRUNCATION + 1), repeat=4))
    groups = []
    while remaining:
        values = min(remaining)
        orbit = {
            values,
            values[::-1],
            tuple(-value for value in values),
            tuple(-value for value in values[::-1]),
        }
        remaining.difference_update(orbit)
        groups.append(tuple(sorted(_code(value) for value in orbit)))
    return tuple(groups)


_SYMMETRY_GROUPS = _symmetry_groups()
_FEATURE_DIMENSION = len(_SYMMETRY_GROUPS) * 2


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
    features = np.zeros((len(rows), len(columns), _HISTOGRAM_DIMENSION), dtype=np.float32)
    row0 = rows[:, None]
    column0 = columns[None, :]
    for code in range(_HISTOGRAM_DIMENSION):
        integral = cv2.integral((codes == code).astype(np.uint8), sdepth=cv2.CV_32S)
        features[..., code] = (
            integral[row0 + block_height, column0 + block_width]
            - integral[row0, column0 + block_width]
            - integral[row0 + block_height, column0]
            + integral[row0, column0]
        )
    return features


def _pool_symmetries(histogram: np.ndarray) -> np.ndarray:
    return np.stack(
        [histogram[..., group].sum(axis=-1) for group in _SYMMETRY_GROUPS],
        axis=-1,
    )


def _pca(features: np.ndarray) -> np.ndarray:
    if len(features) < 2:
        return np.zeros((len(features), _PCA_DIMENSION), dtype=np.float32)
    centered = features - features.mean(axis=0, keepdims=True)
    if len(features) > _EM_MAX_SAMPLES:
        sample_indices = np.linspace(0, len(features) - 1, _EM_MAX_SAMPLES, dtype=np.int64)
        fit_features = centered[sample_indices]
    else:
        fit_features = centered
    _, _, right_singular_vectors = np.linalg.svd(fit_features, full_matrices=False)
    components = right_singular_vectors[: min(_PCA_DIMENSION, right_singular_vectors.shape[0])]
    reduced = centered @ components.T
    if reduced.shape[1] < _PCA_DIMENSION:
        reduced = np.pad(reduced, ((0, 0), (0, _PCA_DIMENSION - reduced.shape[1])))
    return reduced


def _block_features(gray: np.ndarray) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
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
    horizontal = _pool_symmetries(horizontal_histogram)
    vertical = _pool_symmetries(vertical_histogram)
    features = np.concatenate((horizontal, vertical), axis=2)
    features /= np.maximum(features.sum(axis=2, keepdims=True), 1.0)
    features = np.sqrt(features).astype(np.float32, copy=False).reshape(-1, _FEATURE_DIMENSION)

    integral = cv2.integral(gray.astype(np.float32), sdepth=cv2.CV_64F)
    row0 = rows[:, None]
    column0 = columns[None, :]
    block_means = (
        integral[row0 + BLOCK_SIZE, column0 + BLOCK_SIZE]
        - integral[row0, column0 + BLOCK_SIZE]
        - integral[row0 + BLOCK_SIZE, column0]
        + integral[row0, column0]
    ) / float(BLOCK_SIZE * BLOCK_SIZE)
    saturation = (gray <= 0.0) | (gray >= 255.0)
    saturation_integral = cv2.integral(saturation.astype(np.uint8), sdepth=cv2.CV_32S)
    saturated_fraction = (
        saturation_integral[row0 + BLOCK_SIZE, column0 + BLOCK_SIZE]
        - saturation_integral[row0, column0 + BLOCK_SIZE]
        - saturation_integral[row0 + BLOCK_SIZE, column0]
        + saturation_integral[row0, column0]
    ) / float(BLOCK_SIZE * BLOCK_SIZE)
    valid = (
        (block_means > _DARK_MEAN_THRESHOLD)
        & (block_means < 255.0)
        & (saturated_fraction <= _MAX_SATURATED_FRACTION)
    )
    flat_valid = valid.reshape(-1)
    return _pca(features[flat_valid]), (len(rows), len(columns)), flat_valid


def _regularized_covariance(features: np.ndarray, weights: np.ndarray, mean: np.ndarray) -> np.ndarray:
    total = max(float(weights.sum()), 1e-8)
    centered = features - mean
    covariance = (centered * weights[:, None]).T @ centered / total
    diagonal = np.diag(covariance)
    floor = max(float(np.median(diagonal)) * 1e-3, 1e-8)
    covariance.flat[:: covariance.shape[0] + 1] += floor
    return covariance


def _log_gaussian(features: np.ndarray, mean: np.ndarray, inverse: np.ndarray, log_determinant: float) -> np.ndarray:
    centered = features - mean
    quadratic = np.einsum("ni,ij,nj->n", centered, inverse, centered, optimize=True)
    dimension = features.shape[1]
    return -0.5 * (dimension * np.log(2.0 * np.pi) + log_determinant + quadratic)


def _em_scores(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the paper's two-Gaussian mixture and return distance ratio and distance."""
    if len(features) < 2:
        zeros = np.zeros(len(features), dtype=np.float64)
        return zeros, zeros

    if len(features) > _EM_MAX_SAMPLES:
        sample_indices = np.linspace(0, len(features) - 1, _EM_MAX_SAMPLES, dtype=np.int64)
        fit_features = features[sample_indices]
    else:
        fit_features = features
    weights = np.ones(len(fit_features), dtype=np.float64)
    global_mean = fit_features.mean(axis=0)
    global_covariance = _regularized_covariance(fit_features, weights, global_mean)
    rng = np.random.default_rng(2025)
    best_likelihood = -np.inf
    best_means = np.vstack((global_mean, global_mean))
    best_inverses = [np.linalg.pinv(global_covariance, rcond=1e-6)] * 2
    best_priors = np.array([0.5, 0.5])

    for _ in range(_EM_RUNS):
        initial_indices = rng.choice(len(fit_features), size=2, replace=False)
        means = fit_features[initial_indices].copy()
        covariances = [global_covariance.copy(), global_covariance.copy()]
        priors = np.array([0.5, 0.5])
        previous_likelihood = -np.inf
        for _ in range(_EM_ITERATIONS):
            log_probabilities = []
            for prior, mean, covariance in zip(priors, means, covariances):
                sign, log_determinant = np.linalg.slogdet(covariance)
                if sign <= 0:
                    covariance = global_covariance.copy()
                    covariances[len(log_probabilities)] = covariance
                    _, log_determinant = np.linalg.slogdet(covariance)
                inverse = np.linalg.pinv(covariance, rcond=1e-6)
                log_probabilities.append(np.log(prior) + _log_gaussian(fit_features, mean, inverse, log_determinant))
            log_probabilities = np.column_stack(log_probabilities)
            log_likelihoods = logsumexp(log_probabilities, axis=1)
            responsibilities = np.exp(log_probabilities - log_likelihoods[:, None])
            masses = responsibilities.sum(axis=0)
            for index in range(2):
                if masses[index] <= 1e-6:
                    masses[index] = 1.0
                    responsibilities[:, index] = 0.0
                means[index] = (responsibilities[:, index, None] * fit_features).sum(axis=0) / masses[index]
                covariances[index] = _regularized_covariance(fit_features, responsibilities[:, index], means[index])
            priors = np.clip(masses / len(fit_features), 1e-3, 1.0)
            priors /= priors.sum()
            likelihood = float(log_likelihoods.sum())
            if abs(likelihood - previous_likelihood) <= _EM_TOLERANCE:
                break
            previous_likelihood = likelihood

        final_probabilities = []
        final_inverses = []
        for prior, mean, covariance in zip(priors, means, covariances):
            _, log_determinant = np.linalg.slogdet(covariance)
            inverse = np.linalg.pinv(covariance, rcond=1e-6)
            final_inverses.append(inverse)
            final_probabilities.append(np.log(prior) + _log_gaussian(fit_features, mean, inverse, log_determinant))
        log_probabilities = np.column_stack(final_probabilities)
        likelihood = float(logsumexp(log_probabilities, axis=1).sum())
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            best_means = means.copy()
            best_inverses = [inverse.copy() for inverse in final_inverses]
            best_priors = priors.copy()

    genuine_index = int(np.argmax(best_priors))
    forged_index = 1 - genuine_index
    distances = []
    for index in (genuine_index, forged_index):
        centered = features - best_means[index]
        squared = np.einsum("ni,ij,nj->n", centered, best_inverses[index], centered, optimize=True)
        distances.append(np.sqrt(np.maximum(squared, 0.0)))
    genuine_distance, forged_distance = distances
    ratio = genuine_distance / np.maximum(forged_distance, 1e-6)
    return ratio, genuine_distance


def _em_mahalanobis(features: np.ndarray) -> np.ndarray:
    """Compatibility helper returning the genuine-model Mahalanobis distance."""
    return _em_scores(features)[1]


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
        "Uses the paper's Gaussian-Gaussian EM variant; the alternative Gaussian-uniform model is not selected.",
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
        features, block_shape, valid_blocks = _block_features(gray)
        if len(features) < 2:
            return DetectorResult(
                detector_id=self.id,
                state=DetectorState.NOT_APPLICABLE,
                score=None,
                flagged=None,
                threshold=self.threshold,
                reason="Splicebuster found fewer than two informative blocks after dark/saturated masking",
                metrics={},
                visualization=None,
                duration_ms=_duration(started),
            )
        ratios, distances = _em_scores(features)
        block_distances = np.zeros(valid_blocks.shape, dtype=np.float64)
        block_distances[valid_blocks] = ratios
        raw = float(np.max(ratios))
        score = to_probability(raw, self.threshold, self.scale, self.higher_is_worse)
        flagged = score >= 0.5
        return DetectorResult(
            detector_id=self.id,
            state=DetectorState.APPLICABLE,
            score=score,
            flagged=flagged,
            threshold=self.threshold,
            reason=f"maximum block Mahalanobis distance ratio {raw:.3f} {'exceeds' if flagged else 'is below'} the {self.threshold:.3f} threshold",
            metrics={
                "mahalanobis_max": float(np.max(distances)),
                "mahalanobis_mean": float(np.mean(distances)),
                "mahalanobis_median": float(np.median(distances)),
                "mahalanobis_ratio_max": raw,
                "block_count": float(len(distances)),
                "feature_dimension": float(features.shape[1]),
                "pre_pca_feature_dimension": float(_FEATURE_DIMENSION),
                "informative_block_count": float(len(features)),
                "analysis_width": float(image.shape[1]),
                "analysis_height": float(image.shape[0]),
                "block_size": float(BLOCK_SIZE),
                "block_stride": float(BLOCK_STRIDE),
            },
            visualization=_visualization(block_distances, block_shape, gray.shape),
            duration_ms=_duration(started),
        )


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
