"""Training-free neighboring-pixel statistic derived from the NPR paper."""

from time import perf_counter

import cv2
import numpy as np

from backend.app.analysis.adapters import _settings
from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext, to_probability


MAX_ANALYSIS_SIDE = 1024


def _analysis_image(image: np.ndarray) -> np.ndarray:
    longest = max(image.shape[:2])
    if longest <= MAX_ANALYSIS_SIDE:
        return image
    ratio = MAX_ANALYSIS_SIDE / float(longest)
    return cv2.resize(
        image,
        (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))),
        interpolation=cv2.INTER_AREA,
    )


class NprDetector:
    id = "npr"
    name = "Neighboring Pixel Relationships"
    family = "learned"
    applicable_formats = frozenset({"JPEG", "PNG", "WEBP", "TIFF"})
    produces_map = True
    description = "Measures degenerate 2x2 neighboring-pixel relationships without a learned model."
    limitations = [
        "This is a training-free statistic derived from NPR, not the paper's trained detector.",
        "Its score is exploratory and is not comparable with the paper's headline accuracy.",
    ]

    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        if min(ctx.width, ctx.height) < 4:
            return False, "NPR requires at least 4x4 pixels"
        return True, "2x2 neighboring-pixel statistic is applicable"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        config = _settings(self.id)
        applicable, reason = self.applicable(ctx)
        if not applicable:
            return DetectorResult(
                self.id, DetectorState.NOT_APPLICABLE, None, None,
                float(config["threshold"]), reason, {}, None, _duration(started),
            )
        score_raw, visualization, metrics = self.measure(_analysis_image(ctx.downscaled_rgb_uint8))
        score = to_probability(
            score_raw,
            float(config["threshold"]),
            float(config["scale"]),
            bool(config["higher_is_worse"]),
        )
        flagged = score >= 0.5
        return DetectorResult(
            self.id,
            DetectorState.APPLICABLE,
            score,
            flagged,
            float(config["threshold"]),
            f"NPR training-free statistic {score_raw:.3f}",
            metrics,
            visualization,
            _duration(started),
        )

    def measure(self, rgb: np.ndarray) -> tuple[float, np.ndarray, dict[str, float]]:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 4:
            raise ValueError("NPR requires an RGB image at least 4x4 pixels")
        image = image.astype(np.float32) / 255.0

        reference_pixels = image[1:, 1:]
        differences = (
            image[:-1, :-1] - reference_pixels,
            image[:-1, 1:] - reference_pixels,
            image[1:, :-1] - reference_pixels,
        )
        value_sum = sum(difference.sum(axis=2) for difference in differences)
        value_square_sum = sum(np.square(difference).sum(axis=2) for difference in differences)
        intra_variance = value_square_sum / 12.0 - np.square(value_sum / 12.0)
        reference = reference_pixels.mean(axis=2)
        inter_variance = float(np.var(reference))
        ratio = float(np.mean(intra_variance) / (inter_variance + 1e-8))
        near_constant = float(np.mean(intra_variance <= (1.0 / 255.0) ** 2))

        quantized = np.concatenate(
            tuple(np.rint(difference * 255.0).astype(np.int16).ravel() for difference in differences)
            + (np.zeros(reference_pixels.size, dtype=np.int16),)
        ) + 255
        counts = np.bincount(quantized, minlength=511).astype(np.float64)
        probabilities = counts[counts > 0] / counts.sum()
        difference_entropy = float(-(probabilities * np.log2(probabilities)).sum())
        normalized_entropy = difference_entropy / np.log2(511.0)

        ratio_suspicion = 1.0 / (1.0 + ratio)
        entropy_suspicion = 1.0 - normalized_entropy
        statistic = float((near_constant + ratio_suspicion + entropy_suspicion) / 3.0)
        visualization = cv2.resize(
            (near_constant_map(intra_variance) * 255).astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return statistic, visualization, {
            "intra_inter_variance_ratio": ratio,
            "near_constant_fraction": near_constant,
            "difference_entropy": difference_entropy,
            "normalized_difference_entropy": normalized_entropy,
            "npr_statistic": statistic,
        }


def near_constant_map(intra_variance: np.ndarray) -> np.ndarray:
    return intra_variance <= (1.0 / 255.0) ** 2


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
