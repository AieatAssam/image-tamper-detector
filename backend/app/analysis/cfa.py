"""Colour-filter-array periodicity detector.

The statistic is intentionally modest: it measures the 2x2 variance pattern
left by interpolation in a Bayer-demosaiced image.  It is not a camera
classifier and must not be run on an image whose capture dimensions are gone.
"""

from time import perf_counter

import cv2
import numpy as np

from backend.app.analysis.base import (
    DetectorResult,
    DetectorState,
    ImageContext,
    to_probability,
)
from backend.app.analysis.adapters import _settings
from backend.app.analysis.exif import image_was_resized


class CfaDetector:
    id = "cfa"
    name = "CFA Periodicity"
    family = "sensor"
    applicable_formats = frozenset({"JPEG", "PNG", "WEBP", "TIFF"})
    produces_map = True
    description = "Measures Bayer colour-filter-array interpolation periodicity."
    limitations = [
        "Only meaningful on full-resolution CFA camera captures.",
        "Foveon, monochrome, and multi-shot sensors do not have this pattern.",
    ]
    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        if image_was_resized(ctx):
            return False, "CFA is not applicable after image resizing"
        if not ctx.exif and not _plausible_sensor_dimensions(ctx.width, ctx.height):
            return False, "CFA requires EXIF dimensions or plausible sensor dimensions"
        return True, "full-resolution dimensions are suitable for CFA analysis"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        config = _settings(self.id)
        applicable, reason = self.applicable(ctx)
        if not applicable:
            return DetectorResult(
                self.id,
                DetectorState.NOT_APPLICABLE,
                None,
                None,
                float(config["threshold"]),
                reason,
                {},
                None,
                _duration(started),
            )

        ratio, phase, ratio_map = self.measure(ctx.downscaled_rgb_uint8)
        # Higher ratio means less CFA structure and therefore MORE suspicious.
        score = to_probability(ratio, float(config["threshold"]), float(config["scale"]), bool(config["higher_is_worse"]))
        flagged = score >= 0.5
        return DetectorResult(
            self.id,
            DetectorState.APPLICABLE,
            score,
            flagged,
            float(config["threshold"]),
            f"CFA variance ratio {ratio:.3f} ({'exceeds' if flagged else 'is below'} {float(config['threshold']):.3f})",
            {"cfa_ratio": float(ratio), "phase": float(phase)},
            ratio_map,
            _duration(started),
        )

    def measure(self, rgb: np.ndarray) -> tuple[float, int, np.ndarray]:
        """Return the strongest-phase ratio, phase index, and ratio heatmap."""
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("CFA analysis requires an RGB image")
        if min(image.shape[:2]) < 32:
            raise ValueError("CFA analysis requires at least 32x32 pixels")

        candidates = [_phase_measure(image.astype(np.float32), phase) for phase in range(4)]
        phase, (ratio, ratio_map) = min(enumerate(candidates), key=lambda item: item[1][0])
        return float(ratio), phase, ratio_map


def _phase_measure(image: np.ndarray, phase: int) -> tuple[float, np.ndarray]:
    height, width = image.shape[:2]
    dy, dx = divmod(phase, 2)
    residuals = []
    for channel in range(3):
        plane = image[..., channel]
        # The centre-excluding 3x3 bilinear neighbourhood is the local
        # interpolation prediction; demosaicing leaves a 2x2 variance pattern.
        prediction = cv2.blur(plane, (3, 3), borderType=cv2.BORDER_REFLECT)
        residuals.append(plane - prediction)
    residual = np.mean(residuals, axis=0)
    local_mean = cv2.boxFilter(residual, cv2.CV_32F, (32, 32), normalize=True)
    local_second = cv2.boxFilter(residual * residual, cv2.CV_32F, (32, 32), normalize=True)
    local_variance = np.maximum(local_second - local_mean * local_mean, 0.0)

    classes = [
        float(np.var(residual[dy + row::2, dx + col::2]))
        for row, col in ((0, 0), (0, 1), (1, 0), (1, 1))
        if dy + row < height and dx + col < width
    ]
    ordered = np.sort(np.asarray(classes, dtype=np.float64))
    ratio = float(np.mean(ordered[:2]) / (np.mean(ordered[2:]) + 1e-8))

    # A small sliding ratio map is enough for the protocol's localisation map.
    step = 16
    rows = range(0, max(1, height - 31), step)
    cols = range(0, max(1, width - 31), step)
    values = []
    for y in rows:
        row_values = []
        for x in cols:
            tile = residual[y : y + 32, x : x + 32]
            variances = np.asarray(
                [
                    np.var(tile[row::2, col::2])
                    for row, col in ((0, 0), (0, 1), (1, 0), (1, 1))
                ],
                dtype=np.float64,
            )
            ordered_tile = np.sort(variances)
            row_values.append(np.mean(ordered_tile[:2]) / (np.mean(ordered_tile[2:]) + 1e-8))
        values.append(row_values)
    ratio_map = np.asarray(values, dtype=np.float32)
    ratio_map = cv2.resize(ratio_map, (width, height), interpolation=cv2.INTER_LINEAR)
    finite = np.nan_to_num(ratio_map, nan=1.0, posinf=1.0, neginf=0.0)
    return ratio, finite


def _plausible_sensor_dimensions(width: int, height: int) -> bool:
    if min(width, height) < 128 or width * height < 65_536:
        return False
    ratio = max(width, height) / min(width, height)
    return ratio <= 4.0


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
