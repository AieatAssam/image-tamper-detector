"""Colour-filter-array periodicity detector.

The intermediate-value/grid-pattern statistic is an independent
reimplementation of Bammey et al., *Image forgery detection using a Bayer
pattern analysis*, Image Processing On Line 11 (2021), article 355:
https://doi.org/10.5201/ipol.2021.355.  The paper-only AGPL reference source
is not used.  It is not a camera classifier and must not be run on an image
whose capture dimensions are gone.
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
from backend.app.analysis.exif import _first, _metadata, _number


CFA_WINDOW_SIZE = 32
CFA_STRIDE = 16
CFA_CONFIDENCE_THRESHOLD = 0.2


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
        ctx.pil_image
        if ctx.format != "JPEG":
            return False, "CFA requires a strict real-camera JPEG"
        metadata = _metadata(ctx)
        if _first(metadata, 0x010F) is None or _first(metadata, 0x0110) is None:
            return False, "CFA requires EXIF camera Make/Model and strict dimensions"
        capture_width = _number(_first(metadata, 0xA002))
        capture_height = _number(_first(metadata, 0xA003))
        if capture_width != ctx.width or capture_height != ctx.height:
            return False, "CFA is not applicable without strict matching PixelXDimension/PixelYDimension evidence"
        return True, "strict real-camera dimensions match the decoded JPEG"

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

        ratio, phase, ratio_map = self.measure(ctx.rgb_uint8)
        if phase < 0:
            return DetectorResult(
                self.id,
                DetectorState.NOT_APPLICABLE,
                None,
                None,
                float(config["threshold"]),
                "CFA pattern could not be resolved from the full-resolution image",
                {"cfa_ratio": float(ratio), "phase": float(phase)},
                ratio_map,
                _duration(started),
            )
        # Higher inconsistency means the local Bayer pattern disagrees with
        # the dominant image pattern and is therefore MORE suspicious.
        score = to_probability(ratio, float(config["threshold"]), float(config["scale"]), bool(config["higher_is_worse"]))
        flagged = ratio > CFA_CONFIDENCE_THRESHOLD
        return DetectorResult(
            self.id,
            DetectorState.APPLICABLE,
            score,
            flagged,
            float(config["threshold"]),
            f"CFA intermediate-value inconsistency {ratio:.3f} "
            f"({'exceeds' if flagged else 'is below'} {CFA_CONFIDENCE_THRESHOLD:.3f})",
            {
                "cfa_ratio": float(ratio),
                "phase": float(phase),
                "confidence_threshold": CFA_CONFIDENCE_THRESHOLD,
            },
            ratio_map,
            _duration(started),
        )

    def measure(self, rgb: np.ndarray) -> tuple[float, int, np.ndarray]:
        """Return Bammey's inconsistency ratio, dominant phase, and map.

        The method uses intermediate-value masks to identify the Bayer
        diagonal and red/blue arrangement globally, then marks local windows
        whose pattern disagrees with that dominant arrangement.  This is an
        independent reimplementation of the paper-only IPOL method; the
        AGPL reference source is not used.
        """
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("CFA analysis requires an RGB image")
        height, width = image.shape[:2]
        if min(height, width) < 32:
            raise ValueError("CFA analysis requires at least 32x32 pixels")

        image = image[..., :3].astype(np.float32)
        height -= height % 2
        width -= width % 2
        image = image[:height, :width]
        masks = tuple(_intermediate_values(image[..., index]) for index in range(3))
        dominant = _pattern_measure_from_masks(masks)
        if dominant is None:
            return 0.0, -1, np.zeros((height, width), dtype=np.float32)
        dominant_pattern, dominant_diagonal, _, _, _ = dominant
        window = min(CFA_WINDOW_SIZE, height, width)
        window -= window % 2
        if window < 8:
            return 0.0, -1, np.zeros((height, width), dtype=np.float32)
        stride = min(CFA_STRIDE, window)
        y_starts = _window_starts(height, window, stride)
        x_starts = _window_starts(width, window, stride)
        patterns = np.full((len(y_starts), len(x_starts)), "", dtype="<U4")
        diagonals = np.full((len(y_starts), len(x_starts)), "", dtype="<U6")
        main_deltas = np.zeros(patterns.shape, dtype=np.float32)
        diagonal_deltas = np.zeros(patterns.shape, dtype=np.float32)
        for row, top in enumerate(y_starts):
            for column, left in enumerate(x_starts):
                local = _pattern_measure_from_masks(
                    tuple(mask[top:top + window, left:left + window] for mask in masks)
                )
                if local is None:
                    continue
                pattern, diagonal, main_delta, diagonal_delta, _ = local
                patterns[row, column] = pattern
                diagonals[row, column] = diagonal
                main_deltas[row, column] = main_delta
                diagonal_deltas[row, column] = diagonal_delta

        main_confidence = _connected_confidence(patterns, dominant_pattern, main_deltas)
        diagonal_confidence = _connected_confidence(diagonals, dominant_diagonal, diagonal_deltas)
        confidence = np.maximum(main_confidence, diagonal_confidence)
        heatmap = np.zeros((height, width), dtype=np.float32)
        for row, top in enumerate(y_starts):
            for column, left in enumerate(x_starts):
                value = confidence[row, column]
                heatmap[top:top + window, left:left + window] = np.maximum(
                    heatmap[top:top + window, left:left + window], value
                )

        phase = _pattern_phase(dominant_pattern)
        ratio = float(np.max(confidence, initial=0.0))
        return ratio, phase, heatmap


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)


def _intermediate_values(channel: np.ndarray) -> np.ndarray:
    """Return the mean horizontal/vertical intermediate-value mask."""
    channel = np.asarray(channel, dtype=np.float32)
    center = channel[2:-2, 2:-2]
    left, right = channel[2:-2, 1:-3], channel[2:-2, 3:-1]
    up, down = channel[1:-3, 2:-2], channel[3:-1, 2:-2]
    horizontal = ((left <= center) & (center <= right)) | ((right <= center) & (center <= left))
    vertical = ((up <= center) & (center <= down)) | ((down <= center) & (center <= up))
    values = np.zeros_like(channel, dtype=np.float32)
    values[2:-2, 2:-2] = (horizontal.astype(np.float32) + vertical.astype(np.float32)) * 0.5
    return values


def _pattern_measure(image: np.ndarray) -> tuple[str, str, float, float, float] | None:
    """Estimate a Bayer arrangement and its normalized confidence."""
    height, width = image.shape[:2]
    height -= height % 2
    width -= width % 2
    if min(height, width) < 8:
        return None
    masks = tuple(_intermediate_values(image[:height, :width, index]) for index in range(3))
    return _pattern_measure_from_masks(masks)


def _pattern_measure_from_masks(
    masks: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> tuple[str, str, float, float, float] | None:
    """Estimate a pattern from precomputed intermediate-value masks."""
    red, green, blue = (mask[2:-2, 2:-2] for mask in masks)
    height, width = red.shape
    height -= height % 2
    width -= width % 2
    red, green, blue = red[:height, :width], green[:height, :width], blue[:height, :width]
    blocks = float((height // 2) * (width // 2))
    diagonal = (
        (green[1::2, 0::2].sum() + green[0::2, 1::2].sum())
        - (green[0::2, 0::2].sum() + green[1::2, 1::2].sum())
    ) / (2.0 * blocks)
    first = (
        (red[0::2, 0::2].sum() + blue[1::2, 1::2].sum())
        - (red[1::2, 1::2].sum() + blue[0::2, 0::2].sum())
    ) / (2.0 * blocks)
    second = (
        (red[1::2, 0::2].sum() + blue[0::2, 1::2].sum())
        - (red[0::2, 1::2].sum() + blue[1::2, 0::2].sum())
    ) / (2.0 * blocks)
    if diagonal == 0.0:
        return None
    diagonal_name = "dotgg" if diagonal < 0 else "gdotg"
    color_delta = first if diagonal_name == "dotgg" else second
    if color_delta == 0.0:
        return None
    colors = ("rggb", "bggr") if diagonal_name == "dotgg" else ("grbg", "gbrg")
    pattern = colors[0] if color_delta < 0 else colors[1]
    return pattern, diagonal_name, float(color_delta), float(diagonal), float(max(abs(diagonal), abs(color_delta)))


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    starts = list(range(0, length - window + 1, stride))
    if not starts or starts[-1] != length - window:
        starts.append(length - window)
    return starts


def _connected_confidence(
    labels: np.ndarray, global_label: str, deltas: np.ndarray
) -> np.ndarray:
    confidence = np.zeros(labels.shape, dtype=np.float32)
    for label in np.unique(labels):
        if not label or label == global_label:
            continue
        component_mask = (labels == label).astype(np.uint8)
        count, connected = cv2.connectedComponents(component_mask, connectivity=8)
        for component in range(1, count):
            selected = connected == component
            confidence[selected] = float(np.max(np.abs(deltas[selected])))
    return confidence


def _pattern_phase(pattern: str) -> int:
    return {"rggb": 0, "bggr": 1, "grbg": 2, "gbrg": 3}[pattern]
