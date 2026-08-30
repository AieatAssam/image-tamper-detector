"""Colour-filter-array periodicity detector.

The intermediate-value/grid-pattern statistic is an independent
reimplementation of Bammey et al., *Image forgery detection using a Bayer
pattern analysis*, Image Processing On Line 11 (2021), article 355:
https://doi.org/10.5201/ipol.2021.355.  The paper-only AGPL reference source
is not used.  It is not a camera classifier and must not be run on an image
whose capture dimensions are gone.
"""

from time import perf_counter

import numpy as np

from backend.app.analysis.base import (
    DetectorResult,
    DetectorState,
    ImageContext,
    to_probability,
)
from backend.app.analysis.adapters import _settings
from backend.app.analysis.exif import _first, _metadata, _number


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
        # Higher inconsistency means the local Bayer pattern disagrees with
        # the dominant image pattern and is therefore MORE suspicious.
        score = to_probability(ratio, float(config["threshold"]), float(config["scale"]), bool(config["higher_is_worse"]))
        flagged = score >= 0.5
        return DetectorResult(
            self.id,
            DetectorState.APPLICABLE,
            score,
            flagged,
            float(config["threshold"]),
            f"CFA intermediate-value inconsistency {ratio:.3f} ({'exceeds' if flagged else 'is below'} {float(config['threshold']):.3f})",
            {"cfa_ratio": float(ratio), "phase": float(phase)},
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
        dominant = _pattern_measure(image)
        if dominant is None:
            return 0.0, -1, np.zeros((height, width), dtype=np.float32)
        dominant_pattern, dominant_confidence = dominant
        heatmap = np.zeros((height, width), dtype=np.float32)
        window = min(128, height - (height % 2), width - (width % 2))
        stride = max(16, window // 2)
        inconsistent = []
        for top in range(0, max(1, height - window + 1), stride):
            for left in range(0, max(1, width - window + 1), stride):
                bottom = min(height, top + window)
                right = min(width, left + window)
                patch = image[top:bottom, left:right]
                local = _pattern_measure(patch)
                if local is None:
                    continue
                local_pattern, local_confidence = local
                if local_pattern != dominant_pattern:
                    value = max(local_confidence, dominant_confidence)
                    heatmap[top:bottom, left:right] = np.maximum(
                        heatmap[top:bottom, left:right], value
                    )
                    inconsistent.append(value)

        phase = _pattern_phase(dominant_pattern)
        ratio = float(np.mean(inconsistent)) if inconsistent else 0.0
        return ratio, phase, np.asarray(heatmap, dtype=np.float32)


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)


def _intermediate_values(channel: np.ndarray) -> np.ndarray:
    """Return the mean horizontal/vertical intermediate-value mask."""
    channel = np.asarray(channel, dtype=np.float32)
    center = channel[1:-1, 1:-1]
    left, right = channel[1:-1, :-2], channel[1:-1, 2:]
    up, down = channel[:-2, 1:-1], channel[2:, 1:-1]
    horizontal = ((left <= center) & (center <= right)) | ((right <= center) & (center <= left))
    vertical = ((up <= center) & (center <= down)) | ((down <= center) & (center <= up))
    values = np.zeros_like(channel, dtype=np.float32)
    values[1:-1, 1:-1] = (horizontal.astype(np.float32) + vertical.astype(np.float32)) * 0.5
    return values


def _pattern_measure(image: np.ndarray) -> tuple[str, float] | None:
    """Estimate a Bayer arrangement and its normalized confidence."""
    height, width = image.shape[:2]
    height -= height % 2
    width -= width % 2
    if min(height, width) < 8:
        return None
    red, green, blue = (
        _intermediate_values(image[:height, :width, index])
        for index in range(3)
    )
    blocks = float((height // 2) * (width // 2))
    diagonal = (
        (green[1::2, 0::2].sum() + green[0::2, 1::2].sum())
        - (green[0::2, 0::2].sum() + green[1::2, 1::2].sum())
    ) / (2.0 * blocks * 255.0)
    first = (
        (red[0::2, 0::2].sum() + blue[1::2, 1::2].sum())
        - (red[1::2, 1::2].sum() + blue[0::2, 0::2].sum())
    ) / (2.0 * blocks * 255.0)
    second = (
        (red[1::2, 0::2].sum() + blue[0::2, 1::2].sum())
        - (red[0::2, 1::2].sum() + blue[1::2, 0::2].sum())
    ) / (2.0 * blocks * 255.0)
    if abs(diagonal) < 1e-6:
        return None
    diagonal_name = "dotgg" if diagonal < 0 else "gdotg"
    color_delta = first if diagonal_name == "dotgg" else second
    if abs(color_delta) < 1e-6:
        return None
    colors = ("rggb", "bggr") if diagonal_name == "dotgg" else ("grbg", "gbrg")
    pattern = colors[0] if color_delta < 0 else colors[1]
    return pattern, float(max(abs(diagonal), abs(color_delta)))


def _pattern_phase(pattern: str) -> int:
    return {"rggb": 0, "bggr": 1, "grbg": 2, "gbrg": 3}[pattern]
