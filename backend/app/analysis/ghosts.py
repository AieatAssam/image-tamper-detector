"""JPEG ghost analysis using a bounded quality sweep."""

from time import perf_counter

import cv2
import numpy as np
from scipy.stats import ks_2samp

from backend.app.analysis.adapters import _settings
from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext, to_probability


_QUALITIES = tuple(range(30, 91))
_BLOCK_SIZE = 16
_MIN_VARIANCE = 2.5


def _analysis_image(ctx: ImageContext) -> np.ndarray:
    image = ctx.rgb_uint8
    longest = max(image.shape[:2])
    if longest <= 1024:
        return image
    ratio = 1024 / longest
    return cv2.resize(image, (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))), interpolation=cv2.INTER_AREA)


def _coherent_modes(q0: np.ndarray) -> tuple[int, float]:
    values, counts = np.unique(q0[np.isfinite(q0)], return_counts=True)
    if not len(values):
        return 0, 0.0
    order = np.argsort(counts)[::-1]
    minimum_area = max(4, int(np.ceil(q0.size * 0.01)))
    modes: list[tuple[int, int]] = []
    for index in order:
        quality = int(values[index])
        if counts[index] < minimum_area or any(abs(quality - other) <= 4 for other, _ in modes):
            continue
        mask = (np.abs(q0 - quality) <= 4).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        components, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if components > 1 else 0
        if largest >= minimum_area:
            modes.append((quality, largest))
    coherence = max((area / q0.size for _, area in modes), default=0.0)
    return len(modes), float(coherence)


def _block_average(values: np.ndarray, offset_y: int, offset_x: int) -> np.ndarray:
    height = (values.shape[0] - offset_y) // _BLOCK_SIZE * _BLOCK_SIZE
    width = (values.shape[1] - offset_x) // _BLOCK_SIZE * _BLOCK_SIZE
    cropped = values[offset_y:offset_y + height, offset_x:offset_x + width]
    return cropped.reshape(height // _BLOCK_SIZE, _BLOCK_SIZE, width // _BLOCK_SIZE, _BLOCK_SIZE).mean(axis=(1, 3))


def _ks_maximum(curve: np.ndarray, q0: np.ndarray) -> float:
    valid = np.isfinite(q0)
    if not np.any(valid):
        return 0.0
    statistic = 0.0
    for quality in np.unique(q0[valid]):
        region = valid & (np.abs(q0 - quality) <= 4)
        outside = valid & ~region
        if np.count_nonzero(region) < 2 or np.count_nonzero(outside) < 2:
            continue
        quality_index = _QUALITIES.index(int(quality))
        statistic = max(
            statistic,
            float(ks_2samp(curve[..., quality_index][region], curve[..., quality_index][outside]).statistic),
        )
    return statistic


class JpegGhostDetector:
    id = "jpeg_ghosts"
    name = "JPEG Ghosts"
    family = "compression"
    applicable_formats = frozenset({"JPEG"})
    produces_map = True
    description = "Sweeps JPEG quality to find spatially separated compression-history minima."
    limitations = ["Needs a reasonably large spliced region and is not useful after flattening histories."]

    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        if ctx.format not in self.applicable_formats and ctx.raw_bytes[:2] != b"\xff\xd8":
            return False, f"JPEG ghosts requires JPEG input; decoded format is {ctx.format or 'unknown'}"
        if min(ctx.width, ctx.height) < _BLOCK_SIZE * 4:
            return False, "JPEG ghosts requires at least four 16px blocks per image dimension"
        return True, "JPEG input is large enough for spatial ghost modes"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        config = _settings(self.id)
        applicable, reason = self.applicable(ctx)
        if not applicable:
            return DetectorResult(self.id, DetectorState.NOT_APPLICABLE, None, None, float(config["threshold"]), reason, {}, None, _duration(started))

        image = _analysis_image(ctx)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_float32 = image.astype(np.float32)
        curves: list[np.ndarray] = []
        for quality in _QUALITIES:
            encoded_ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not encoded_ok:
                raise ValueError(f"JPEG re-encoding failed at quality {quality}")
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            decoded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            error = np.mean((image_float32 - decoded_rgb.astype(np.float32)) ** 2, axis=2)
            curves.append(error)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        best = None
        for offset_y in range(8):
            for offset_x in range(8):
                curve = np.stack([_block_average(error, offset_y, offset_x) for error in curves], axis=-1)
                low = curve.min(axis=-1, keepdims=True)
                high = curve.max(axis=-1, keepdims=True)
                normalised = (curve - low) / np.maximum(high - low, 1e-12)
                q0 = np.asarray(_QUALITIES, dtype=np.float32)[np.argmin(normalised, axis=-1)]
                variance = _block_average(gray**2, offset_y, offset_x) - _block_average(gray, offset_y, offset_x) ** 2
                q0[variance < _MIN_VARIANCE] = np.nan
                modes, coherence = _coherent_modes(q0)
                ks_max = _ks_maximum(curve, q0)
                candidate = (modes, ks_max, coherence)
                if best is None or candidate > best[0]:
                    best = (candidate, q0, offset_y, offset_x)

        assert best is not None
        (modes, ks_max, coherence), q0, offset_y, offset_x = best
        raw = float(modes)
        score = to_probability(raw, float(config["threshold"]), float(config["scale"]), bool(config["higher_is_worse"]))
        flagged = score >= 0.5
        q0_image = np.uint8(np.clip((np.nan_to_num(q0, nan=_QUALITIES[0]) - _QUALITIES[0]) * (255 / (_QUALITIES[-1] - _QUALITIES[0])), 0, 255))
        q0_image = cv2.resize(q0_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        visualization = cv2.applyColorMap(q0_image, cv2.COLORMAP_TURBO)
        return DetectorResult(
            self.id, DetectorState.APPLICABLE, score, flagged, float(config["threshold"]),
            f"found {modes} spatially coherent q0 mode(s); K-S max {ks_max:.3f}; alignment ({offset_y},{offset_x})",
            {
                "distinct_modes": raw,
                "spatial_coherence": coherence,
                "ks_max": ks_max,
                "alignment_y": float(offset_y),
                "alignment_x": float(offset_x),
                "q0_min": float(np.nanmin(q0)) if np.any(np.isfinite(q0)) else float(_QUALITIES[0]),
                "q0_max": float(np.nanmax(q0)) if np.any(np.isfinite(q0)) else float(_QUALITIES[0]),
            },
            visualization, _duration(started),
        )


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
