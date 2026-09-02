"""JPEG quantisation-table fingerprinting."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from backend.app.analysis.adapters import _settings
from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext, to_probability


# ITU-T T.81, Annex K, in natural 8x8 order. Pillow converts the JPEG DQT
# zig-zag payload before exposing image.quantization.
LUMINANCE_TABLE = (
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
)
CHROMINANCE_TABLE = (
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
)


def _jpeg_tables(ctx: ImageContext) -> tuple[str, dict[int, list[int]]]:
    with Image.open(BytesIO(ctx.raw_bytes)) as image:
        return (image.format or "").upper(), dict(getattr(image, "quantization", {}) or {})


def _scaled_table(base: tuple[int, ...], quality: int) -> list[int]:
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    return [max(1, min(255, (value * scale + 50) // 100)) for value in base]


def _estimated_qualities(tables: dict[int, list[int]]) -> dict[int, int]:
    qualities = {}
    for index, table in sorted(tables.items()):
        base = LUMINANCE_TABLE if index == 0 else CHROMINANCE_TABLE
        qualities[index] = min(
            (sum(abs(actual - expected) for actual, expected in zip(table, _scaled_table(base, quality))), quality)
            for quality in range(1, 101)
        )[1]
    return qualities


def jpeg_quality_proxy(ctx: ImageContext) -> float | None:
    """Return the lowest estimated libjpeg quality across the JPEG tables."""
    image_format, tables = _jpeg_tables(ctx)
    if image_format not in {"JPEG"} or not tables:
        return None
    return float(min(_estimated_qualities(tables).values()))


class QuantizationTableDetector:
    id = "qtable"
    name = "JPEG Quantization Table Fingerprint"
    family = "compression"
    applicable_formats = frozenset({"JPEG"})
    produces_map = False
    description = "Repository heuristic comparing JPEG quantization tables with standard libjpeg quality tables."
    limitations = ["Requires JPEG plus EXIF Make/Model provenance; this is not a camera/software table database and standard tables alone are not proof of a re-save."]

    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        image_format, tables = _jpeg_tables(ctx)
        if image_format not in self.applicable_formats or not tables:
            return False, f"qtable requires JPEG quantization tables; decoded format is {image_format or 'unknown'}"
        if not ctx.exif.get(0x010F) or not ctx.exif.get(0x0110):
            return False, "qtable requires EXIF Make and Model to compare encoder identity with claimed provenance"
        return True, "JPEG quantization tables are available"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        config = _settings(self.id)
        applicable, reason = self.applicable(ctx)
        if not applicable:
            return DetectorResult(
                detector_id=self.id, state=DetectorState.NOT_APPLICABLE, score=None, flagged=None,
                threshold=float(config["threshold"]), reason=reason, metrics={}, visualization=None,
                duration_ms=_duration(started),
            )

        image_format, tables = _jpeg_tables(ctx)
        estimated_qualities = _estimated_qualities(tables)
        distances: list[int] = []
        qualities: list[int] = []
        for index in sorted(tables):
            base = LUMINANCE_TABLE if index == 0 else CHROMINANCE_TABLE
            quality = estimated_qualities[index]
            distances.append(sum(abs(actual - expected) for actual, expected in zip(tables[index], _scaled_table(base, quality))))
            qualities.append(quality)

        concatenated = b"".join(bytes(tables[index]) for index in sorted(tables))
        fingerprint = sha256(concatenated).hexdigest()
        raw = float(sum(distances))
        score = to_probability(raw, float(config["threshold"]), float(config["scale"]), bool(config["higher_is_worse"]))
        flagged = score >= 0.5
        exact = raw == 0
        return DetectorResult(
            detector_id=self.id, state=DetectorState.APPLICABLE, score=score, flagged=flagged,
            threshold=float(config["threshold"]),
            reason=(
                f"libjpeg_distance {int(raw)} {'matches' if exact else 'differs from'} standard tables; "
                f"estimated_quality {min(qualities)}; table_sha256 {fingerprint}"
            ),
            metrics={
                "libjpeg_distance": raw,
                "estimated_quality": float(min(qualities)),
                "table_count": float(len(tables)),
            },
            visualization=None, duration_ms=_duration(started),
        )


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
