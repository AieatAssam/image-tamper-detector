"""EXIF consistency checks.

EXIF absence is deliberately not evidence.  This module only scores
inconsistencies or explicit processing metadata that is actually present.
"""

from datetime import datetime
from io import BytesIO
import struct
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from backend.app.analysis.base import (
    DetectorResult,
    DetectorState,
    ImageContext,
    to_probability,
)
from backend.app.analysis.adapters import _settings


class ExifConsistencyDetector:
    id = "exif"
    name = "EXIF Consistency"
    family = "metadata"
    applicable_formats = frozenset({"JPEG", "PNG", "WEBP", "TIFF"})
    produces_map = True
    description = "Checks internal EXIF consistency and embedded-thumbnail agreement."
    limitations = [
        "Metadata can be forged, removed, or rewritten by social platforms.",
        "EXIF absence is not evidence of manipulation.",
    ]
    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        if not _metadata(ctx) and not _thumbnail_bytes(ctx):
            return False, "no EXIF metadata; EXIF detector is not applicable"
        return True, "EXIF metadata is present for consistency checks"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        config = _settings(self.id)
        applicable, reason = self.applicable(ctx)
        if not applicable:
            return DetectorResult(
                self.id, DetectorState.NOT_APPLICABLE, None, None,
                float(config["threshold"]), reason, {}, None, _duration(started),
            )

        metadata = _metadata(ctx)
        evidence: list[tuple[float, str]] = []
        metrics: dict[str, float] = {"resized": float(image_was_resized(ctx))}
        visualization = None

        thumbnail = _thumbnail_bytes(ctx)
        if thumbnail:
            thumb_image = _decode_thumbnail(thumbnail)
            if thumb_image is not None:
                similarity, diff_map = _thumbnail_comparison(ctx.pil_image, thumb_image)
                metrics["thumbnail_similarity"] = similarity
                metrics["thumbnail_difference"] = 1.0 - similarity
                visualization = diff_map
                mismatch_score = min(1.0, max(0.0, (1.0 - similarity) * 3.0))
                if mismatch_score >= 0.5:
                    evidence.append((mismatch_score, f"embedded thumbnail similarity is {similarity:.3f}"))

        software = _first(metadata, 0x0131, 0x000B)
        if software:
            software_text = str(software)
            metrics["editor_software"] = 1.0
            evidence.append((0.85, f"EXIF software tag is {software_text!r}"))

        camera_tags = (0x010F, 0x0110, 0x9003, 0x829A)
        missing_camera = not any(_first(metadata, tag) is not None for tag in camera_tags)
        if missing_camera and _jpeg_like(ctx) and _plausible_dimensions(ctx.width, ctx.height):
            metrics["missing_camera_block"] = 1.0
            evidence.append((0.15, "camera EXIF block is absent from a camera-sized JPEG"))

        original = _parse_datetime(_first(metadata, 0x9003))
        digitized = _parse_datetime(_first(metadata, 0x9004))
        modified = _parse_datetime(_first(metadata, 0x0132))
        if original and ((digitized and digitized > original) or (modified and modified > original)):
            metrics["datetime_disagreement"] = 1.0
            evidence.append((0.7, "EXIF date fields disagree with DateTimeOriginal"))

        if image_was_resized(ctx):
            metrics["dimension_disagreement"] = 1.0
            evidence.append((0.9, "EXIF pixel dimensions disagree with decoded dimensions"))

        raw = max((item[0] for item in evidence), default=0.0)
        score = to_probability(raw, float(config["threshold"]), float(config["scale"]), bool(config["higher_is_worse"]))
        flagged = score >= 0.5
        if evidence:
            reason = "; ".join(item[1] for item in evidence)
        else:
            reason = "EXIF fields are internally consistent"
        return DetectorResult(
            self.id, DetectorState.APPLICABLE, score, flagged, float(config["threshold"]),
            reason, metrics, visualization, _duration(started),
        )


def image_was_resized(ctx: ImageContext) -> bool:
    """Return whether EXIF capture dimensions differ from decoded dimensions."""
    metadata = _metadata(ctx)
    width = _number(_first(metadata, 0xA002))
    height = _number(_first(metadata, 0xA003))
    return bool(width and height and (int(width) != ctx.width or int(height) != ctx.height))


def _metadata(ctx: ImageContext) -> dict[int, object]:
    metadata = dict(ctx.exif)
    try:
        metadata.update(ctx.pil_image.getexif().get_ifd(0x8769))
    except Exception:
        pass
    return metadata


def _first(metadata: dict[int, object], *tags: int) -> object | None:
    for tag in tags:
        value = metadata.get(tag)
        if value not in (None, "", b""):
            return value
    return None


def _number(value: object | None) -> int | None:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, bytes) and len(value) in (2, 4, 8):
        return int.from_bytes(value, "big")
    return None


def _thumbnail_bytes(ctx: ImageContext) -> bytes | None:
    # Pillow exposes IFD1 values on some builds, but commonly exposes only the
    # offset/length pair. Resolve that pair from the original JPEG APP1 bytes.
    try:
        ifd = ctx.pil_image.getexif().get_ifd(0x5100)
        for value in ifd.values():
            if isinstance(value, bytes) and value.startswith(b"\xff\xd8"):
                return value
    except Exception:
        pass
    return _jpeg_exif_thumbnail(ctx.raw_bytes)


def _jpeg_exif_thumbnail(raw: bytes) -> bytes | None:
    if not raw.startswith(b"\xff\xd8"):
        return None
    pos = 2
    while pos + 4 <= len(raw):
        if raw[pos] != 0xFF:
            pos += 1
            continue
        marker = raw[pos + 1]
        pos += 2
        if marker in (0xD8, 0xD9):
            continue
        if marker == 0xDA:
            break
        length = int.from_bytes(raw[pos : pos + 2], "big")
        payload = raw[pos + 2 : pos + length]
        pos += length
        if marker != 0xE1 or not payload.startswith(b"Exif\x00\x00"):
            continue
        tiff = payload[6:]
        found = _tiff_ifd1_thumbnail(tiff)
        if found:
            offset, size = found
            start = 6 + offset
            return payload[start : start + size]
    return None


def _tiff_ifd1_thumbnail(tiff: bytes) -> tuple[int, int] | None:
    if len(tiff) < 8 or tiff[:2] not in (b"II", b"MM"):
        return None
    endian = "<" if tiff[:2] == b"II" else ">"
    unpack = lambda fmt, offset: struct.unpack_from(endian + fmt, tiff, offset)[0]
    try:
        first_offset = unpack("I", 4)
        count = unpack("H", first_offset)
        next_offset_at = first_offset + 2 + count * 12
        if next_offset_at + 4 > len(tiff):
            return None
        ifd1_offset = unpack("I", next_offset_at)
        if not ifd1_offset or ifd1_offset + 2 > len(tiff):
            return None
        count = unpack("H", ifd1_offset)
        thumb_offset = thumb_size = None
        for index in range(count):
            entry = ifd1_offset + 2 + index * 12
            tag, kind, items = unpack("H", entry), unpack("H", entry + 2), unpack("I", entry + 4)
            if tag not in (0x0201, 0x0202):
                continue
            size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8}.get(kind, 0) * items
            value_offset = entry + 8 if size <= 4 else unpack("I", entry + 8)
            if value_offset + min(size, 4) > len(tiff):
                return None
            value = int.from_bytes(tiff[value_offset : value_offset + min(size, 4)], "little" if endian == "<" else "big")
            if tag == 0x0201:
                thumb_offset = value
            else:
                thumb_size = value
        if thumb_offset is not None and thumb_size is not None:
            return thumb_offset, thumb_size
    except (IndexError, struct.error, ValueError):
        return None
    return None


def _decode_thumbnail(raw: bytes) -> np.ndarray | None:
    try:
        with Image.open(BytesIO(raw)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except Exception:
        return None


def _thumbnail_comparison(image: Image.Image, thumbnail: np.ndarray) -> tuple[float, np.ndarray]:
    main = np.asarray(image.convert("RGB"), dtype=np.uint8)
    resized = cv2.resize(main, (thumbnail.shape[1], thumbnail.shape[0]), interpolation=cv2.INTER_AREA)
    difference = cv2.absdiff(resized, thumbnail)
    similarity = 1.0 - float(np.mean(difference)) / 255.0
    diff = cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY)
    diff = cv2.resize(diff, (main.shape[1], main.shape[0]), interpolation=cv2.INTER_NEAREST)
    return similarity, cv2.applyColorMap(diff, cv2.COLORMAP_JET)


def _parse_datetime(value: object | None) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def _jpeg_like(ctx: ImageContext) -> bool:
    return ctx.raw_bytes.startswith(b"\xff\xd8")


def _plausible_dimensions(width: int, height: int) -> bool:
    return min(width, height) >= 128 and width * height >= 65_536


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
