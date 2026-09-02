from io import BytesIO
import struct

import numpy as np
import pytest
from PIL import Image

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.exif import ExifConsistencyDetector, image_was_resized


def _jpeg(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="JPEG", quality=95)
    return output.getvalue()


def _with_thumbnail(main: Image.Image, thumbnail: Image.Image) -> bytes:
    base = _jpeg(main)
    payload = _thumbnail_exif(thumbnail)
    segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return base[:2] + segment + base[2:]


def _thumbnail_exif(thumbnail: Image.Image) -> bytes:
    thumb = _jpeg(thumbnail.resize((32, 32)))
    # Minimal little-endian EXIF TIFF: IFD0, then IFD1 with JPEG offset/length.
    ifd1_offset = 14
    thumb_offset = ifd1_offset + 2 + 24 + 4
    tiff = bytearray(b"II*\x00" + struct.pack("<I", 8))
    tiff += struct.pack("<H", 0) + struct.pack("<I", ifd1_offset)
    tiff += struct.pack("<H", 2)
    tiff += struct.pack("<HHI", 0x0201, 4, 1) + struct.pack("<I", thumb_offset)
    tiff += struct.pack("<HHI", 0x0202, 4, 1) + struct.pack("<I", len(thumb))
    tiff += struct.pack("<I", 0) + thumb
    return b"Exif\x00\x00" + bytes(tiff)


def _tiff_with_thumbnail(main: Image.Image, thumbnail: Image.Image) -> bytes:
    output = BytesIO()
    main.save(output, format="TIFF")
    raw = bytearray(output.getvalue())
    first_ifd = struct.unpack_from("<I", raw, 4)[0]
    entry_count = struct.unpack_from("<H", raw, first_ifd)[0]
    next_ifd = first_ifd + 2 + entry_count * 12
    ifd1_offset = len(raw)
    raw[next_ifd : next_ifd + 4] = struct.pack("<I", ifd1_offset)
    thumb = _jpeg(thumbnail.resize((32, 32)))
    thumb_offset = ifd1_offset + 2 + 24 + 4
    raw += struct.pack("<H", 2)
    raw += struct.pack("<HHI", 0x0201, 4, 1) + struct.pack("<I", thumb_offset)
    raw += struct.pack("<HHI", 0x0202, 4, 1) + struct.pack("<I", len(thumb))
    raw += struct.pack("<I", 0) + thumb
    return bytes(raw)


def _with_container_thumbnail(main: Image.Image, thumbnail: Image.Image, fmt: str) -> bytes:
    output = BytesIO()
    if fmt == "TIFF":
        return _tiff_with_thumbnail(main, thumbnail)
    main.save(output, format=fmt, exif=_thumbnail_exif(thumbnail))
    return output.getvalue()


def test_matching_and_mismatching_embedded_thumbnails():
    main = Image.fromarray(np.full((128, 128, 3), 120, dtype=np.uint8), "RGB")
    matching = ExifConsistencyDetector().run(ImageContext(_with_thumbnail(main, main)))
    edited = main.copy()
    edited.paste((240, 240, 240), (32, 32, 128, 128))
    mismatch = ExifConsistencyDetector().run(ImageContext(_with_thumbnail(edited, main)))
    assert matching.state is DetectorState.APPLICABLE
    assert matching.metrics["thumbnail_similarity"] > 0.95
    assert mismatch.flagged is True
    assert mismatch.metrics["thumbnail_difference"] > 0.2
    assert mismatch.visualization is not None


def test_no_exif_is_not_applicable():
    result = ExifConsistencyDetector().run(ImageContext(_jpeg(Image.new("RGB", (128, 128)))))
    assert result.state is DetectorState.NOT_APPLICABLE
    assert result.score is None and result.flagged is None


def test_editor_tag_is_literal_evidence():
    exif = Image.Exif()
    exif[0x0131] = "Adobe Photoshop"
    output = BytesIO()
    Image.new("RGB", (128, 128), "white").save(output, format="JPEG", exif=exif.tobytes())
    result = ExifConsistencyDetector().run(ImageContext(output.getvalue()))
    assert result.flagged is True
    assert "Adobe Photoshop" in result.reason


def test_non_editor_software_tag_is_not_strong_editor_evidence():
    exif = Image.Exif()
    exif[0x0131] = "NIKON CORPORATION"
    output = BytesIO()
    Image.new("RGB", (128, 128), "white").save(output, format="JPEG", exif=exif.tobytes())
    result = ExifConsistencyDetector().run(ImageContext(output.getvalue()))
    assert result.metrics["software_tag"] == 1.0
    assert result.metrics.get("editor_software", 0.0) == 0.0
    assert result.metrics["raw_score"] == 0.0


@pytest.mark.parametrize("fmt", ["PNG", "WEBP", "TIFF"])
def test_non_jpeg_exif_thumbnails_are_read(fmt: str):
    main = Image.new("RGB", (128, 128), "white")
    raw = _with_container_thumbnail(main, main, fmt)
    result = ExifConsistencyDetector().run(ImageContext(raw))
    assert result.state is DetectorState.APPLICABLE
    assert result.metrics["thumbnail_similarity"] > 0.95


def test_dimension_disagreement_gates_cfa():
    exif = Image.Exif()
    exif[0xA002] = 256
    exif[0xA003] = 256
    output = BytesIO()
    Image.new("RGB", (128, 128), "white").save(output, format="JPEG", exif=exif.tobytes())
    ctx = ImageContext(output.getvalue())
    assert image_was_resized(ctx)
    result = ExifConsistencyDetector().run(ctx)
    assert result.metrics["resized"] == 1.0
    assert "dimensions" in result.reason
