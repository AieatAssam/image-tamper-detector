from io import BytesIO
import struct

import numpy as np
from PIL import Image

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.exif import ExifConsistencyDetector, image_was_resized


def _jpeg(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="JPEG", quality=95)
    return output.getvalue()


def _with_thumbnail(main: Image.Image, thumbnail: Image.Image) -> bytes:
    base = _jpeg(main)
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
    payload = b"Exif\x00\x00" + bytes(tiff)
    segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return base[:2] + segment + base[2:]


def test_matching_and_mismatching_embedded_thumbnails():
    main = Image.fromarray(np.full((128, 128, 3), 120, dtype=np.uint8), "RGB")
    matching = ExifConsistencyDetector().run(ImageContext(_with_thumbnail(main, main)))
    different = Image.fromarray(np.full((128, 128, 3), 240, dtype=np.uint8), "RGB")
    mismatch = ExifConsistencyDetector().run(ImageContext(_with_thumbnail(main, different)))
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
