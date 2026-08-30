from io import BytesIO

import numpy as np
from PIL import Image

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.qtable import QuantizationTableDetector


def _jpeg(quality: int) -> bytes:
    image = Image.fromarray(np.random.default_rng(7).integers(0, 256, (256, 256, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[0x010F] = "Test Camera"
    exif[0x0110] = "Test Model"
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality, exif=exif.tobytes())
    return output.getvalue()


def test_pil_quality_fingerprint_and_estimate():
    detector = QuantizationTableDetector()
    fingerprints = set()
    for quality in (60, 75, 85, 95):
        result = detector.run(ImageContext(_jpeg(quality)))
        assert result.state is DetectorState.APPLICABLE
        assert result.metrics["libjpeg_distance"] == 0
        assert abs(result.metrics["estimated_quality"] - quality) <= 2
        fingerprints.add(result.reason.rsplit("table_sha256 ", 1)[1])
    assert len(fingerprints) == 4


def test_exact_libjpeg_table_is_more_suspicious_than_a_distant_table():
    image = Image.fromarray(np.random.default_rng(8).integers(0, 256, (256, 256, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[0x010F] = "Test Camera"
    exif[0x0110] = "Test Model"
    exact = BytesIO()
    image.save(exact, format="JPEG", quality=90, exif=exif.tobytes())
    distant = BytesIO()
    custom_table = [index + 1 for index in range(64)]
    image.save(distant, format="JPEG", qtables=[custom_table, custom_table], exif=exif.tobytes())

    detector = QuantizationTableDetector()
    exact_result = detector.run(ImageContext(exact.getvalue()))
    distant_result = detector.run(ImageContext(distant.getvalue()))

    assert exact_result.metrics["libjpeg_distance"] == 0
    assert distant_result.metrics["libjpeg_distance"] > exact_result.metrics["libjpeg_distance"]
    assert exact_result.score > distant_result.score


def test_missing_camera_provenance_is_not_applicable():
    output = BytesIO()
    Image.new("RGB", (256, 256), "gray").save(output, format="JPEG", quality=90)
    result = QuantizationTableDetector().run(ImageContext(output.getvalue()))
    assert result.state is DetectorState.NOT_APPLICABLE
    assert result.score is None
    assert "Make and Model" in result.reason


def test_png_is_not_applicable():
    output = BytesIO()
    Image.new("RGB", (256, 256), "gray").save(output, format="PNG")
    result = QuantizationTableDetector().run(ImageContext(output.getvalue()))
    assert result.state is DetectorState.NOT_APPLICABLE
    assert result.score is None
    assert result.flagged is None
