from io import BytesIO

import numpy as np
from PIL import Image

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.qtable import QuantizationTableDetector


def _jpeg(quality: int) -> bytes:
    image = Image.fromarray(np.random.default_rng(7).integers(0, 256, (256, 256, 3), dtype=np.uint8))
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality)
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


def test_png_is_not_applicable():
    output = BytesIO()
    Image.new("RGB", (256, 256), "gray").save(output, format="PNG")
    result = QuantizationTableDetector().run(ImageContext(output.getvalue()))
    assert result.state is DetectorState.NOT_APPLICABLE
    assert result.score is None
    assert result.flagged is None
