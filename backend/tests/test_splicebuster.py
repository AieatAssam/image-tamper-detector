from io import BytesIO

import numpy as np
from PIL import Image

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.splicebuster import SpliceBusterDetector


def _encoded(image: Image.Image, format: str, quality: int = 85) -> bytes:
    output = BytesIO()
    options = {"format": format}
    if format == "JPEG":
        options["quality"] = quality
    image.save(output, **options)
    return output.getvalue()


def test_splicebuster_detects_a_different_processing_fingerprint():
    rng = np.random.default_rng(12)
    host = Image.fromarray(rng.integers(0, 256, (384, 512, 3), dtype=np.uint8))
    donor = Image.fromarray(rng.integers(0, 256, (384, 512, 3), dtype=np.uint8))
    clean = Image.open(BytesIO(_encoded(host, "JPEG", quality=95))).convert("RGB")
    donor_jpeg = Image.open(BytesIO(_encoded(donor, "JPEG", quality=20))).convert("RGB")
    forged = clean.copy()
    forged.paste(donor_jpeg.crop((144, 112, 368, 336)), (144, 112))

    detector = SpliceBusterDetector({"threshold": 0.0, "scale": 1.0})
    clean_result = detector.run(ImageContext(_encoded(clean, "PNG")))
    forged_result = detector.run(ImageContext(_encoded(forged, "PNG")))

    assert clean_result.state is DetectorState.APPLICABLE
    assert forged_result.state is DetectorState.APPLICABLE
    assert forged_result.metrics["mahalanobis_max"] > clean_result.metrics["mahalanobis_max"]
    assert forged_result.score > clean_result.score
    assert forged_result.visualization is not None
    assert forged_result.visualization.shape == (384, 512)
    assert forged_result.visualization.dtype == np.uint8
    assert all(np.isfinite(value) for value in forged_result.metrics.values())


def test_splicebuster_reports_not_applicable_for_small_images():
    image = Image.new("RGB", (255, 256), "gray")
    result = SpliceBusterDetector().run(ImageContext(_encoded(image, "PNG")))

    assert result.state is DetectorState.NOT_APPLICABLE
    assert result.score is None
    assert result.flagged is None
