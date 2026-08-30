from io import BytesIO

import numpy as np
from PIL import Image

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.zero import ZeroDetector


def _encoded(image: Image.Image, format: str, quality: int = 70) -> bytes:
    output = BytesIO()
    options = {"format": format}
    if format == "JPEG":
        options["quality"] = quality
    image.save(output, **options)
    return output.getvalue()


def test_zero_finds_a_foreign_grid_and_keeps_clean_jpeg_low():
    rng = np.random.default_rng(123)
    host = Image.fromarray(rng.integers(0, 256, (256, 384, 3), dtype=np.uint8))
    donor = Image.fromarray(rng.integers(0, 256, (256, 384, 3), dtype=np.uint8))
    host_jpeg = Image.open(BytesIO(_encoded(host, "JPEG"))).convert("RGB")
    donor_jpeg = Image.open(BytesIO(_encoded(donor, "JPEG"))).convert("RGB")

    detector = ZeroDetector()
    clean = detector.run(ImageContext(_encoded(host, "JPEG")))
    splice = host_jpeg.copy()
    splice.paste(donor_jpeg.crop((101, 103, 301, 203)), (100, 100))
    forged = detector.run(ImageContext(_encoded(splice, "PNG")))

    assert clean.state is DetectorState.APPLICABLE
    assert forged.state is DetectorState.APPLICABLE
    assert clean.score < 0.5 < forged.score
    assert forged.metrics["dominant_phase"] == 0
    assert forged.metrics["foreign_region_count"] >= 1
    assert forged.visualization is not None
    assert forged.visualization.shape == (256, 384)
    assert forged.visualization.dtype == np.uint8
    assert np.count_nonzero(forged.visualization) > 0
    assert all(np.isfinite(value) for value in forged.metrics.values())


def test_zero_is_not_applicable_to_tiny_images():
    image = Image.new("RGB", (31, 31), "gray")
    result = ZeroDetector().run(ImageContext(_encoded(image, "PNG")))

    assert result.state is DetectorState.NOT_APPLICABLE
    assert result.score is None
    assert result.flagged is None
