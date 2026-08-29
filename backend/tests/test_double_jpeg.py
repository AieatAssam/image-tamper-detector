from io import BytesIO

import numpy as np
from PIL import Image

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.double_jpeg import DoubleJpegDetector


def _save(image: Image.Image, quality: int) -> bytes:
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality)
    return output.getvalue()


def test_double_jpeg_metrics_are_finite_and_directional():
    image = Image.fromarray(np.random.default_rng(8).integers(0, 256, (512, 512, 3), dtype=np.uint8))
    single = DoubleJpegDetector().run(ImageContext(_save(image, 85)))
    double = DoubleJpegDetector().run(ImageContext(_save(Image.open(BytesIO(_save(image, 85))), 75)))
    assert single.state is DetectorState.APPLICABLE
    assert double.state is DetectorState.APPLICABLE
    assert double.metrics["periodicity_ratio"] > single.metrics["periodicity_ratio"]
    assert double.metrics["benford_divergence"] > single.metrics["benford_divergence"]
    assert all(np.isfinite(value) for value in double.metrics.values())


def test_small_and_flat_images_are_safe():
    small = BytesIO()
    Image.new("RGB", (128, 128), "gray").save(small, format="JPEG")
    result = DoubleJpegDetector().run(ImageContext(small.getvalue()))
    assert result.state is DetectorState.NOT_APPLICABLE
    flat = BytesIO()
    Image.new("RGB", (256, 256), "gray").save(flat, format="JPEG")
    result = DoubleJpegDetector().run(ImageContext(flat.getvalue()))
    assert result.state is DetectorState.APPLICABLE
    assert all(np.isfinite(value) for value in result.metrics.values())
