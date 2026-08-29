from io import BytesIO

import numpy as np
from PIL import Image

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.ghosts import JpegGhostDetector


def test_ghosts_returns_a_bounded_map():
    image = Image.fromarray(np.random.default_rng(9).integers(0, 256, (512, 768, 3), dtype=np.uint8))
    output = BytesIO()
    image.save(output, format="JPEG", quality=85)
    result = JpegGhostDetector().run(ImageContext(output.getvalue()))
    assert result.state is DetectorState.APPLICABLE
    assert result.visualization is not None
    assert result.visualization.shape[:2] == (512, 768)
    assert result.duration_ms < 8_000
