import time
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


def test_ghosts_duration_cap_is_wall_clock():
    started = time.perf_counter()
    result = JpegGhostDetector().run(ImageContext.from_path("data/samples/original/landscape_original.jpg"))
    elapsed = time.perf_counter() - started
    assert result.state is DetectorState.APPLICABLE
    assert elapsed < 8.0, f"jpeg_ghosts took {elapsed:.1f}s"
