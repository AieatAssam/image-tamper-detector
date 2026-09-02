import time
from io import BytesIO

import numpy as np
from PIL import Image

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.ghosts import JpegGhostDetector, _MIN_VARIANCE, _QUALITIES


def test_ghosts_use_paper_sweep_and_variance_guard():
    assert _QUALITIES == tuple(range(30, 91))
    assert _MIN_VARIANCE == 2.5


def test_ghosts_returns_a_bounded_map():
    image = Image.fromarray(np.random.default_rng(9).integers(0, 256, (512, 768, 3), dtype=np.uint8))
    output = BytesIO()
    image.save(output, format="JPEG", quality=85)
    result = JpegGhostDetector().run(ImageContext(output.getvalue()))
    assert result.state is DetectorState.APPLICABLE
    assert result.visualization is not None
    assert result.visualization.shape[:2] == (512, 768)
    assert 0.0 <= result.metrics["ks_max"] <= 1.0
    assert 0 <= result.metrics["alignment_y"] < 8
    assert 0 <= result.metrics["alignment_x"] < 8


def test_ghosts_duration_cap_is_wall_clock():
    started = time.perf_counter()
    result = JpegGhostDetector().run(ImageContext.from_path("data/samples/original/landscape_original.jpg"))
    elapsed = time.perf_counter() - started
    assert result.state is DetectorState.APPLICABLE
    assert elapsed < 8.0, f"jpeg_ghosts took {elapsed:.1f}s"
