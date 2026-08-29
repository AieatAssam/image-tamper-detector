from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.cfa import CfaDetector


def _context(rgb: np.ndarray, exif=None) -> ImageContext:
    output = BytesIO()
    image = Image.fromarray(rgb.astype(np.uint8), "RGB")
    image.save(output, format="PNG", exif=exif or b"")
    return ImageContext(output.getvalue())


def test_demosaiced_image_has_lower_cfa_ratio_than_rendered_image():
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    mosaic = np.zeros(rgb.shape[:2], dtype=np.uint8)
    mosaic[0::2, 0::2] = rgb[0::2, 0::2, 2]
    mosaic[0::2, 1::2] = rgb[0::2, 1::2, 1]
    mosaic[1::2, 0::2] = rgb[1::2, 0::2, 1]
    mosaic[1::2, 1::2] = rgb[1::2, 1::2, 0]
    demosaiced = cv2.cvtColor(mosaic, cv2.COLOR_BayerBG2RGB)

    detector = CfaDetector()
    rendered_ratio, _, rendered_map = detector.measure(rgb)
    camera_ratio, _, camera_map = detector.measure(demosaiced)

    assert camera_ratio < 0.8
    assert rendered_ratio > camera_ratio
    assert rendered_map.shape == camera_map.shape == rgb.shape[:2]
    assert rendered_map.dtype == np.float32


def test_cfa_is_not_applicable_when_exif_dimensions_disagree():
    exif = Image.Exif()
    exif[0xA002] = 512
    exif[0xA003] = 512
    ctx = _context(np.zeros((256, 256, 3), dtype=np.uint8), exif.tobytes())
    result = CfaDetector().run(ctx)
    assert result.state is DetectorState.NOT_APPLICABLE
    assert result.score is None and result.flagged is None


def test_small_image_without_exif_is_not_applicable():
    result = CfaDetector().run(_context(np.zeros((64, 64, 3), dtype=np.uint8)))
    assert result.state is DetectorState.NOT_APPLICABLE
