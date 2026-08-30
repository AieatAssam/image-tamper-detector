from io import BytesIO
from unittest.mock import patch

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


def test_cfa_requires_strict_real_camera_evidence():
    exif = Image.Exif()
    exif[0x010F] = "Example Camera"
    exif[0x0110] = "Example Model"
    output = BytesIO()
    image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8), "RGB")
    image.save(output, format="JPEG", exif=exif.tobytes())
    relaxed = CfaDetector().run(ImageContext(output.getvalue()))
    assert relaxed.state is DetectorState.NOT_APPLICABLE
    assert relaxed.score is None and relaxed.flagged is None

    exif[0xA002] = 256
    exif[0xA003] = 128
    output = BytesIO()
    image.save(output, format="JPEG", exif=exif.tobytes())
    height_mismatch = CfaDetector().run(ImageContext(output.getvalue()))
    assert height_mismatch.state is DetectorState.NOT_APPLICABLE

    exif[0xA003] = 256
    output = BytesIO()
    image.save(output, format="JPEG", exif=exif.tobytes())
    strict = CfaDetector().run(ImageContext(output.getvalue()))
    assert strict.state is DetectorState.APPLICABLE
    assert strict.score is not None


def test_cfa_reads_nested_exif_pixel_dimensions():
    output = BytesIO()
    Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8), "RGB").save(
        output, format="JPEG"
    )
    ctx = ImageContext(output.getvalue())
    nested_metadata = {0x010F: "Example Camera", 0x0110: "Example Model", 0xA002: 256, 0xA003: 256}
    with patch("backend.app.analysis.cfa._metadata", return_value=nested_metadata):
        result = CfaDetector().run(ctx)
    assert result.state is DetectorState.APPLICABLE


def test_cfa_uses_full_resolution_after_strict_gate():
    image = Image.fromarray(np.zeros((256, 2048, 3), dtype=np.uint8), "RGB")
    exif = Image.Exif()
    exif[0x010F] = "Example Camera"
    exif[0x0110] = "Example Model"
    exif[0xA002] = 2048
    exif[0xA003] = 256
    output = BytesIO()
    image.save(output, format="JPEG", exif=exif.tobytes())
    detector = CfaDetector()
    ratio_map = np.ones((256, 2048), dtype=np.float32)
    with patch.object(detector, "measure", return_value=(0.5, 0, ratio_map)) as measure:
        result = detector.run(ImageContext(output.getvalue()))
    assert result.state is DetectorState.APPLICABLE
    assert measure.call_args.args[0].shape == (256, 2048, 3)
