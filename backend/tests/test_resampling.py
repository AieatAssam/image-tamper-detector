from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.resampling import ResamplingDetector, _absolute_residual


def _jpeg(array: np.ndarray, quality: int = 95) -> bytes:
    output = BytesIO()
    Image.fromarray(array).save(output, format="JPEG", quality=quality)
    return output.getvalue()


def _textured_image(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    cv2.rectangle(image, (80, 80), (430, 430), (30, 190, 100), 3)
    return image


def test_uses_kirchner_fixed_predictor_and_absolute_residual():
    gray = np.arange(25, dtype=np.uint8).reshape(5, 5)
    expected = np.abs(
        gray.astype(np.float32)
        - cv2.filter2D(
            gray.astype(np.float32), cv2.CV_32F, np.array(
                [[-.25, .5, -.25], [.5, 0, .5], [-.25, .5, -.25]], dtype=np.float32
            ), borderType=cv2.BORDER_REFLECT101
        )
    )
    np.testing.assert_allclose(_absolute_residual(gray), expected)


def test_global_resize_is_not_tampering_signal():
    original = cv2.resize(_textured_image(5), (768, 768), interpolation=cv2.INTER_CUBIC)
    resized = cv2.resize(original, (512, 512), interpolation=cv2.INTER_CUBIC)
    result = ResamplingDetector().run(ImageContext(_jpeg(resized)))

    assert result.state is DetectorState.APPLICABLE
    assert result.score is not None and result.score < 0.5
    assert result.metrics["local_inconsistency"] < 0.25
    assert result.visualization is not None
    assert result.visualization.shape == (512, 512)


def test_local_resized_region_is_detected_as_block_disagreement():
    host = _textured_image(7)
    source = host[40:200, 40:200]
    transformed = cv2.resize(source, (230, 230), interpolation=cv2.INTER_CUBIC)
    tampered = host.copy()
    tampered[160:390, 160:390] = transformed
    result = ResamplingDetector().run(ImageContext(_jpeg(tampered)))

    assert result.state is DetectorState.APPLICABLE
    assert result.score is not None and result.score > 0.5
    assert result.metrics["local_inconsistency"] > 0.115


def test_small_image_is_not_applicable():
    result = ResamplingDetector().run(ImageContext(_jpeg(np.zeros((128, 128, 3), dtype=np.uint8))))

    assert result.state is DetectorState.NOT_APPLICABLE
    assert result.score is None
    assert result.flagged is None
