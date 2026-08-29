from io import BytesIO
import numpy as np
from PIL import Image

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.copy_move import CopyMoveDetector


def _jpeg(array: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(array).save(output, format="JPEG", quality=95)
    return output.getvalue()


def test_constructed_copy_move_and_negative():
    rng = np.random.default_rng(10)
    original = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    tampered = original.copy()
    tampered[300:396, 300:396] = tampered[64:160, 64:160]
    detector = CopyMoveDetector()
    positive = detector.run(ImageContext(_jpeg(tampered)))
    negative = detector.run(ImageContext(_jpeg(original)))
    assert positive.state is DetectorState.APPLICABLE
    assert positive.metrics["verified_clusters"] >= 1
    assert abs(abs(positive.metrics["translation_dx"]) - 236) <= 8
    assert negative.metrics["verified_clusters"] == 0


def test_textureless_is_not_applicable():
    result = CopyMoveDetector().run(ImageContext(_jpeg(np.full((512, 512, 3), 128, dtype=np.uint8))))
    assert result.state is DetectorState.NOT_APPLICABLE
    assert "keypoint" in result.reason
