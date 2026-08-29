import time
from io import BytesIO
from unittest.mock import patch

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


def test_real_textureless_copy_move_is_low_confidence():
    result = CopyMoveDetector().run(ImageContext.from_path("data/samples/tampered/landscape_copy_paste.jpg"))
    assert result.state is DetectorState.NOT_APPLICABLE
    assert result.score is None and result.flagged is None
    assert "local keypoint support" in result.reason
    assert result.metrics["keypoints"] >= 100
    assert result.metrics["surviving_matches"] == 64
    assert result.metrics["largest_candidate_region_matches"] == 2
    assert result.metrics["candidate_region_keypoints"] < 6


def test_keypoints_with_local_candidate_but_no_verified_cluster_are_a_negative():
    rng = np.random.default_rng(11)
    image = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    candidate = {(0, 0): [(0, 1), (2, 3), (4, 5)]}
    with patch("backend.app.analysis.copy_move._cluster_matches", return_value=candidate):
        result = CopyMoveDetector().run(ImageContext(_jpeg(image)))
    assert result.state is DetectorState.APPLICABLE
    assert result.score is not None and 0.0 <= result.score < 0.5
    assert result.flagged is False
    assert result.metrics["verified_clusters"] == 0
    assert result.metrics["candidate_region_keypoints"] == 6
    assert "no verified affine cluster" in result.reason


def test_copy_move_duration_cap_is_wall_clock():
    started = time.perf_counter()
    result = CopyMoveDetector().run(ImageContext.from_path("data/samples/tampered/landscape_copy_paste.jpg"))
    elapsed = time.perf_counter() - started
    assert result.state is DetectorState.NOT_APPLICABLE
    assert elapsed < 15.0, f"copy_move took {elapsed:.1f}s"
