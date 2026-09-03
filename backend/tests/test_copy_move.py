import time
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.copy_move import (
    CopyMoveDetector,
    _cluster_matches,
    _estimate_affine,
    _generalized_matches,
    _is_plausible_affine,
)


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
    assert positive.metrics["verified_cluster_pairs"] >= 1
    assert negative.state is DetectorState.APPLICABLE
    assert negative.score is not None


def test_textureless_is_not_applicable():
    result = CopyMoveDetector().run(ImageContext(_jpeg(np.full((512, 512, 3), 128, dtype=np.uint8))))
    assert result.state is DetectorState.NOT_APPLICABLE
    assert "keypoint" in result.reason


def test_real_textureless_copy_move_is_applicable_when_global_sift_exists():
    result = CopyMoveDetector().run(ImageContext.from_path("data/samples/tampered/landscape_copy_paste.jpg"))
    assert result.state is DetectorState.APPLICABLE
    assert result.score is not None
    assert result.metrics["keypoints"] >= 100
    assert "keypoint" not in result.reason


def test_keypoints_with_local_candidate_but_no_verified_cluster_are_a_negative():
    rng = np.random.default_rng(11)
    image = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    candidate = {(1, 2): [(0, 1), (2, 3)]}
    with patch("backend.app.analysis.copy_move._cluster_matches", return_value=candidate):
        result = CopyMoveDetector().run(ImageContext(_jpeg(image)))
    assert result.state is DetectorState.APPLICABLE
    assert result.score is not None and 0.0 <= result.score < 0.5
    assert result.flagged is False
    assert result.metrics["verified_clusters"] == 0
    assert result.metrics["candidate_region_keypoints"] == 4
    assert "no verified affine cluster" in result.reason


def test_copy_move_duration_cap_is_wall_clock():
    started = time.perf_counter()
    result = CopyMoveDetector().run(ImageContext.from_path("data/samples/tampered/landscape_copy_paste.jpg"))
    elapsed = time.perf_counter() - started
    assert result.state is DetectorState.APPLICABLE
    assert elapsed < 15.0, f"copy_move took {elapsed:.1f}s"


def test_generalized_2nn_keeps_multiple_matches_before_ratio_break():
    def match(query, train, distance):
        return SimpleNamespace(queryIdx=query, trainIdx=train, distance=distance)

    rows = [
        [
            match(query, query, 0.0),
            match(query, (query + 1) % 4, 1.0),
            match(query, (query + 2) % 4, 3.0),
            match(query, (query + 3) % 4, 7.0),
        ]
        for query in range(4)
    ]
    fake_matcher = SimpleNamespace(knnMatch=lambda *_args, **_kwargs: rows)
    descriptors = np.zeros((4, 128), dtype=np.float32)
    with patch("backend.app.analysis.copy_move.cv2.BFMatcher", return_value=fake_matcher):
        matches = _generalized_matches(descriptors)
    assert len(matches) == 12
    assert all(query != train for query, train in matches)


def test_copy_move_clusters_the_union_of_matched_keypoints():
    keypoints = [cv2.KeyPoint(float(index), 0.0, 1.0) for index in range(8)]
    descriptors = np.zeros((8, 128), dtype=np.float32)
    matches = [(0, 4), (1, 5), (2, 6), (3, 7)]
    labels = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    with patch("backend.app.analysis.copy_move._generalized_matches", return_value=matches), patch(
        "backend.app.analysis.copy_move.linkage", return_value=object()
    ) as cluster, patch("backend.app.analysis.copy_move.fcluster", return_value=labels):
        candidates = _cluster_matches(keypoints, descriptors)
    assert candidates == {(1, 2): matches}
    assert cluster.call_count == 1


def test_copy_move_uses_normalized_full_affine_ransac():
    rng = np.random.default_rng(12)
    image = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    candidate = {(1, 2): [(0, 1), (2, 3), (4, 5), (6, 7)]}
    affine = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]], dtype=np.float64)
    mask = np.ones((4, 1), dtype=np.uint8)
    with patch("backend.app.analysis.copy_move._cluster_matches", return_value=candidate), patch(
        "backend.app.analysis.copy_move.cv2.estimateAffine2D", return_value=(affine, mask)
    ) as estimate:
        result = CopyMoveDetector().run(ImageContext(_jpeg(image)))
    # `flagged` is a calibrated operating point (threshold 9.0 verified clusters
    # after the R19 refit), not the paper's two-cluster decision, so this test
    # asserts the mechanism it is named for and leaves flagging to calibration.
    assert result.metrics["verified_clusters"] == 2
    assert estimate.call_args.kwargs["ransacReprojThreshold"] == 0.05
    assert estimate.call_args.kwargs["maxIters"] == 1000


def test_copy_move_accepts_the_paper_minimum_of_three_inliers():
    source = np.float32([[0, 0], [1, 0], [0, 1]])
    destination = source + np.float32([10, 20])
    affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    mask = np.ones((3, 1), dtype=np.uint8)
    with patch("backend.app.analysis.copy_move.cv2.estimateAffine2D", return_value=(affine, mask)):
        estimated = _estimate_affine(source, destination)
    assert estimated is not None
    assert estimated[0] == 3


def test_copy_move_rejects_numerically_unstable_affine_transform():
    unstable = np.array([[12.8, 0.0, 0.0], [0.0, 12.8, 0.0], [0.0, 0.0, 1.0]])
    assert _is_plausible_affine(unstable) is False
