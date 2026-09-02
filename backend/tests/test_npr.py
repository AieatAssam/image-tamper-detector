import numpy as np

from backend.app.analysis.npr import MAX_ANALYSIS_SIDE, NprDetector, _analysis_image, _npr_relationships


def test_npr_measure_reports_training_free_features() -> None:
    image = np.random.default_rng(7).integers(0, 256, (32, 40, 3), dtype=np.uint8)
    statistic, visualization, metrics = NprDetector().measure(image)
    assert 0.0 <= statistic <= 1.0
    assert visualization.shape == image.shape[:2]
    assert metrics["difference_entropy"] >= 0.0
    assert metrics["npr_statistic"] == statistic


def test_npr_analysis_is_bounded():
    image = np.zeros((1600, 800, 3), dtype=np.uint8)
    assert _analysis_image(image).shape == (MAX_ANALYSIS_SIDE, 512, 3)


def test_npr_relationships_use_aligned_grids_and_include_zero_reference():
    image = np.zeros((4, 4, 3), dtype=np.float32)
    image[:2, :2] = 10.0
    image[:2, 2:] = np.array([[20.0, 30.0], [40.0, 50.0]])[..., None]
    image[2:, :2] = 30.0
    image[2:, 2:] = 40.0

    relationships = _npr_relationships(image)

    assert relationships.shape == (2, 2, 2, 2, 3)
    assert np.all(relationships[..., 0, 0, :] == 0.0)
    assert np.all(relationships[0, 1, 0, 1, :] == 10.0)


def test_npr_statistics_use_all_relationship_values(monkeypatch):
    relationships = np.zeros((2, 2, 2, 2, 3), dtype=np.float32)
    relationships[0, 0, 1, 1, :] = 1.0
    monkeypatch.setattr(
        "backend.app.analysis.npr._npr_relationships",
        lambda _image: relationships,
    )

    _, _, metrics = NprDetector().measure(np.zeros((4, 4, 3), dtype=np.uint8))

    assert metrics["near_constant_fraction"] == 0.75
