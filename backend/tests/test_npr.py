import numpy as np

from backend.app.analysis.npr import NprDetector


def test_npr_measure_reports_training_free_features() -> None:
    image = np.random.default_rng(7).integers(0, 256, (32, 40, 3), dtype=np.uint8)
    statistic, visualization, metrics = NprDetector().measure(image)
    assert 0.0 <= statistic <= 1.0
    assert visualization.shape == image.shape[:2]
    assert metrics["difference_entropy"] >= 0.0
    assert metrics["npr_statistic"] == statistic
