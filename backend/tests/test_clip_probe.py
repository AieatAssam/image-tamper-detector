from pathlib import Path

import numpy as np

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.clip_probe import ClipProbeDetector


SAMPLE = Path(__file__).parents[2] / "data/samples/original/landscape_original.jpg"


def test_missing_clip_weights_are_not_applicable(tmp_path: Path) -> None:
    result = ClipProbeDetector(tmp_path / "probe.npz", tmp_path / "backbone.safetensors").run(
        ImageContext.from_path(SAMPLE)
    )
    assert result.state is DetectorState.NOT_APPLICABLE
    assert result.score is None
    assert "weights" in result.reason.lower()


def test_linear_probe_scores_frozen_features(tmp_path: Path, monkeypatch) -> None:
    probe = tmp_path / "probe.npz"
    backbone = tmp_path / "backbone.safetensors"
    backbone.touch()
    np.savez(probe, weight=np.asarray([1.0, -0.5]), bias=np.asarray([0.0]))
    detector = ClipProbeDetector(probe, backbone)
    monkeypatch.setattr(detector, "_load_models", lambda: (None, None, None), raising=False)
    monkeypatch.setattr(
        "backend.app.analysis.clip_probe._load_backbone",
        lambda _path: (object(), object(), object()),
    )
    monkeypatch.setattr(
        "backend.app.analysis.clip_probe._encode_image",
        lambda _ctx, _torch, _model, _preprocess: np.asarray([2.0, 1.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        "backend.app.analysis.clip_probe._calibration_settings",
        lambda _detector: {"threshold": 0.5, "scale": 0.25, "higher_is_worse": True},
    )
    result = detector.run(ImageContext.from_path(SAMPLE))
    assert result.state is DetectorState.APPLICABLE
    assert result.metrics["clip_probability"] > 0.5


def test_torch_is_lazy() -> None:
    source = (Path(__file__).parents[1] / "app/analysis/clip_probe.py").read_text()
    assert "import torch" in source
    assert source.index("import torch") > source.index("def _load_backbone")
