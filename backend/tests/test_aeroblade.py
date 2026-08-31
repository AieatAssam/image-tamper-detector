from contextlib import nullcontext
from pathlib import Path

from PIL import Image

from backend.app.analysis.aeroblade import AerobladeDetector
from backend.app.analysis.base import DetectorState, ImageContext


SAMPLE = Path(__file__).parents[2] / "data/samples/original/landscape_original.jpg"


def test_missing_taesd_weights_are_not_applicable(tmp_path: Path) -> None:
    detector = AerobladeDetector(tmp_path / "taesd")
    result = detector.run(ImageContext.from_path(SAMPLE))
    assert result.state is DetectorState.NOT_APPLICABLE
    assert result.score is None
    assert "weights" in result.reason.lower()


def test_small_input_is_not_applicable() -> None:
    from io import BytesIO

    output = BytesIO()
    Image.new("RGB", (16, 16), "white").save(output, format="PNG")
    result = AerobladeDetector().run(ImageContext(output.getvalue()))
    assert result.state is DetectorState.NOT_APPLICABLE
    assert "32x32" in result.reason


def test_uses_lpips_and_lower_error_is_more_suspicious(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config.json").touch()

    class FakeTensor:
        def __init__(self, value: float) -> None:
            self.value = value

        def mean(self):
            return self

        def item(self):
            return self.value

    class FakeVae:
        def encode(self, image):
            return type("Encoded", (), {"latents": image})()

        def decode(self, latent):
            return type("Decoded", (), {"sample": latent})()

    class FakeLpips:
        def __call__(self, image, reconstruction):
            return FakeTensor(image.value)

    class FakeTorch:
        def inference_mode(self):
            return nullcontext()

    detector = AerobladeDetector(tmp_path)
    monkeypatch.setattr(detector, "_load_models", lambda: (FakeTorch(), FakeVae(), FakeLpips()))
    monkeypatch.setattr(
        "backend.app.analysis.aeroblade._prepare_input",
        lambda _ctx, _torch: FakeTensor(0.2),
    )
    monkeypatch.setattr(
        "backend.app.analysis.aeroblade._calibration_settings",
        lambda _detector: {"threshold": 0.5, "scale": 0.5, "higher_is_worse": False},
    )

    result = detector.run(ImageContext.from_path(SAMPLE))
    assert result.state is DetectorState.APPLICABLE
    assert result.metrics["reconstruction_lpips"] == 0.2
    assert "LPIPS" in result.reason


def test_torch_is_lazy() -> None:
    source = (Path(__file__).parents[1] / "app/analysis/aeroblade.py").read_text()
    assert "import torch" in source
    assert source.index("import torch") > source.index("def _load_models")
