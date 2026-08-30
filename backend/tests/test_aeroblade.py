from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from backend.app.analysis.aeroblade import AerobladeDetector
from backend.app.analysis.base import DetectorState, ImageContext


SAMPLE = Path(__file__).parents[2] / "data/samples/original/landscape_original.jpg"


def test_missing_taesd_models_are_not_applicable(tmp_path: Path) -> None:
    detector = AerobladeDetector(tmp_path / "encoder.onnx", tmp_path / "decoder.onnx")
    result = detector.run(ImageContext.from_path(SAMPLE))
    assert result.state is DetectorState.NOT_APPLICABLE
    assert result.score is None
    assert "not installed" in result.reason.lower()


def test_reconstruction_uses_inferred_io_and_lower_error_is_more_suspicious(
    tmp_path: Path, monkeypatch
) -> None:
    encoder_path = tmp_path / "encoder.onnx"
    decoder_path = tmp_path / "decoder.onnx"
    encoder_path.touch()
    decoder_path.touch()

    class FakeSession:
        def __init__(self, path: str, providers: list[str]) -> None:
            self.encoder = Path(path).name == "encoder.onnx"

        def get_inputs(self):
            name = "arbitrary_image_input" if self.encoder else "arbitrary_latent_input"
            shape = ["batch", 3, "height", "width"] if self.encoder else ["batch", "channels", "height", "width"]
            return [SimpleNamespace(name=name, shape=shape)]

        def get_outputs(self):
            name = "arbitrary_latent_output" if self.encoder else "arbitrary_image_output"
            return [SimpleNamespace(name=name)]

        def run(self, output_names, feeds):
            value = next(iter(feeds.values()))
            if self.encoder:
                return [np.zeros((1, 4, value.shape[2] // 8, value.shape[3] // 8), dtype=np.float32)]
            return [np.ones((1, 3, value.shape[2] * 8, value.shape[3] * 8), dtype=np.float32)]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=FakeSession))
    monkeypatch.setattr(
        "backend.app.analysis.aeroblade._calibration_settings",
        lambda detector: {"threshold": 0.5, "scale": 0.5, "higher_is_worse": False},
    )

    detector = AerobladeDetector(encoder_path, decoder_path)
    white = Image.new("RGB", (32, 24), "white")
    black = Image.new("RGB", (32, 24), "black")
    from io import BytesIO

    def context(image: Image.Image) -> ImageContext:
        output = BytesIO()
        image.save(output, format="PNG")
        return ImageContext(output.getvalue())

    white_result = detector.run(context(white))
    black_result = detector.run(context(black))
    assert white_result.state is DetectorState.APPLICABLE
    assert white_result.metrics["reconstruction_l1"] == 0.0
    assert black_result.metrics["reconstruction_l1"] == 1.0
    assert white_result.score > black_result.score


def test_no_torch_dependency_is_imported() -> None:
    source = Path(__file__).parents[1] / "app/analysis/aeroblade.py"
    assert "import torch" not in source.read_text()
