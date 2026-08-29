from pathlib import Path

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.registry import get


def test_missing_optional_model_degrades_cleanly(tmp_path: Path) -> None:
    detector = get(["learned"])["learned"]
    detector.model_path = tmp_path / "missing.onnx"
    result = detector.run(ImageContext.from_path(Path("data/samples/original/landscape_original.jpg")))
    assert result.state is DetectorState.NOT_APPLICABLE
    assert "not installed" in result.reason.lower()
