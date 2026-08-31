from pathlib import Path
import sys

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.registry import get


def test_missing_optional_model_degrades_cleanly(tmp_path: Path, monkeypatch) -> None:
    detector = get(["learned"])["learned"]
    monkeypatch.setattr(detector, "model_path", tmp_path / "missing.onnx")
    monkeypatch.setattr(detector, "_has_face", lambda _ctx: True)
    result = detector.run(ImageContext.from_path(Path("data/samples/original/landscape_original.jpg")))
    assert result.state is DetectorState.NOT_APPLICABLE
    assert "not installed" in result.reason.lower()


def test_learned_is_not_applicable_without_a_face() -> None:
    detector = get(["learned"])["learned"]
    result = detector.run(ImageContext.from_path(Path("data/samples/original/landscape_original.jpg")))
    assert result.state is DetectorState.NOT_APPLICABLE
    assert "face" in result.reason.lower()


def test_onnx_session_is_cached(monkeypatch) -> None:
    import backend.app.analysis.learned as learned

    class FakeSession:
        pass

    class FakeRuntime:
        calls = 0

        def InferenceSession(self, _path, providers):
            assert providers == ["CPUExecutionProvider"]
            self.calls += 1
            return FakeSession()

    runtime = FakeRuntime()
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    learned._load_session.cache_clear()
    try:
        first = learned._load_session("model.onnx")
        second = learned._load_session("model.onnx")
        assert first is second
        assert runtime.calls == 1
    finally:
        learned._load_session.cache_clear()
