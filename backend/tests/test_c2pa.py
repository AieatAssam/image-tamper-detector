import json
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.c2pa import C2PAAnalyzer


SAMPLE = Path(__file__).parents[2] / "data/samples/original/landscape_original.jpg"


def _reader_for(store: dict) -> Mock:
    reader = Mock()
    reader.__enter__ = Mock(return_value=reader)
    reader.__exit__ = Mock(return_value=None)
    reader.json.return_value = json.dumps(store)
    return reader


def test_absent_manifest_is_not_applicable_for_path_and_bytes():
    analyzer = C2PAAnalyzer()
    path_result = analyzer.analyze_image(SAMPLE)
    bytes_result = analyzer.analyze_image(SAMPLE.read_bytes())
    for result in (path_result, bytes_result):
        assert result["state"] is DetectorState.NOT_APPLICABLE
        assert result["reason"] == "no C2PA manifest"
        assert result["score"] is None
        assert result["flagged"] is None
        assert result["issues"] == []


def test_current_reader_api_parses_valid_generative_manifest():
    store = {
        "active_manifest": "claim-1",
        "manifests": {
            "claim-1": {
                "claim_generator": "example-tool",
                "assertions": [{"label": "c2pa.actions", "data": {
                    "actions": [{"action": "c2pa.created",
                                  "digitalSourceType": "trainedAlgorithmicMedia"}]
                }}],
            }
        },
    }
    analyzer = C2PAAnalyzer()
    with patch("backend.app.analysis.c2pa.Reader", return_value=_reader_for(store)) as reader:
        result = analyzer.analyze_image(b"not-an-image")
    reader.assert_called_once()
    assert result["state"] is DetectorState.APPLICABLE
    assert result["flagged"] is True
    assert result["score"] == 1.0
    assert "identifies generative image creation" in result["reason"]


def test_failed_validation_is_high_evidence_when_manifest_exists():
    store = {
        "active_manifest": "claim-1",
        "manifests": {"claim-1": {"claim_generator": "camera"}},
        "validation_status": "failed",
        "validation_results": [{"code": "bad_signature"}],
    }
    with patch("backend.app.analysis.c2pa.Reader", return_value=_reader_for(store)):
        result = C2PAAnalyzer().analyze_image(b"image bytes")
    assert result["flagged"] is True
    assert result["score"] == 0.95
    assert result["issues"][0]["severity"] == "high"


def test_detector_run_accepts_image_context_bytes():
    output = BytesIO()
    Image.new("RGB", (64, 64), "white").save(output, format="PNG")
    result = C2PAAnalyzer().run(ImageContext(output.getvalue()))
    assert result.state is DetectorState.NOT_APPLICABLE
    assert result.score is None
