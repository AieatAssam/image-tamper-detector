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
        "validation_state": "Valid",
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
        "validation_state": "Invalid",
        "active_manifest": "claim-1",
        "manifests": {"claim-1": {"claim_generator": "camera"}},
        "validation_results": [{"code": "assertion.dataHash.mismatch"}],
    }
    with patch("backend.app.analysis.c2pa.Reader", return_value=_reader_for(store)):
        result = C2PAAnalyzer().analyze_image(b"image bytes")
    assert result["flagged"] is True
    assert result["score"] == 0.95
    assert result["issues"][0]["severity"] == "high"
    assert result["issues"][0]["type"] == "post_signing_mismatch"


def test_generic_validation_failure_is_not_called_signature_failure():
    store = {
        "validation_state": "Invalid",
        "active_manifest": "claim-1",
        "manifests": {"claim-1": {"claim_generator": "camera"}},
        "validation_results": [{"code": "assertion.missingclaim"}],
    }
    with patch("backend.app.analysis.c2pa.Reader", return_value=_reader_for(store)):
        result = C2PAAnalyzer().analyze_image(b"image bytes")
    assert result["issues"][0]["type"] == "validation_failed"
    assert result["issues"][0]["type"] != "signature_invalid"


def test_missing_validation_state_is_not_positive_or_clean_evidence():
    store = {
        "active_manifest": "claim-1",
        "manifests": {"claim-1": {"claim_generator": "camera"}},
    }
    with patch("backend.app.analysis.c2pa.Reader", return_value=_reader_for(store)):
        result = C2PAAnalyzer().analyze_image(b"image bytes")
    assert result["state"] is DetectorState.APPLICABLE
    assert result["score"] is None
    assert result["flagged"] is None
    assert result["issues"][0]["type"] == "validation_unknown"


def test_trusted_state_is_preserved_separately_from_valid():
    store = {
        "validation_state": "Trusted",
        "active_manifest": "claim-1",
        "manifests": {"claim-1": {"claim_generator": "camera"}},
    }
    with patch("backend.app.analysis.c2pa.Reader", return_value=_reader_for(store)):
        result = C2PAAnalyzer().analyze_image(b"image bytes")
    assert result["metadata"]["validation_state"] == "trusted"
    assert result["metadata"]["trusted"] is True
    assert result["metrics"]["trusted"] == 1.0


def test_ingredient_generation_is_not_active_manifest_generation():
    store = {
        "validation_state": "Valid",
        "active_manifest": "claim-1",
        "manifests": {
            "claim-1": {
                "claim_generator": "camera",
                "assertions": [],
                "ingredients": [{
                    "manifest_data": {
                        "assertions": [{"data": {"actions": [{
                            "action": "c2pa.created",
                            "digitalSourceType": "trainedAlgorithmicMedia",
                        }]}}]
                    }
                }],
            }
        },
    }
    with patch("backend.app.analysis.c2pa.Reader", return_value=_reader_for(store)):
        result = C2PAAnalyzer().analyze_image(b"image bytes")
    assert result["flagged"] is False
    assert result["metrics"]["generative_assertion"] == 0.0


def test_valid_state_is_not_failed_by_failure_details():
    store = {
        "active_manifest": "claim-1",
        "manifests": {"claim-1": {"claim_generator": "camera"}},
        "validation_state": "Valid",
        "validation_results": {"activeManifest": {"failure": [{"code": "signingCredential.untrusted"}]}},
    }
    with patch("backend.app.analysis.c2pa.Reader", return_value=_reader_for(store)):
        result = C2PAAnalyzer().analyze_image(b"image bytes")
    assert result["flagged"] is False
    assert result["score"] == 0.05


def test_detector_run_accepts_image_context_bytes():
    output = BytesIO()
    Image.new("RGB", (64, 64), "white").save(output, format="PNG")
    result = C2PAAnalyzer().run(ImageContext(output.getvalue()))
    assert result.state is DetectorState.NOT_APPLICABLE
    assert result.score is None


def test_bytes_path_uses_detected_png_mime():
    output = BytesIO()
    Image.new("RGB", (64, 64), "white").save(output, format="PNG")
    store = {
        "validation_state": "Valid",
        "active_manifest": "claim-1",
        "manifests": {"claim-1": {"claim_generator": "camera"}},
    }
    reader = _reader_for(store)
    with patch("backend.app.analysis.c2pa.Reader", return_value=reader) as mocked_reader:
        C2PAAnalyzer().analyze_image(output.getvalue())
    assert mocked_reader.call_args.args[0] == "image/png"
