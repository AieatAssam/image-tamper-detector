from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext, to_probability
from backend.app.analysis.fusion import calibration_metrics, fuse
from backend.app.analysis.registry import get, register
from backend.app.main import app


SAMPLE = Path(__file__).parents[2] / "data/samples/original/landscape_original.jpg"
CAMERA_SAMPLE = Path(__file__).parents[2] / "data/corpus/real/cam_002.jpg"


def _png_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (32, 32), "white").save(stream, format="PNG")
    return stream.getvalue()


def test_score_mapping_is_monotonic_and_centered() -> None:
    scores = [to_probability(value, 10.0, 2.0, True) for value in (6.0, 10.0, 14.0)]
    assert scores[0] < scores[1] < scores[2]
    assert scores[1] == 0.5


def test_registry_contains_optional_learned_detector() -> None:
    assert "learned" in get(["learned"])


def test_inconclusive_when_too_few_detectors_apply() -> None:
    result = DetectorResult("test", DetectorState.APPLICABLE, 0.9, True, 0.5, "test", {}, None, 0)
    assert fuse([result, result])["verdict"] == "inconclusive"


def test_ela_is_not_applicable_to_png() -> None:
    result = get(["ela"]) ["ela"].run(ImageContext(_png_bytes()))
    assert result.state is DetectorState.NOT_APPLICABLE


def test_raising_detector_does_not_500_request() -> None:
    class RaisingDetector:
        id = "raising-test"
        name = "Raising test"
        family = "test"
        applicable_formats = frozenset({"JPEG"})
        produces_map = False
        description = "test"
        limitations = []

        def applicable(self, ctx):
            return True, "test"

        def run(self, ctx):
            raise RuntimeError("expected")

    register(RaisingDetector())
    response = TestClient(app).post(
        "/api/v1/analyze?detectors=raising-test",
        files={"file": ("sample.jpg", SAMPLE.read_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["detectors"][0]["state"] == "error"


def test_legacy_paths_are_gone_but_point_to_v1() -> None:
    client = TestClient(app)
    for suffix in ("ela", "prnu", "entropy", "combined"):
        response = client.post(f"/analyze/{suffix}", files={"file": ("x.jpg", b"x", "image/jpeg")})
        assert response.status_code == 410
        assert "/api/v1/analyze" in response.text


def test_image_context_decodes_once() -> None:
    context = ImageContext(SAMPLE.read_bytes())
    assert context.pil_image is context.pil_image
    assert context.rgb_uint8 is context.rgb_uint8


def test_api_exposes_training_calibration_metrics() -> None:
    response = TestClient(app).post(
        "/api/v1/analyze?detectors=qtable,c2pa,learned&include_maps=false",
        files={"file": ("sample.jpg", CAMERA_SAMPLE.read_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    detectors = {item["id"]: item for item in response.json()["detectors"]}
    expected = calibration_metrics("qtable")
    assert detectors["qtable"]["metrics"]["auc"] == expected["auc"]
    assert detectors["qtable"]["metrics"]["auc_standard_error"] == expected["auc_standard_error"]
    assert detectors["c2pa"]["metrics"]["auc"] is None
    expected = calibration_metrics("learned")
    assert detectors["learned"]["metrics"]["auc"] == expected["auc"]
    assert detectors["learned"]["metrics"]["auc_standard_error"] == expected["auc_standard_error"]
