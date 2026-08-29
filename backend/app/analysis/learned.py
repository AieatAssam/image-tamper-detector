"""Optional face deepfake detector backed by a local ONNX model."""

from pathlib import Path
from time import perf_counter
import json

import numpy as np
from PIL import Image

from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext, to_probability


class LearnedDetector:
    id = "learned"
    name = "Optional Face Deepfake Model"
    family = "learned"
    applicable_formats = frozenset({"JPEG", "PNG", "WEBP", "TIFF"})
    produces_map = False
    description = "Optional ONNX classifier trained for face deepfake detection."
    limitations = ["This model is face-deepfake-specific, not a general splice or document detector."]
    threshold = 0.5
    model_path = Path("models/onnx/model_quantized.onnx")

    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        return True, "optional learned detector"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        try:
            import onnxruntime as ort
        except Exception:
            return self._unavailable(started)
        model_path = self.model_path
        if not model_path.is_file():
            return self._unavailable(started)
        try:
            from backend.app.analysis.adapters import _settings

            config = _settings(self.id)
            image = Image.fromarray(ctx.rgb_uint8).convert("RGB").resize((224, 224), resample=2)
            array = np.asarray(image, dtype=np.float32) * 0.00392156862745098
            array = ((array - np.asarray([0.5, 0.5, 0.5], dtype=np.float32)) / 0.5).transpose(2, 0, 1)[None]
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            raw = np.asarray(session.run(None, {input_name: array})[0]).reshape(-1)
            if raw.size < 2:
                raise ValueError("learned model returned fewer than two logits")
            if np.isclose(float(raw[:2].sum()), 1.0, atol=1e-5):
                score = float(raw[1])
            else:
                logits = raw[:2] - np.max(raw[:2])
                probabilities = np.exp(logits) / np.exp(logits).sum()
                score = float(probabilities[1])
            score = max(0.0, min(1.0, score))
            calibrated = to_probability(
                score,
                float(config["threshold"]),
                float(config["scale"]),
                bool(config["higher_is_worse"]),
            )
            return DetectorResult(self.id, DetectorState.APPLICABLE, calibrated, calibrated >= 0.5, float(config["threshold"]), f"face deepfake model score {score:.3f}", {"deepfake_probability": score}, None, _duration(started))
        except Exception:
            return DetectorResult(self.id, DetectorState.ERROR, None, None, self.threshold, "learned detector failed", {}, None, _duration(started), "learned detector failure")

    def _unavailable(self, started: float) -> DetectorResult:
        return DetectorResult(self.id, DetectorState.NOT_APPLICABLE, None, None, self.threshold, "learned detector not installed", {}, None, _duration(started))


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
