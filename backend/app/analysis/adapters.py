"""Adapters from the existing analyzers to the shared detector protocol."""

import json
import logging
from pathlib import Path
from time import perf_counter

from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext, to_probability
from backend.app.analysis.ela import ELAAnalyzer
from backend.app.analysis.entropy import EntropyAnalyzer
from backend.app.analysis.prnu import PRNUAnalyzer

logger = logging.getLogger(__name__)
_CALIBRATION_PATH = Path(__file__).with_name("calibration.json")
_CALIBRATION = json.loads(_CALIBRATION_PATH.read_text())


def _settings(detector_id: str) -> dict[str, float | bool]:
    return _CALIBRATION["detectors"][detector_id]


class ELAAdapter:
    id = "ela"
    name = "Error Level Analysis"
    family = "compression"
    applicable_formats = frozenset({"JPEG"})
    produces_map = True
    description = "Measures JPEG error-level discontinuities associated with local recompression."
    limitations = ["Meaningful only for JPEG files."]

    def __init__(self) -> None:
        self.analyzer = ELAAnalyzer()

    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        if ctx.format not in self.applicable_formats:
            return False, f"ELA requires JPEG input; decoded format is {ctx.format or 'unknown'}"
        return True, "JPEG input is applicable to ELA"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        config = _settings(self.id)
        try:
            is_applicable, reason = self.applicable(ctx)
            if not is_applicable:
                return DetectorResult(self.id, DetectorState.NOT_APPLICABLE, None, None, float(config["threshold"]), reason, {}, None, _duration(started))
            _, visualization, features = self.analyzer.detect_tampering(ctx.pil_image)
            raw = float(features.edge_discontinuity)
            score = to_probability(raw, float(config["threshold"]), float(config["scale"]), bool(config["higher_is_worse"]))
            flagged = score >= 0.5
            return DetectorResult(
                self.id, DetectorState.APPLICABLE, score, flagged, float(config["threshold"]),
                f"edge discontinuity {raw:.3f} {'exceeds' if flagged else 'is below'} the {float(config['threshold']):.3f} threshold",
                {"edge_discontinuity": raw, "texture_variance": float(features.texture_variance), "noise_consistency": float(features.noise_consistency), "compression_artifacts": float(features.compression_artifacts)},
                visualization, _duration(started),
            )
        except Exception:
            logger.exception("ELA detector failed")
            raise


class NoiseResidualAdapter:
    id = "prnu"
    name = "Noise Residual Inconsistency"
    family = "sensor"
    applicable_formats = frozenset({"JPEG", "PNG", "WEBP", "TIFF"})
    produces_map = True
    description = "Runs the Noisesniffer a-contrario test for locally improbable noise structure."
    limitations = ["This is not camera attribution without a reference fingerprint."]

    def __init__(self) -> None:
        self.analyzer = PRNUAnalyzer()

    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        return True, "Noise residual analysis supports this decoded image format"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        config = _settings(self.id)
        is_tampered, visualization, raw = self.analyzer.detect_tampering(ctx.downscaled_rgb_uint8)
        raw = float(raw)
        score = to_probability(raw, float(config["threshold"]), float(config["scale"]), bool(config["higher_is_worse"]))
        flagged = score >= 0.5
        return DetectorResult(
            self.id, DetectorState.APPLICABLE, score, flagged, float(config["threshold"]),
            f"Noisesniffer significance -log10(NFA) {raw:.3f} {'exceeds' if flagged else 'is below'} the {float(config['threshold']):.3f} threshold",
            {"noisesniffer_significance": raw}, visualization, _duration(started),
        )


class EntropyAdapter:
    id = "entropy"
    name = "Entropy Analysis"
    family = "spectral"
    applicable_formats = frozenset({"JPEG", "PNG", "WEBP", "TIFF"})
    produces_map = True
    description = "Compares local channel entropy patterns associated with generated imagery."
    limitations = ["Heuristic cue with limited independent validation."]

    def __init__(self) -> None:
        self.analyzer = EntropyAnalyzer(matching_threshold=float(_settings(self.id)["threshold"]))

    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        return True, "Entropy analysis supports this decoded image format"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        config = _settings(self.id)
        is_ai_generated, visualization, raw = self.analyzer.detect_ai_generated(ctx.downscaled_rgb_uint8)
        raw = float(raw)
        score = to_probability(raw, float(config["threshold"]), float(config["scale"]), bool(config["higher_is_worse"]))
        flagged = score >= 0.5
        return DetectorResult(
            self.id, DetectorState.APPLICABLE, score, flagged, float(config["threshold"]),
            f"matching entropy proportion {raw:.3f} {'is below' if flagged else 'exceeds'} the {float(config['threshold']):.3f} threshold",
            {"matching_proportion": raw}, visualization, _duration(started),
        )


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
