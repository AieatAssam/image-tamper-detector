"""Calibrated weighted-logit fusion for detector results."""

import json
import math
from pathlib import Path

from backend.app.analysis.base import DetectorResult, DetectorState

CALIBRATION_PATH = Path(__file__).with_name("calibration.json")


def _calibration() -> dict:
    return json.loads(CALIBRATION_PATH.read_text())


def fuse(results: list[DetectorResult]) -> dict:
    calibration = _calibration()
    applicable = [result for result in results if result.state is DetectorState.APPLICABLE and result.score is not None]
    contributions = []
    z = float(calibration.get("fusion", {}).get("intercept", 0.0))
    for result in applicable:
        config = calibration["detectors"].get(result.detector_id, {})
        weight = float(config.get("weight", 0.0))
        clipped = min(0.99, max(0.01, float(result.score)))
        signed = weight * math.log(clipped / (1.0 - clipped))
        z += signed
        contributions.append({"id": result.detector_id, "weight": weight, "signed_contribution": signed})
    score = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))
    verdict = verdict_for(score) if len(applicable) >= 3 else "inconclusive"
    return {
        "score": score,
        "verdict": verdict,
        "method": "weighted_logit",
        "contributions": contributions,
        "calibration_version": calibration.get("version", "unknown"),
        "applicable_count": len(applicable),
    }


def verdict_for(score: float) -> str:
    if score < 0.15:
        return "authentic"
    if score < 0.35:
        return "likely_authentic"
    if score < 0.55:
        return "inconclusive"
    if score < 0.80:
        return "likely_manipulated"
    return "manipulated"
