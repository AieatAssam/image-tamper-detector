"""Explicit detector registry and failure-isolated execution."""

from concurrent.futures import ThreadPoolExecutor
import logging
from time import perf_counter

from backend.app.analysis.adapters import ELAAdapter, EntropyAdapter, NoiseResidualAdapter, _settings
from backend.app.analysis.base import Detector, DetectorResult, DetectorState, ImageContext
from backend.app.config import settings
from backend.app.analysis.copy_move import CopyMoveDetector
from backend.app.analysis.double_jpeg import DoubleJpegDetector
from backend.app.analysis.ghosts import JpegGhostDetector
from backend.app.analysis.qtable import QuantizationTableDetector
from backend.app.analysis.c2pa import C2PAAnalyzer
from backend.app.analysis.learned import LearnedDetector
from backend.app.analysis.cfa import CfaDetector
from backend.app.analysis.exif import ExifConsistencyDetector
from backend.app.analysis.spectral import SpectralPeakDetector
from backend.app.analysis.zero import ZeroDetector
from backend.app.analysis.splicebuster import SpliceBusterDetector
from backend.app.analysis.resampling import ResamplingDetector
from backend.app.analysis.aeroblade import AerobladeDetector
from backend.app.analysis.clip_probe import ClipProbeDetector
from backend.app.analysis.npr import NprDetector

logger = logging.getLogger(__name__)
_REGISTRY: dict[str, Detector] = {}
DEFAULT_ENABLED = frozenset({"ela", "prnu", "entropy", "qtable", "double_jpeg", "jpeg_ghosts", "copy_move", "cfa", "spectral", "exif", "c2pa", "zero", "splicebuster", "resampling"})


def register(detector: Detector) -> Detector:
    if detector.id in _REGISTRY:
        raise ValueError(f"detector already registered: {detector.id}")
    _REGISTRY[detector.id] = detector
    return detector


def get_all() -> dict[str, Detector]:
    return dict(_REGISTRY)


def get(ids: list[str] | None = None) -> dict[str, Detector]:
    selected = sorted(DEFAULT_ENABLED) if ids is None else ids
    unknown = sorted(set(selected) - _REGISTRY.keys())
    if unknown:
        raise KeyError(f"unknown detector id(s): {', '.join(unknown)}")
    return {detector_id: _REGISTRY[detector_id] for detector_id in selected}


def run_all(ctx: ImageContext, ids: list[str] | None = None) -> list[DetectorResult]:
    detectors = get(ids).values()

    def run_one(detector: Detector) -> DetectorResult:
        started = perf_counter()
        try:
            applicable, reason = detector.applicable(ctx)
            if not applicable:
                return DetectorResult(
                    detector.id, DetectorState.NOT_APPLICABLE, None, None,
                    _threshold(detector.id), reason, {}, None, _duration(started),
                )
            result = detector.run(ctx)
            return result
        except Exception:
            logger.exception("detector failed: %s", detector.id)
            return DetectorResult(
                detector.id, DetectorState.ERROR, None, None, _threshold(detector.id),
                "Detector failed while processing this image", {}, None, _duration(started),
                "detector failure",
            )

    with ThreadPoolExecutor(max_workers=min(len(detectors), settings.MAX_CONCURRENT_ANALYSES)) as executor:
        return list(executor.map(run_one, detectors))


def _threshold(detector_id: str) -> float:
    detector = _REGISTRY.get(detector_id)
    if detector is None:
        return 0.0
    from backend.app.analysis.adapters import _settings

    try:
        return float(_settings(detector_id).get("threshold", 0.0))
    except KeyError:
        return 0.0


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)


register(ELAAdapter())
register(NoiseResidualAdapter())
register(EntropyAdapter())
register(QuantizationTableDetector())
register(ZeroDetector())
register(DoubleJpegDetector())
register(JpegGhostDetector())
register(CopyMoveDetector())
register(CfaDetector())
register(SpectralPeakDetector())
register(ExifConsistencyDetector())
register(C2PAAnalyzer())
register(LearnedDetector())
register(SpliceBusterDetector(_settings("splicebuster")))
register(ResamplingDetector(_settings("resampling")))
register(AerobladeDetector())
register(ClipProbeDetector())
register(NprDetector())
