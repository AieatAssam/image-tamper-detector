"""Optional frozen CLIP ViT-L/14 linear probe for AI-generation detection."""

from functools import lru_cache
from math import exp
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext, to_probability


DEFAULT_BACKBONE_PATH = Path("models/clip/open_clip_pytorch_model.safetensors")
DEFAULT_PROBE_PATH = Path("models/clip/linear_probe.npz")
DEFAULT_THRESHOLD = 0.5
DEFAULT_SCALE = 0.25
MODEL_NAME = "ViT-L-14"
MODEL_REPO = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"


class OptionalDependencyUnavailable(RuntimeError):
    pass


class ClipProbeDetector:
    id = "clip_probe"
    name = "Frozen CLIP Linear Probe"
    family = "learned"
    applicable_formats = frozenset({"JPEG", "PNG", "WEBP", "TIFF"})
    produces_map = False
    description = "Scores a frozen CLIP ViT-L/14 feature with an optional linear AI-generation probe."
    limitations = [
        "The CLIP backbone is frozen; only the external linear probe is fitted.",
        "The probe is a corpus-calibrated AI-generation screen, not an image-origin oracle.",
        "It requires externally fetched backbone and probe weights.",
    ]
    threshold = DEFAULT_THRESHOLD
    scale = DEFAULT_SCALE
    higher_is_worse = True

    def __init__(
        self,
        probe_path: str | Path = DEFAULT_PROBE_PATH,
        backbone_path: str | Path = DEFAULT_BACKBONE_PATH,
    ) -> None:
        self.probe_path = Path(probe_path)
        self.backbone_path = Path(backbone_path)

    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        if ctx.format not in self.applicable_formats:
            return False, f"CLIP probe does not support decoded format {ctx.format or 'unknown'}"
        if min(ctx.width, ctx.height) < 32:
            return False, "CLIP probe requires at least 32x32 pixels"
        if not self.probe_path.is_file():
            return False, "CLIP linear probe weights are not installed"
        if not self.backbone_path.is_file():
            return False, "CLIP ViT-L/14 backbone weights are not installed"
        return True, "frozen CLIP ViT-L/14 linear AI-generation probe"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        config = _calibration_settings(self)
        applicable, reason = self.applicable(ctx)
        if not applicable:
            return self._unavailable(started, reason, float(config["threshold"]))
        try:
            torch, model, preprocess = _load_backbone(str(self.backbone_path))
            weight, bias = _load_probe(self.probe_path)
        except OptionalDependencyUnavailable as exc:
            return self._unavailable(started, str(exc), float(config["threshold"]))
        except (ImportError, OSError, ValueError) as exc:
            return self._unavailable(started, f"CLIP optional extra unavailable: {exc}", float(config["threshold"]))

        try:
            feature = _encode_image(ctx, torch, model, preprocess)
            if feature.shape != weight.shape:
                raise ValueError(f"CLIP probe feature shape {feature.shape} does not match {weight.shape}")
            probability = _sigmoid(float(np.dot(feature, weight) + bias))
            score = to_probability(
                probability,
                float(config["threshold"]),
                float(config["scale"]),
                bool(config["higher_is_worse"]),
            )
            return DetectorResult(
                self.id,
                DetectorState.APPLICABLE,
                score,
                score >= 0.5,
                float(config["threshold"]),
                f"frozen CLIP ViT-L/14 probe probability {probability:.3f}",
                {"clip_probability": probability},
                None,
                _duration(started),
            )
        except Exception as exc:
            return DetectorResult(
                self.id,
                DetectorState.ERROR,
                None,
                None,
                float(config["threshold"]),
                "CLIP probe failed",
                {},
                None,
                _duration(started),
                str(exc),
            )

    def _unavailable(self, started: float, reason: str, threshold: float) -> DetectorResult:
        return DetectorResult(
            self.id,
            DetectorState.NOT_APPLICABLE,
            None,
            None,
            threshold,
            reason,
            {},
            None,
            _duration(started),
        )


@lru_cache(maxsize=2)
def _load_backbone(backbone_path: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        import open_clip
    except Exception as exc:
        raise OptionalDependencyUnavailable(f"CLIP optional extra is not installed: {exc}") from exc
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=backbone_path,
            device="cpu",
        )
    except Exception as exc:
        raise OptionalDependencyUnavailable(f"CLIP backbone could not be loaded: {exc}") from exc
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return torch, model, preprocess


def _load_probe(path: Path) -> tuple[np.ndarray, float]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            weight = np.asarray(archive["weight"], dtype=np.float32).reshape(-1)
            bias = float(np.asarray(archive["bias"], dtype=np.float32).reshape(-1)[0])
    except (KeyError, IndexError, OSError, ValueError) as exc:
        raise ValueError(f"invalid CLIP probe weights: {exc}") from exc
    if weight.size == 0:
        raise ValueError("invalid CLIP probe weights: empty weight")
    return weight, bias


def _encode_image(ctx: ImageContext, torch: Any, model: Any, preprocess: Any) -> np.ndarray:
    tensor = preprocess(ctx.pil_image).unsqueeze(0)
    with torch.inference_mode():
        feature = model.encode_image(tensor)
    return feature.detach().float().cpu().numpy().reshape(-1).astype(np.float32)


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + exp(-value))


def _calibration_settings(detector: ClipProbeDetector) -> dict[str, float | bool]:
    try:
        from backend.app.analysis.adapters import _settings

        return _settings(detector.id)
    except KeyError:
        return {
            "threshold": detector.threshold,
            "scale": detector.scale,
            "higher_is_worse": detector.higher_is_worse,
        }


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
