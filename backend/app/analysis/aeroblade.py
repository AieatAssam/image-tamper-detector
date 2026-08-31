"""Optional AEROBLADE-style detector using TAESD and LPIPS."""

from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext, to_probability


DEFAULT_TAESD_PATH = Path("models/taesd")
DEFAULT_LPIPS_CACHE = Path("models/lpips")
DEFAULT_LPIPS_WEIGHTS = DEFAULT_LPIPS_CACHE / "checkpoints/alexnet-owt-7be5be79.pth"
DEFAULT_THRESHOLD = 0.05
DEFAULT_SCALE = 0.02
MAX_ANALYSIS_SIDE = 512


class OptionalDependencyUnavailable(RuntimeError):
    pass


class AerobladeDetector:
    id = "aeroblade"
    name = "AEROBLADE Reconstruction Error"
    family = "learned"
    applicable_formats = frozenset({"JPEG", "PNG", "WEBP", "TIFF"})
    produces_map = False
    description = "Measures TAESD perceptual reconstruction error as a latent-diffusion cue."
    limitations = [
        "Uses distilled TAESD, an approximation of the Stable Diffusion autoencoder used by AEROBLADE.",
        "Uses LPIPS, but not the paper's exact autoencoder or training setup.",
        "Latent-diffusion-specific; useless against splicing, copy-move, or GAN output.",
        "Requires the optional torch extra and externally fetched TAESD and LPIPS weights.",
    ]
    threshold = DEFAULT_THRESHOLD
    scale = DEFAULT_SCALE
    higher_is_worse = False

    def __init__(self, taesd_path: str | Path = DEFAULT_TAESD_PATH) -> None:
        self.taesd_path = Path(taesd_path)

    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        if ctx.format not in self.applicable_formats:
            return False, f"AEROBLADE is latent-diffusion-specific and does not support decoded format {ctx.format or 'unknown'}"
        if min(ctx.width, ctx.height) < 32:
            return False, "AEROBLADE is latent-diffusion-specific and requires at least 32x32 pixels"
        return True, "latent-diffusion-specific TAESD reconstruction cue"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        config = _calibration_settings(self)
        applicable, reason = self.applicable(ctx)
        if not applicable:
            return self._unavailable(started, reason, float(config["threshold"]))
        try:
            torch, vae, perceptual = self._load_models()
        except OptionalDependencyUnavailable as exc:
            return self._unavailable(started, str(exc), float(config["threshold"]))
        except (ImportError, OSError) as exc:
            return self._unavailable(started, f"AEROBLADE optional extra unavailable: {exc}", float(config["threshold"]))

        try:
            image = _prepare_input(ctx, torch)
            with _inference_mode(torch):
                latent = vae.encode(image).latents
                reconstruction = vae.decode(latent).sample
                distance = float(perceptual(image, reconstruction).mean().item())
            score = to_probability(
                distance,
                float(config["threshold"]),
                float(config["scale"]),
                bool(config["higher_is_worse"]),
            )
            flagged = score >= 0.5
            return DetectorResult(
                self.id,
                DetectorState.APPLICABLE,
                score,
                flagged,
                float(config["threshold"]),
                f"latent-diffusion-specific TAESD LPIPS reconstruction error {distance:.6f}",
                {"reconstruction_lpips": distance},
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
                "AEROBLADE detector failed",
                {},
                None,
                _duration(started),
                str(exc),
            )

    @lru_cache(maxsize=2)
    def _load_models(self) -> tuple[Any, Any, Any]:
        if not (self.taesd_path / "config.json").is_file():
            raise OptionalDependencyUnavailable("AEROBLADE TAESD weights are not installed")
        if not DEFAULT_LPIPS_WEIGHTS.is_file():
            raise OptionalDependencyUnavailable("AEROBLADE LPIPS AlexNet weights are not installed")
        try:
            import torch
            import lpips
            from diffusers import AutoencoderTiny
        except Exception as exc:
            raise OptionalDependencyUnavailable(f"AEROBLADE optional extra is not installed: {exc}") from exc
        vae = AutoencoderTiny.from_pretrained(str(self.taesd_path), local_files_only=True).eval()
        torch.hub.set_dir(str(DEFAULT_LPIPS_CACHE))
        perceptual = lpips.LPIPS(net="alex").eval()
        return torch, vae, perceptual

    def _unavailable(self, started: float, reason: str, threshold: float) -> DetectorResult:
        return DetectorResult(
            self.id,
            DetectorState.NOT_APPLICABLE,
            None,
            None,
            threshold,
            f"AEROBLADE is latent-diffusion-specific; {reason}",
            {},
            None,
            _duration(started),
        )


def _prepare_input(ctx: ImageContext, torch: Any) -> Any:
    from PIL import Image

    image = Image.fromarray(ctx.downscaled_rgb_uint8, mode="RGB")
    ratio = min(1.0, MAX_ANALYSIS_SIDE / max(image.size))
    size = (
        max(32, round(image.width * ratio / 8) * 8),
        max(32, round(image.height * ratio / 8) * 8),
    )
    image = image.resize(size, Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 127.5 - 1.0
    return torch.from_numpy(array)


def _inference_mode(torch: Any) -> Any:
    return torch.inference_mode() if hasattr(torch, "inference_mode") else nullcontext()


def _calibration_settings(detector: AerobladeDetector) -> dict[str, float | bool]:
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
