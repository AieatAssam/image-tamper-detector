"""Optional AEROBLADE-style detector using external MIT TAESD ONNX models.

The detector is an independent implementation of the AEROBLADE paper's
reconstruction-error idea.  It uses the MIT-licensed, distilled TAESD ONNX
encoder/decoder pair from ``julienkay/taesd`` (revision
``c8a437fd0201c21c3bcd298fcf3181b063bcc1eb``); the weights are not bundled.
The paper uses LPIPS.  This implementation deliberately uses mean L1 because
the runtime extra has no perceptual backbone, and therefore must not be read as
paper-level AEROBLADE performance.
"""

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext, to_probability


DEFAULT_ENCODER_PATH = Path("models/taesd/encoder.onnx")
DEFAULT_DECODER_PATH = Path("models/taesd/decoder.onnx")
DEFAULT_THRESHOLD = 0.05
DEFAULT_SCALE = 0.02
MAX_ANALYSIS_SIDE = 512


class AerobladeDetector:
    id = "aeroblade"
    name = "AEROBLADE Reconstruction Error"
    family = "learned"
    applicable_formats = frozenset({"JPEG", "PNG", "WEBP", "TIFF"})
    produces_map = False
    description = "Measures TAESD autoencoder reconstruction error as a latent-diffusion cue."
    limitations = [
        "Uses distilled TAESD, an approximation of the Stable Diffusion autoencoder used by AEROBLADE.",
        "Uses mean L1 reconstruction error instead of the paper's stronger LPIPS distance.",
        "Latent-diffusion-only; useless against splicing, copy-move, or GAN output.",
        "Requires externally supplied ONNX encoder and decoder weights.",
    ]
    threshold = DEFAULT_THRESHOLD
    scale = DEFAULT_SCALE
    higher_is_worse = False

    def __init__(
        self,
        encoder_path: str | Path = DEFAULT_ENCODER_PATH,
        decoder_path: str | Path = DEFAULT_DECODER_PATH,
    ) -> None:
        self.encoder_path = Path(encoder_path)
        self.decoder_path = Path(decoder_path)

    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        return True, "optional TAESD reconstruction detector"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        config = _calibration_settings(self)
        if not self.encoder_path.is_file() or not self.decoder_path.is_file():
            return self._unavailable(started, "AEROBLADE TAESD ONNX models not installed", float(config["threshold"]))
        try:
            import onnxruntime as ort
        except Exception:
            return self._unavailable(started, "AEROBLADE onnxruntime extra not installed", float(config["threshold"]))

        try:
            encoder = ort.InferenceSession(str(self.encoder_path), providers=["CPUExecutionProvider"])
            decoder = ort.InferenceSession(str(self.decoder_path), providers=["CPUExecutionProvider"])
            image = _prepare_input(ctx, encoder)
            latent = _run_first_output(encoder, image)
            reconstruction = _run_first_output(decoder, latent)
            if reconstruction.shape != image.shape:
                raise ValueError(
                    f"TAESD reconstruction shape {reconstruction.shape} does not match input {image.shape}"
                )
            error = float(np.mean(np.abs(np.clip(reconstruction, 0.0, 1.0) - image)))
            score = to_probability(
                error,
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
                f"TAESD mean L1 reconstruction error {error:.6f}",
                {
                    "reconstruction_l1": error,
                    "analysis_height": float(image.shape[2]),
                    "analysis_width": float(image.shape[3]),
                },
                None,
                _duration(started),
            )
        except Exception as exc:
            return DetectorResult(
                self.id,
                DetectorState.ERROR,
                None,
                None,
                self.threshold,
                "AEROBLADE detector failed",
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


def _prepare_input(ctx: ImageContext, encoder: Any) -> np.ndarray:
    """Create RGB [0, 1] NCHW input, honoring fixed ONNX spatial dimensions."""
    from PIL import Image

    shape = encoder.get_inputs()[0].shape
    if len(shape) != 4:
        raise ValueError(f"TAESD encoder input must be rank 4, got {shape}")
    fixed_height = _fixed_dimension(shape[2])
    fixed_width = _fixed_dimension(shape[3])
    image = Image.fromarray(ctx.rgb_uint8, mode="RGB")
    if fixed_height and fixed_width:
        size = (fixed_width, fixed_height)
    else:
        height, width = image.height, image.width
        ratio = min(1.0, MAX_ANALYSIS_SIDE / max(height, width))
        size = (
            max(8, round(width * ratio / 8) * 8),
            max(8, round(height * ratio / 8) * 8),
        )
    image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0


def _fixed_dimension(value: Any) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _run_first_output(session: Any, value: np.ndarray) -> np.ndarray:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs or not outputs:
        raise ValueError("TAESD ONNX model has no input or output")
    return np.asarray(session.run([outputs[0].name], {inputs[0].name: value})[0], dtype=np.float32)


def _calibration_settings(detector: AerobladeDetector) -> dict[str, float | bool]:
    try:
        from backend.app.analysis.adapters import _settings

        return _settings(detector.id)
    except KeyError:
        # The calibration entry is added with the opt-in registry integration.
        return {
            "threshold": detector.threshold,
            "scale": detector.scale,
            "higher_is_worse": detector.higher_is_worse,
        }


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
