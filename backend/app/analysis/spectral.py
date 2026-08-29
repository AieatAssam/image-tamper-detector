"""FFT periodic-peak detector for resampling and generative upsampling artefacts."""

from time import perf_counter

import cv2
import numpy as np

from backend.app.analysis.base import (
    DetectorResult,
    DetectorState,
    ImageContext,
    to_probability,
)
from backend.app.analysis.adapters import _settings


class SpectralPeakDetector:
    id = "spectral"
    name = "Spectral Peaks"
    family = "spectral"
    applicable_formats = frozenset({"JPEG", "PNG", "WEBP", "TIFF"})
    produces_map = True
    description = "Measures anisotropic periodic peaks in a high-pass image spectrum."
    limitations = [
        "Modern learned upsamplers can leave weak or absent peaks.",
        "A spectral peak is a cue, not proof of generation by itself.",
    ]
    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        if min(ctx.width, ctx.height) < 32:
            return False, "spectral analysis requires at least 32x32 pixels"
        return True, "image is large enough for a 512x512 spectrum"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        config = _settings(self.id)
        applicable, reason = self.applicable(ctx)
        if not applicable:
            return DetectorResult(
                self.id, DetectorState.NOT_APPLICABLE, None, None,
                float(config["threshold"]), reason, {}, None, _duration(started),
            )
        peak_sigma, peak_count, visualization = self.measure(
            cv2.cvtColor(ctx.downscaled_rgb_uint8, cv2.COLOR_RGB2GRAY)
        )
        score = to_probability(peak_sigma, float(config["threshold"]), float(config["scale"]), bool(config["higher_is_worse"]))
        flagged = score >= 0.5
        return DetectorResult(
            self.id, DetectorState.APPLICABLE, score, flagged, float(config["threshold"]),
            f"flattened spectral peak is {peak_sigma:.3f} sigma ({'exceeds' if flagged else 'is below'} {float(config['threshold']):.3f})",
            {"peak_to_sigma": float(peak_sigma), "peak_count": float(peak_count)},
            visualization,
            _duration(started),
        )

    def measure(self, gray: np.ndarray) -> tuple[float, int, np.ndarray]:
        image = np.asarray(gray)
        if image.ndim == 3:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if image.ndim != 2:
            raise ValueError("spectral analysis requires a grayscale image")
        image = cv2.resize(image.astype(np.float32), (512, 512), interpolation=cv2.INTER_AREA)

        # Keep this order: changing it reintroduces edge/low-frequency artefacts.
        residual = image - cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
        hann = np.outer(np.hanning(512), np.hanning(512)).astype(np.float32)
        spectrum = np.fft.fftshift(np.fft.fft2(residual * hann))
        log_magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)
        flattened = log_magnitude - _azimuthal_average(log_magnitude)

        yy, xx = np.indices(flattened.shape)
        cy = cx = 256
        radius = np.hypot(yy - cy, xx - cx)
        valid = radius > 5
        grid = _jpeg_grid_mask(xx - cx, yy - cy, neighbourhood=3)
        valid &= ~grid
        values = flattened[valid]
        sigma = float(np.std(values))
        if sigma <= 1e-8:
            return 0.0, 0, np.zeros_like(flattened, dtype=np.uint8)
        standardized = (flattened - float(np.mean(values))) / sigma
        peak_sigma = float(np.max(standardized[valid]))

        local_max = standardized == cv2.dilate(standardized, np.ones((3, 3), np.uint8))
        peaks = valid & local_max & (standardized >= 4.0)
        peak_count = int(np.count_nonzero(peaks))
        display = np.where(valid, np.maximum(standardized, 0), 0)
        display = cv2.normalize(display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return peak_sigma, peak_count, display


def _azimuthal_average(spectrum: np.ndarray) -> np.ndarray:
    height, width = spectrum.shape
    yy, xx = np.indices((height, width))
    radius = np.sqrt((yy - height // 2) ** 2 + (xx - width // 2) ** 2).astype(np.int32)
    sums = np.bincount(radius.ravel(), weights=spectrum.ravel())
    counts = np.bincount(radius.ravel())
    averages = sums / np.maximum(counts, 1)
    return averages[radius].astype(np.float32)


def _jpeg_grid_mask(dx: np.ndarray, dy: np.ndarray, neighbourhood: int) -> np.ndarray:
    """Mask the 8x8 JPEG lattice: offsets at (64*i, 64*j), not image axes."""
    x = np.mod(dx, 64)
    y = np.mod(dy, 64)
    near_x = (x <= neighbourhood) | (x >= 64 - neighbourhood)
    near_y = (y <= neighbourhood) | (y >= 64 - neighbourhood)
    return near_x & near_y


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
