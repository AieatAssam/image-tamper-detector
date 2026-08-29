"""Double-JPEG indicators based on DCT coefficient distributions."""

from time import perf_counter

import cv2
import numpy as np

from backend.app.analysis.adapters import _settings
from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext, to_probability


_ZIGZAG = tuple(
    (row, col)
    for total in range(15)
    for row, col in (
        ((total - col, col) for col in range(total + 1))
        if total % 2 == 0
        else ((col, total - col) for col in range(total + 1))
    )
    if row < 8 and col < 8
)
_AC_POSITIONS = _ZIGZAG[1:21]


def _coefficients(ctx: ImageContext) -> tuple[np.ndarray, tuple[int, int]]:
    y_plane = cv2.cvtColor(ctx.rgb_uint8, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32) - 128.0
    height, width = y_plane.shape
    height -= height % 8
    width -= width % 8
    y_plane = y_plane[:height, :width]
    blocks = y_plane.reshape(height // 8, 8, width // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    dct_blocks = np.stack([cv2.dct(block) for block in blocks])
    coefficients = np.column_stack([dct_blocks[:, row, col] for row, col in _AC_POSITIONS])
    return coefficients, (height // 8, width // 8)


def _leading_digits(values: np.ndarray) -> np.ndarray:
    values = np.abs(values)
    values = values[np.isfinite(values) & (values >= 1)]
    if values.size == 0:
        return np.empty(0, dtype=np.int8)
    return np.floor(values / np.power(10.0, np.floor(np.log10(values)))).astype(np.int8).clip(1, 9)


def _benford_divergence(values: np.ndarray) -> float:
    digits = _leading_digits(values)
    if digits.size == 0:
        return 0.0
    observed = np.bincount(digits, minlength=10)[1:10].astype(np.float64)
    observed /= observed.sum()
    best_error = float("inf")
    # Small deterministic grid fit: enough for this diagnostic and avoids a new optimizer dependency.
    for s in np.linspace(0.0, 2.0, 9):
        for q in np.linspace(0.5, 2.0, 7):
            model = np.log10(1.0 + 1.0 / (s + np.arange(1, 10, dtype=float) ** q))
            n = float(np.dot(observed, model) / max(np.dot(model, model), 1e-12))
            fitted = n * model
            fitted /= max(fitted.sum(), 1e-12)
            error = float(np.sum((observed - fitted) ** 2 / np.maximum(fitted, 1e-9)))
            best_error = min(best_error, error)
    return 0.0 if not np.isfinite(best_error) else best_error


def _periodicity_ratio(values: np.ndarray) -> float:
    histogram, _ = np.histogram(np.rint(values).clip(-50, 50), bins=np.arange(-50.5, 51.5, 1))
    spectrum = np.abs(np.fft.rfft(histogram.astype(float)))[1:]
    mean = float(spectrum.mean()) if spectrum.size else 0.0
    return 0.0 if mean <= 1e-12 else float(spectrum.max() / mean)


def _block_visualization(coefficients: np.ndarray, block_shape: tuple[int, int], image_shape: tuple[int, int]) -> np.ndarray:
    energy = np.mean(np.abs(coefficients), axis=1).reshape(block_shape)
    if float(energy.max()) <= 0:
        block_map = np.zeros(block_shape, dtype=np.uint8)
    else:
        block_map = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    block_map = cv2.resize(block_map, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(block_map, cv2.COLORMAP_TURBO)


class DoubleJpegDetector:
    id = "double_jpeg"
    name = "Double-JPEG / Benford Analysis"
    family = "compression"
    applicable_formats = frozenset({"JPEG"})
    produces_map = True
    description = "Measures Benford divergence and periodic DCT histogram structure from recompression."
    limitations = ["Needs at least a 256px image and is weak for same-quality aligned recompression."]

    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        if ctx.format not in self.applicable_formats and ctx.raw_bytes[:2] != b"\xff\xd8":
            return False, f"double-JPEG requires JPEG input; decoded format is {ctx.format or 'unknown'}"
        if min(ctx.width, ctx.height) < 256:
            return False, "double-JPEG requires both image dimensions to be at least 256px"
        return True, "JPEG has enough 8x8 blocks for histogram analysis"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        config = _settings(self.id)
        applicable, reason = self.applicable(ctx)
        if not applicable:
            return DetectorResult(self.id, DetectorState.NOT_APPLICABLE, None, None, float(config["threshold"]), reason, {}, None, _duration(started))
        coefficients, block_shape = _coefficients(ctx)
        benford = [_benford_divergence(coefficients[:, index]) for index in range(coefficients.shape[1])]
        periodicity = [_periodicity_ratio(coefficients[:, index]) for index in range(coefficients.shape[1])]
        benford_score = float(np.mean(benford))
        periodicity_score = float(np.mean(periodicity))
        aggregate = 0.5 * benford_score + 0.5 * periodicity_score
        score = to_probability(aggregate, float(config["threshold"]), float(config["scale"]), bool(config["higher_is_worse"]))
        flagged = score >= 0.5
        visualization = _block_visualization(coefficients, block_shape, ctx.gray_uint8.shape)
        return DetectorResult(
            self.id, DetectorState.APPLICABLE, score, flagged, float(config["threshold"]),
            f"Benford divergence {benford_score:.3f}; periodicity ratio {periodicity_score:.3f}; aggregate {aggregate:.3f}",
            {
                "benford_divergence": benford_score,
                "periodicity_ratio": periodicity_score,
                "aggregate": aggregate,
                "block_count": float(coefficients.shape[0]),
                "benford_position_max": float(max(benford)),
                "periodicity_position_max": float(max(periodicity)),
            }, visualization, _duration(started),
        )


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
