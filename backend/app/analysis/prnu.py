"""Blind noise inconsistency detection adapted from IPOL Noisesniffer.

The block-selection and a-contrario NFA method is adapted from Marina Gardella,
Pablo Musé, Miguel Colom, and Jean-Michel Morel, *Image Forgery Detection
Based on Noise Inspection: Analysis and Refinement of the Noisesniffer Method*,
Image Processing On Line 14 (2024), article 462:
https://www.ipol.im/pub/art/2024/462/ .

The IPOL reference implementation (``462-main.zip``, ``Noisesniffer.py`` and
``functions.py``) is Apache-2.0 licensed. This file is a modified adaptation
under that license: it integrates the method with this repository's image
inputs and detector tuple, uses NumPy/SciPy arrays instead of files on disk,
and returns an edge-preserving visualization. It is not PRNU camera
attribution; it is a blind noise-inconsistency cue.
"""

from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image
from scipy.fft import dct
from scipy.ndimage import gaussian_filter, median_filter
from scipy.stats import binom


ImageInput = Union[str, Path, bytes, np.ndarray]
MAX_ANALYSIS_SIDE = 1024


def _as_rgb_float32(image_input: ImageInput) -> np.ndarray:
    if isinstance(image_input, (str, Path)):
        try:
            with Image.open(image_input) as image:
                image = image.convert("RGB")
                image.load()
                return np.asarray(image, dtype=np.float32).copy()
        except Exception as exc:
            raise ValueError(f"Could not load image: {image_input}") from exc
    if isinstance(image_input, bytes):
        try:
            with Image.open(BytesIO(image_input)) as image:
                image = image.convert("RGB")
                image.load()
                return np.asarray(image, dtype=np.float32).copy()
        except Exception as exc:
            raise ValueError("Failed to decode image bytes") from exc
    if not isinstance(image_input, np.ndarray):
        raise ValueError("Input must be a file path, bytes, or numpy array")
    image = image_input
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Image array must have shape HxW, HxWx1, HxWx3, or HxWx4")
    return np.asarray(image[..., :3], dtype=np.float32).copy()


def _analysis_image(image: np.ndarray) -> np.ndarray:
    longest = max(image.shape[:2])
    if longest <= MAX_ANALYSIS_SIDE:
        return image
    ratio = MAX_ANALYSIS_SIDE / float(longest)
    return cv2.resize(
        image,
        (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))),
        interpolation=cv2.INTER_AREA,
    )


def _log_binomial_tail(red_blocks: int, all_blocks: int, block_size: int, percentile: float) -> float:
    """Return log P(X >= floor(K/w²)), using the paper's block model."""
    trials = int(math.ceil(all_blocks / block_size**2))
    successes = int(math.floor(red_blocks / block_size**2))
    if successes <= 0:
        return 0.0
    if successes > trials:
        return -math.inf
    return float(binom.logsf(successes - 1, trials, percentile))


def _log_nfa(
    image_shape: tuple[int, int],
    region_size: int,
    red_blocks: int,
    all_blocks: int,
    block_size: int,
    percentile: float,
    cell_size: int,
) -> float:
    """Return log NFA from the Noisesniffer a-contrario model."""
    if region_size <= 0 or all_blocks <= 0:
        return math.inf
    height, width = image_shape
    tests = (height * width / cell_size**2) ** 2
    log_tests = (
        math.log(0.5 * block_size**2)
        + math.log(tests)
        + math.log(0.316915)
        - math.log(region_size)
        + region_size * math.log(4.062570)
    )
    return log_tests + _log_binomial_tail(red_blocks, all_blocks, block_size, percentile)


def _significance(log_nfa: float) -> float:
    """Map log NFA to a bounded, higher-is-more-suspicious raw statistic."""
    if log_nfa == math.inf:
        return -60.0
    if log_nfa == -math.inf:
        return 60.0
    return float(np.clip(-log_nfa / math.log(10.0), -60.0, 60.0))


def _block_view(channel: np.ndarray, block_size: int) -> np.ndarray:
    view = np.lib.stride_tricks.sliding_window_view(channel, (block_size, block_size))
    return view.reshape(-1, block_size, block_size)


def _channel_candidates(
    image: np.ndarray,
    channel: int,
    block_size: int,
    samples_per_bin: int,
    low_energy_percentile: float,
    std_percentile: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Select V and S block origins for one channel (Algorithms 1-10)."""
    height, width = image.shape[:2]
    channel_image = image[..., channel]
    blocks = _block_view(channel_image, block_size)
    grid_shape = (height - block_size + 1, width - block_size + 1)

    minima = image.min(axis=(0, 1))
    maxima = image.max(axis=(0, 1))
    unsaturated = np.all((image > minima) & (image < maxima), axis=2)
    valid_mask = _block_view(unsaturated.astype(np.float32), block_size).min(axis=(1, 2)) > 0.5
    valid = np.flatnonzero(valid_mask)
    if not len(valid):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), grid_shape

    means = blocks.mean(axis=(1, 2))
    stds = blocks.std(axis=(1, 2))
    valid_blocks = blocks[valid]
    transformed = dct(dct(valid_blocks, axis=1, norm="ortho"), axis=2, norm="ortho")
    frequency_mask = np.zeros((block_size, block_size), dtype=np.float32)
    threshold = {3: 3, 5: 5, 8: 9}.get(block_size)
    if threshold is None:
        raise ValueError("Noisesniffer supports block sizes 3, 5, or 8")
    for row in range(block_size):
        for col in range(block_size):
            if 0 < row + col < threshold:
                frequency_mask[row, col] = 1.0
    low_frequency_variance = np.square(transformed * frequency_mask).sum(axis=(1, 2))
    low_frequency_by_block = dict(zip(valid.tolist(), low_frequency_variance.tolist()))

    bins = max(1, int(round(len(valid) / samples_per_bin)))
    bin_size = max(1, int(len(valid) / bins))
    ordered = valid[np.argsort(means[valid], kind="stable")]
    selected_all: list[np.ndarray] = []
    selected_low_std: list[np.ndarray] = []
    for index in range(bins):
        start = index * bin_size
        end = len(ordered) if index == bins - 1 else min(len(ordered), (index + 1) * bin_size)
        in_bin = ordered[start:end]
        if not len(in_bin):
            continue
        count = max(1, int(len(in_bin) * low_energy_percentile))
        low_energy = np.array([low_frequency_by_block[int(pos)] for pos in in_bin])
        selected = in_bin[np.argsort(low_energy, kind="stable")[:count]]
        selected = selected[np.argsort(stds[selected], kind="stable")]
        if (stds[selected] == 0).sum() >= max(1, int(count * std_percentile)):
            continue
        selected_all.append(selected)
        selected_low_std.append(selected[: max(1, int(len(selected) * std_percentile))])

    if not selected_all:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), grid_shape
    return np.concatenate(selected_all), np.concatenate(selected_low_std), grid_shape


def _neighbours(row: int, col: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = shape
    return [
        (next_row, next_col)
        for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
        if 0 <= next_row < height and 0 <= next_col < width
    ]


def _grow_region(
    seed: tuple[int, int],
    all_counts: np.ndarray,
    red_counts: np.ndarray,
    block_size: int,
    percentile: float,
) -> tuple[list[tuple[int, int]], int, int]:
    region = [seed]
    members = {seed}
    all_total = int(all_counts[seed])
    red_total = int(red_counts[seed])
    while True:
        grew = False
        current_tail = _log_binomial_tail(red_total, all_total, block_size, percentile)
        for row, col in list(region):
            for neighbour in _neighbours(row, col, all_counts.shape):
                if neighbour in members or all_counts[neighbour] <= 0:
                    continue
                next_all = all_total + int(all_counts[neighbour])
                next_red = red_total + int(red_counts[neighbour])
                next_tail = _log_binomial_tail(next_red, next_all, block_size, percentile)
                if current_tail - math.log(len(region)) > math.log(4.062570) + next_tail - math.log(len(region) + 1):
                    members.add(neighbour)
                    region.append(neighbour)
                    all_total = next_all
                    red_total = next_red
                    current_tail = next_tail
                    grew = True
        if not grew:
            return region, all_total, red_total


def _detect(image: np.ndarray, block_size: int, samples_per_bin: int, low_energy_percentile: float, std_percentile: float, cell_size: int) -> tuple[bool, np.ndarray, float]:
    height, width = image.shape[:2]
    visualization = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if min(height, width) < block_size:
        return False, visualization, -60.0

    cells = ((height + cell_size - 1) // cell_size, (width + cell_size - 1) // cell_size)
    all_counts = np.zeros(cells, dtype=np.int32)
    red_counts = np.zeros(cells, dtype=np.int32)
    block_grid_width = width - block_size + 1
    for channel in range(3):
        all_blocks, red_blocks, _ = _channel_candidates(
            image, channel, block_size, samples_per_bin, low_energy_percentile, std_percentile
        )
        for positions, counts in ((all_blocks, all_counts), (red_blocks, red_counts)):
            if len(positions):
                rows, cols = np.divmod(positions, block_grid_width)
                np.add.at(counts, (rows // cell_size, cols // cell_size), 1)

    detected = np.zeros(cells, dtype=bool)
    best_log_nfa = math.inf
    for row in range(cells[0]):
        for col in range(cells[1]):
            if all_counts[row, col] <= 0 or detected[row, col]:
                continue
            if red_counts[row, col] / all_counts[row, col] <= std_percentile:
                continue
            region, all_total, red_total = _grow_region(
                (row, col), all_counts, red_counts, block_size, std_percentile
            )
            log_nfa = _log_nfa(
                (height, width), len(region), red_total, all_total,
                block_size, std_percentile, cell_size,
            )
            best_log_nfa = min(best_log_nfa, log_nfa)
            if log_nfa < 0:
                for region_row, region_col in region:
                    detected[region_row, region_col] = True

    mask = np.repeat(np.repeat(detected, cell_size, axis=0), cell_size, axis=1)[:height, :width]
    if mask.any():
        edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        visualization[edge] = [255, 255, 0]
    return bool(best_log_nfa < 0), visualization, _significance(best_log_nfa)


class PRNUAnalyzer:
    """Compatibility wrapper exposing the Noisesniffer detector tuple."""

    def __init__(
        self,
        noise_filter_sigma: float = 3.0,
        window_size: int = 64,
        stride: int = 32,
        variance_threshold: float | None = None,
        *,
        block_size: int = 3,
        samples_per_bin: int = 20_000,
        low_energy_percentile: float = 0.1,
        std_percentile: float = 0.5,
        cell_size: int = 100,
    ) -> None:
        self.noise_filter_sigma = noise_filter_sigma
        self.window_size = window_size
        self.stride = stride
        self.variance_threshold = variance_threshold
        self.block_size = block_size
        self.samples_per_bin = samples_per_bin
        self.low_energy_percentile = low_energy_percentile
        self.std_percentile = std_percentile
        self.cell_size = cell_size
        if block_size not in (3, 5, 8):
            raise ValueError("block_size must be 3, 5, or 8")
        if samples_per_bin <= 0 or cell_size <= 0:
            raise ValueError("samples_per_bin and cell_size must be positive")
        if not 0 < low_energy_percentile <= 1 or not 0 < std_percentile <= 1:
            raise ValueError("percentiles must be in (0, 1]")

    def detect_tampering(self, image_input: ImageInput, overlay_alpha: float = 0.8) -> tuple[bool, np.ndarray, float]:
        """Return tampering flag, edge-preserving map, and ``-log10(NFA)``."""
        del overlay_alpha  # retained for callers of the former implementation
        image = _analysis_image(_as_rgb_float32(image_input))
        detected, visualization, score = _detect(
            image,
            self.block_size,
            self.samples_per_bin,
            self.low_energy_percentile,
            self.std_percentile,
            self.cell_size,
        )
        if self.variance_threshold is not None:
            # Compatibility-only decision for callers of the removed variance
            # API. The detector adapter does not use this path; its score is
            # always the Noisesniffer NFA significance above.
            detected = _legacy_uniformity(image, self.window_size, self.stride) > self.variance_threshold
        return detected, visualization, score

    def extract_noise_residual(self, image: np.ndarray) -> np.ndarray:
        """Return a small compatibility residual for legacy callers."""
        rgb = _as_rgb_float32(image)
        return rgb - cv2.GaussianBlur(rgb, (0, 0), self.noise_filter_sigma)

    def analyze(self, image_input: ImageInput) -> tuple[np.ndarray, np.ndarray]:
        image = _as_rgb_float32(image_input)
        return image.astype(np.uint8), self.extract_noise_residual(image)


def prnu_uniformity(image: ImageInput, **kwargs: object) -> tuple[bool, float, np.ndarray]:
    """Compatibility entry point; the returned raw score is ``-log10(NFA)``."""
    analyzer = PRNUAnalyzer(**kwargs)
    flagged, visualization, score = analyzer.detect_tampering(image)
    return flagged, score, visualization


def _legacy_uniformity(image: np.ndarray, window_size: int, stride: int) -> float:
    """Preserve the old explicit-threshold wrapper without feeding fusion."""
    if min(image.shape[:2]) < window_size:
        return 0.0
    denoised = median_filter(image, size=3)
    noise = image - gaussian_filter(denoised, sigma=3.0)
    values = [
        float(np.var(noise[top : top + window_size, left : left + window_size]))
        for top in range(0, image.shape[0] - window_size + 1, stride)
        for left in range(0, image.shape[1] - window_size + 1, stride)
    ]
    return float(np.mean(values)) if values else 0.0
