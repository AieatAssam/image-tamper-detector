"""Independent reimplementation of ZERO (IPOL 2021/390), paper only.

ZERO: a Local JPEG Grid Origin Detector Based on the Number of DCT Zeros and
its Applications in Image Forensics, Nikoukhah et al., Image Processing On
Line 11 (2021), https://doi.org/10.5201/ipol.2021.390.

The IPOL reference implementation is AGPL-3.0-or-later and is deliberately
not used here. This module derives the method from the paper: overlapping DCT
blocks vote for one of the 64 grid phases, and binomial a-contrario tests
validate global and local foreign-grid evidence.
"""

from __future__ import annotations

from math import log10
from time import perf_counter

import cv2
import numpy as np
from scipy.fft import dct
from scipy.stats import binom

from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext, to_probability


BLOCK_SIZE = 8
GRID_COUNT = BLOCK_SIZE * BLOCK_SIZE
NEIGHBORHOOD = 9
CELL_SIZE = 32
SAMPLE_OFFSETS = (8, 24)
SCORE_THRESHOLD = 0.0
SCORE_SCALE = 1.0


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return np.rint(
        rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    ).astype(np.float32)


def _vote_map(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute a bounded per-cell winning grid phase and vote strength.

    Every cell evaluates four 8x8 blocks for each of the 64 candidate origins.
    This retains the paper's phase comparison while avoiding an exhaustive
    overlapping-window DCT over large uploads.
    """
    height, width = gray.shape
    cell_rows = (height + CELL_SIZE - 1) // CELL_SIZE
    cell_columns = (width + CELL_SIZE - 1) // CELL_SIZE
    counts = np.full((GRID_COUNT, cell_rows, cell_columns), -1, dtype=np.int16)
    for grid_y in range(BLOCK_SIZE):
        for grid_x in range(BLOCK_SIZE):
            origins = []
            for cell_y in range(cell_rows):
                for cell_x in range(cell_columns):
                    for offset_y in SAMPLE_OFFSETS:
                        row_target = cell_y * CELL_SIZE + offset_y
                        row = row_target - (row_target - grid_y) % BLOCK_SIZE
                        row = min(height - BLOCK_SIZE, max(0, row))
                        for offset_x in SAMPLE_OFFSETS:
                            column_target = cell_x * CELL_SIZE + offset_x
                            column = column_target - (column_target - grid_x) % BLOCK_SIZE
                            column = min(width - BLOCK_SIZE, max(0, column))
                            origins.append((row, column))
            blocks = np.asarray(
                [gray[row : row + BLOCK_SIZE, column : column + BLOCK_SIZE] for row, column in origins],
                dtype=np.float32,
            )
            coefficients = dct(dct(blocks, type=2, norm="ortho", axis=-1), type=2, norm="ortho", axis=-2)
            zero_counts = np.count_nonzero(
                np.abs(coefficients[:, 1:, :].reshape(len(origins), -1)) < 0.5,
                axis=1,
            )
            constant = np.std(blocks, axis=(1, 2)) == 0
            values = zero_counts.astype(np.int16).reshape(cell_rows, cell_columns, 4)
            values[constant.reshape(cell_rows, cell_columns, 4)] = 0
            valid = (~constant).reshape(cell_rows, cell_columns, 4).any(axis=2)
            values = values.sum(axis=2)
            values[~valid] = -1
            counts[grid_y * BLOCK_SIZE + grid_x] = values

    best = counts.max(axis=0)
    unique = (counts == best).sum(axis=0) == 1
    votes = np.argmax(np.where(unique[None, ...], counts, -1), axis=0).astype(np.int16)
    votes[~unique] = -1
    return votes, best


def _log10_binomial_tail(n: int, k: int) -> float:
    if k <= 0:
        return 0.0
    if n <= 0 or k > n:
        return -1_000_000.0
    tail = float(binom.logsf(k - 1, n, 1.0 / GRID_COUNT))
    if not np.isfinite(tail):
        return -1_000_000.0
    return max(-1_000_000.0, tail / np.log(10.0))


def _log10_nfa(votes: int, support: int, image_shape: tuple[int, int]) -> float:
    """Evaluate the paper's Bonferroni-corrected log10 NFA."""
    height, width = image_shape
    if votes <= 0 or support <= 0:
        return np.inf
    # Votes already represent one selected block per candidate phase in each
    # coarse cell, so the paper's block-to-grid reduction is not repeated.
    n = support
    k = min(n, votes)
    tests_log10 = log10(GRID_COUNT) + 2.0 * log10(max(1, height * width))
    return float(max(-1_000_000.0, tests_log10 + _log10_binomial_tail(n, k)))


def _global_nfas(votes: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    valid_votes = votes[votes >= 0]
    counts = np.bincount(valid_votes, minlength=GRID_COUNT) if len(valid_votes) else np.zeros(GRID_COUNT, dtype=int)
    nfas = np.full(GRID_COUNT, np.inf, dtype=float)
    for grid in range(GRID_COUNT):
        nfas[grid] = _log10_nfa(int(counts[grid]), len(valid_votes), votes.shape)
    dominant = int(np.argmax(counts)) if len(valid_votes) else -1
    return counts, nfas, dominant


def _foreign_regions(votes: np.ndarray, dominant: int, dominant_nfa: float) -> tuple[np.ndarray, float, int, int]:
    mask = np.zeros(votes.shape, dtype=np.uint8)
    if dominant < 0:
        return mask, np.inf, 0, 0

    region_nfa = np.inf
    region_count = 0
    region_area = 0
    neighborhood = np.ones((NEIGHBORHOOD, NEIGHBORHOOD), dtype=np.uint8)
    for grid in range(GRID_COUNT):
        if grid == dominant:
            continue
        candidate = (votes == grid).astype(np.uint8)
        if not np.any(candidate):
            continue
        grown = cv2.dilate(candidate, neighborhood)
        component_count, labels, _, _ = cv2.connectedComponentsWithStats(grown, 8)
        for label in range(1, component_count):
            region = (candidate != 0) & (labels == label)
            ys, xs = np.nonzero(region)
            if not len(xs):
                continue
            bbox_area = (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1)
            current_nfa = _log10_nfa(len(xs), bbox_area, votes.shape)
            if current_nfa < 0:
                mask[region] = 255
                region_count += 1
                region_area += len(xs)
                region_nfa = min(region_nfa, current_nfa)

    if np.any(mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, neighborhood)
    return mask, region_nfa, region_count, region_area


class ZeroDetector:
    id = "zero"
    name = "ZERO JPEG Grid Origin"
    family = "compression"
    applicable_formats = frozenset({"JPEG", "PNG", "WEBP", "TIFF"})
    produces_map = True
    description = "Finds statistically meaningful foreign JPEG grid phases from DCT zeros."
    limitations = [
        "Needs a sufficiently large image with visible JPEG traces; it cannot detect a foreign grid that was erased or re-aligned.",
        "The score is evidence, not proof of manipulation.",
    ]

    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        height, width = ctx.downscaled_rgb_uint8.shape[:2]
        if min(height, width) < 32:
            return False, "ZERO requires an image at least 32 pixels on its shorter side"
        return True, "image is large enough for overlapping 8x8 grid analysis"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        applicable, reason = self.applicable(ctx)
        if not applicable:
            return DetectorResult(
                detector_id=self.id,
                state=DetectorState.NOT_APPLICABLE,
                score=None,
                flagged=None,
                threshold=SCORE_THRESHOLD,
                reason=reason,
                metrics={},
                visualization=None,
                duration_ms=_duration(started),
            )

        # Preserve the pixel grid. Resampling here would erase the phase this
        # detector measures; ImageContext already limits shared work to 1600px.
        rgb = ctx.downscaled_rgb_uint8
        gray = _luminance(rgb)
        votes, zero_counts = _vote_map(gray)
        counts, global_nfas, dominant = _global_nfas(votes)
        dominant_nfa = float(global_nfas[dominant]) if dominant >= 0 else np.inf
        foreign_mask, local_nfa, region_count, region_area = _foreign_regions(votes, dominant, dominant_nfa)
        foreign_mask = np.repeat(np.repeat(foreign_mask, CELL_SIZE, axis=0), CELL_SIZE, axis=1)[: gray.shape[0], : gray.shape[1]]
        global_foreign_nfa = min(
            (float(global_nfas[grid]) for grid in range(GRID_COUNT) if grid != dominant),
            default=np.inf,
        )
        evidence_nfa = min(local_nfa, global_foreign_nfa)
        evidence = -1.0 if not np.isfinite(evidence_nfa) else -evidence_nfa
        score = to_probability(evidence, SCORE_THRESHOLD, SCORE_SCALE, True)
        flagged = score >= 0.5
        valid_count = int(np.count_nonzero(votes >= 0))
        meaningful_global = int(np.count_nonzero(global_nfas < 0))
        return DetectorResult(
            detector_id=self.id,
            state=DetectorState.APPLICABLE,
            score=score,
            flagged=flagged,
            threshold=SCORE_THRESHOLD,
            reason=(
                f"dominant grid {dominant} log10_nfa {dominant_nfa:.3f}; "
                f"foreign evidence log10_nfa {evidence_nfa:.3f}"
                if np.isfinite(evidence_nfa)
                else f"dominant grid {dominant} has no meaningful foreign-grid evidence"
            ),
            metrics={
                "dominant_phase": float(dominant),
                "dominant_log10_nfa": dominant_nfa,
                "foreign_log10_nfa": evidence_nfa if np.isfinite(evidence_nfa) else 1.0,
                "foreign_grid_strength": max(0.0, evidence),
                "meaningful_global_grids": float(meaningful_global),
                "valid_vote_fraction": valid_count / float(max(1, votes.size)),
                "foreign_region_count": float(region_count),
                "foreign_region_area": float(region_area),
                "mean_ac_zero_count": float(np.mean(zero_counts)),
            },
            visualization=foreign_mask,
            duration_ms=_duration(started),
        )


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
