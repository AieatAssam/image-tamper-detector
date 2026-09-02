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

from math import ceil, log10
from time import perf_counter

import cv2
import numpy as np
from scipy.fft import dct
from scipy.ndimage import find_objects
from scipy.stats import binom

from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext, to_probability


BLOCK_SIZE = 8
GRID_COUNT = BLOCK_SIZE * BLOCK_SIZE
NEIGHBORHOOD = 9
SCORE_THRESHOLD = 0.0
SCORE_SCALE = 1.0


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return np.rint(
        rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    ).astype(np.float32)


def _vote_map(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the paper's per-pixel winning grid phase and zero strength."""
    height, width = gray.shape
    best = np.zeros((height, width), dtype=np.int16)
    votes = np.full((height, width), -1, dtype=np.int16)
    windows = np.lib.stride_tricks.sliding_window_view(gray, (BLOCK_SIZE, BLOCK_SIZE))
    for grid_y in range(BLOCK_SIZE):
        for grid_x in range(BLOCK_SIZE):
            blocks = windows[grid_y::BLOCK_SIZE, grid_x::BLOCK_SIZE]
            block_rows, block_columns = blocks.shape[:2]
            if not block_rows or not block_columns:
                continue
            blocks = np.ascontiguousarray(blocks.reshape(-1, BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)
            coefficients = dct(dct(blocks, type=2, norm="ortho", axis=-1), type=2, norm="ortho", axis=-2)
            zero_counts = np.count_nonzero(
                np.abs(coefficients.reshape(len(blocks), -1)[:, 1:]) < 0.5,
                axis=1,
            )
            constant_horizontal = np.all(blocks == blocks[:, :, :1], axis=(1, 2))
            constant_vertical = np.all(blocks == blocks[:, :1, :], axis=(1, 2))
            valid = ~(constant_horizontal | constant_vertical)
            zero_grid = zero_counts.astype(np.int16).reshape(block_rows, block_columns)
            valid_grid = valid.reshape(block_rows, block_columns)
            zero_map = np.repeat(np.repeat(zero_grid, BLOCK_SIZE, axis=0), BLOCK_SIZE, axis=1)
            valid_map = np.repeat(np.repeat(valid_grid, BLOCK_SIZE, axis=0), BLOCK_SIZE, axis=1)
            row_end = grid_y + block_rows * BLOCK_SIZE
            column_end = grid_x + block_columns * BLOCK_SIZE
            phase_zeros = np.zeros((height, width), dtype=np.int16)
            phase_valid = np.zeros((height, width), dtype=bool)
            phase_zeros[grid_y:row_end, grid_x:column_end] = zero_map
            phase_valid[grid_y:row_end, grid_x:column_end] = valid_map
            greater = phase_zeros > best
            equal = phase_zeros == best
            best[greater] = phase_zeros[greater]
            votes[greater] = np.where(phase_valid[greater], grid_y * BLOCK_SIZE + grid_x, -1)
            votes[~greater & equal] = -1

    votes[: BLOCK_SIZE - 1] = -1
    votes[-(BLOCK_SIZE - 1) :] = -1
    votes[:, : BLOCK_SIZE - 1] = -1
    votes[:, -(BLOCK_SIZE - 1) :] = -1
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
    """Evaluate the paper's conservative /64 Bonferroni-corrected log10 NFA."""
    height, width = image_shape
    if votes <= 0 or support <= 0:
        return np.inf
    n = max(1, int(ceil(support / GRID_COUNT)))
    k = min(n, max(1, int(ceil(votes / GRID_COUNT))))
    tests_log10 = 2.0 * log10(GRID_COUNT) + 2.0 * log10(max(1, height * width))
    return float(max(-1_000_000.0, tests_log10 + _log10_binomial_tail(n, k)))


def _global_nfas(votes: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    valid_votes = votes[votes >= 0]
    counts = np.bincount(valid_votes, minlength=GRID_COUNT) if len(valid_votes) else np.zeros(GRID_COUNT, dtype=int)
    nfas = np.full(GRID_COUNT, np.inf, dtype=float)
    for grid in range(GRID_COUNT):
        nfas[grid] = _log10_nfa(int(counts[grid]), len(valid_votes), votes.shape)
    meaningful = np.flatnonzero(nfas < 0)
    dominant = int(meaningful[np.argmin(nfas[meaningful])]) if len(meaningful) else -1
    return counts, nfas, dominant


def _foreign_regions(
    votes: np.ndarray,
    excluded: set[int],
    allowed: set[int] | None = None,
) -> tuple[np.ndarray, float, int, int]:
    mask = np.zeros(votes.shape, dtype=np.uint8)

    region_nfa = np.inf
    region_count = 0
    region_area = 0
    neighborhood = np.ones((2 * NEIGHBORHOOD + 1, 2 * NEIGHBORHOOD + 1), dtype=np.uint8)
    grids = range(GRID_COUNT) if allowed is None else sorted(allowed)
    for grid in grids:
        if grid in excluded:
            continue
        candidate = (votes == grid).astype(np.uint8)
        if not np.any(candidate):
            continue
        grown = cv2.dilate(candidate, neighborhood)
        component_count, labels, _, _ = cv2.connectedComponentsWithStats(grown, 8)
        # One pass per grid rather than one full-image pass per label: the
        # per-label pixel count and bounding box are the only quantities the
        # NFA needs, and both come from the masked label image directly.
        region_labels = np.where(candidate != 0, labels, 0)
        pixel_counts = np.bincount(region_labels.ravel(), minlength=component_count)
        boxes = find_objects(region_labels, max_label=component_count - 1)
        accepted = []
        for label in range(1, component_count):
            support = int(pixel_counts[label])
            box = boxes[label - 1]
            if support == 0 or box is None:
                continue
            row_slice, column_slice = box
            bbox_area = (row_slice.stop - row_slice.start) * (column_slice.stop - column_slice.start)
            current_nfa = _log10_nfa(support, bbox_area, votes.shape)
            if current_nfa < 0:
                accepted.append(label)
                region_count += 1
                region_area += support
                region_nfa = min(region_nfa, current_nfa)
        if accepted:
            mask[np.isin(region_labels, accepted)] = 255

    if np.any(mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, neighborhood)
    return mask, region_nfa, region_count, region_area


class ZeroDetector:
    id = "zero"
    name = "ZERO JPEG Grid Origin"
    family = "compression"
    applicable_formats = frozenset({"JPEG", "PNG", "WEBP", "TIFF"})
    produces_map = True
    description = "Finds statistically meaningful foreign or missing JPEG grid phases from DCT zeros."
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
        foreign_mask, local_nfa, region_count, region_area = _foreign_regions(
            votes,
            excluded={dominant} if dominant >= 0 else set(),
        )
        missing_mask = np.zeros_like(foreign_mask)
        missing_nfa = np.inf
        missing_count = 0
        missing_area = 0
        if dominant >= 0:
            encoded_ok, encoded = cv2.imencode(
                ".jpg",
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 99],
            )
            if encoded_ok:
                recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                recompressed_rgb = cv2.cvtColor(recompressed, cv2.COLOR_BGR2RGB)
                secondary_votes, _ = _vote_map(_luminance(recompressed_rgb))
                secondary_votes[votes == dominant] = -1
                missing_mask, missing_nfa, missing_count, missing_area = _foreign_regions(
                    secondary_votes,
                    excluded=set(range(1, GRID_COUNT)),
                    allowed={0},
                )
        foreign_mask = np.maximum(foreign_mask, missing_mask)
        global_foreign_nfa = min(
            (float(global_nfas[grid]) for grid in range(GRID_COUNT) if grid != dominant),
            default=np.inf,
        )
        evidence_nfa = min(local_nfa, global_foreign_nfa, missing_nfa)
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
                "missing_grid_log10_nfa": missing_nfa if np.isfinite(missing_nfa) else 1.0,
                "missing_region_count": float(missing_count),
                "missing_region_area": float(missing_area),
                "mean_ac_zero_count": float(np.mean(zero_counts)),
            },
            visualization=foreign_mask,
            duration_ms=_duration(started),
        )


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
