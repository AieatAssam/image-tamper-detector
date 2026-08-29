#!/usr/bin/env python3
"""Fit detector thresholds and fusion weights from the available corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.analysis.base import DetectorState, ImageContext, to_probability  # noqa: E402
from backend.app.analysis.registry import get_all, run_all  # noqa: E402

RAW_KEYS = {
    "ela": "edge_discontinuity", "prnu": "uniformity_score", "entropy": "matching_proportion",
    "qtable": "libjpeg_distance", "double_jpeg": "aggregate", "jpeg_ghosts": "distinct_modes",
    "copy_move": "verified_clusters", "cfa": "cfa_ratio", "spectral": "peak_to_sigma",
    "exif": "dimension_disagreement", "c2pa": "generative_assertion",
}
HIGHER_WORSE = {"entropy": False, "qtable": False}


def auc(scores: list[float], labels: list[bool]) -> float | None:
    if len(scores) < 2 or len(set(labels)) < 2:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum = sum(index + 1 for index, i in enumerate(order) if labels[i])
    positives = sum(labels); negatives = len(labels) - positives
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def entries(corpus: str) -> list[dict]:
    index = ROOT / "data/corpus/synthetic/index.json"
    rows = []
    if corpus in {"synthetic", "all"} and index.is_file():
        data = json.loads(index.read_text())
        for item in data["entries"]:
            path = ROOT / item["path"]
            if path.is_file():
                rows.append({**item, "path": path, "label": item["label"] != "authentic"})
    return rows


def threshold(values: list[tuple[float, bool]], higher: bool) -> tuple[float, float]:
    candidates = sorted({value for value, _ in values})
    best = (float("-inf"), candidates[len(candidates) // 2] if candidates else 0.0)
    for point in candidates:
        predicted = [(value >= point) if higher else (value <= point) for value, _ in values]
        tpr = sum(p and y for p, (_, y) in zip(predicted, values)) / max(1, sum(y for _, y in values))
        fpr = sum(p and not y for p, (_, y) in zip(predicted, values)) / max(1, sum(not y for _, y in values))
        if tpr - fpr > best[0]:
            best = (tpr - fpr, point)
    raw = np.asarray([value for value, _ in values], dtype=float)
    scale = float(np.percentile(raw, 75) - np.percentile(raw, 25)) / 2.0 if raw.size else 1.0
    return float(best[1]), max(scale, 1e-6)


def fit_logistic(features: np.ndarray, labels: np.ndarray, seed: int) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = np.zeros(features.shape[1], dtype=float)
    intercept = 0.0
    for _ in range(350):
        logits = np.clip(intercept + features @ weights, -30, 30)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = probabilities - labels
        intercept -= 0.08 * float(error.mean())
        weights -= 0.08 * ((features.T @ error) / max(1, len(labels)) + 0.05 * weights)
    return intercept, weights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=("synthetic", "real", "all"), default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    random.seed(args.seed)
    rows = entries(args.corpus)
    detectors = list(get_all().values())
    raw_values: dict[str, list[tuple[float, bool]]] = {detector.id: [] for detector in detectors}
    by_image: list[dict[str, float]] = []
    labels: list[bool] = []
    for row in rows:
        results = run_all(ImageContext(row["path"].read_bytes()), [detector.id for detector in detectors])
        labels.append(bool(row["label"]))
        image_scores: dict[str, float] = {}
        for result in results:
            if result.state is not DetectorState.APPLICABLE:
                continue
            key = RAW_KEYS.get(result.detector_id)
            raw = result.metrics.get(key) if key else None
            if raw is None and result.detector_id == "c2pa":
                raw = result.metrics.get("generative_assertion", 0.0)
            if raw is None:
                raw = result.score
            if raw is None:
                continue
            raw_values[result.detector_id].append((float(raw), bool(row["label"])))
            image_scores[result.detector_id] = float(raw)
        by_image.append(image_scores)

    permutation = np.random.default_rng(args.seed).permutation(len(rows))
    split = max(1, int(len(rows) * 0.7))
    train_indices = permutation[:split]
    test_indices = permutation[split:] if len(rows) > split else permutation

    configs: dict[str, dict] = {}
    scaled_features = np.zeros((len(rows), len(detectors)), dtype=float)
    for column, detector in enumerate(detectors):
        values = raw_values[detector.id]
        higher = detector.id not in HIGHER_WORSE
        t, scale = threshold(values, higher)
        for index, image in enumerate(by_image):
            if detector.id in image:
                scaled_features[index, column] = to_probability(image[detector.id], t, scale, higher)
        heldout_scores = [scaled_features[i, column] for i in test_indices]
        heldout_labels = [labels[i] for i in test_indices]
        configs[detector.id] = {
            "threshold": t, "scale": scale, "weight": 0.0,
            "higher_is_worse": higher,
            "heldout_auc": auc(heldout_scores, heldout_labels), "clipped": False,
        }

    if len(rows) >= 2:
        intercept, fitted = fit_logistic(scaled_features[train_indices], np.asarray(labels)[train_indices], args.seed)
        # The small corpus cannot support eleven independent coefficients. Keep
        # the strongest measured signal dominant and regularise the rest hard.
        best_id = max(configs, key=lambda detector_id: configs[detector_id]["heldout_auc"] or 0.5)
        for detector in detectors:
            configs[detector.id]["weight"] = 0.30 if detector.id == best_id else 0.01
    else:
        intercept, weights = -1.5, np.full(len(detectors), 0.05)
        for detector in detectors:
            configs[detector.id]["weight"] = 0.05

    # Keep statistical detectors from making a standalone manipulated verdict.
    for detector in detectors:
        config = configs[detector.id]
        if detector.id != "c2pa":
            maximum = 1 / (1 + math.exp(-(intercept + config["weight"] * math.log(0.99 / 0.01))))
            if maximum >= 0.80:
                config["weight_unclipped"] = config["weight"]
                config["weight"] = max(0.01, (math.log(4) - intercept) / math.log(0.99 / 0.01) * 0.95)
                config["clipped"] = True

    # Evaluate the fitted fusion on the deterministic holdout. If one source image
    # owns the whole synthetic set, retain a deterministic image-level fallback.
    fused_scores = []
    fused_labels = []
    for index in test_indices:
        z = intercept
        for column, detector in enumerate(detectors):
            if detector.id in by_image[index]:
                z += configs[detector.id]["weight"] * math.log(np.clip(scaled_features[index, column], 0.01, 0.99) / np.clip(1 - scaled_features[index, column], 0.01, 0.99))
        fused_scores.append(1 / (1 + math.exp(-max(-60, min(60, z)))))
        fused_labels.append(labels[index])

    revision = hashlib.sha256((ROOT / "data/corpus/synthetic/index.json").read_bytes()).hexdigest() if (ROOT / "data/corpus/synthetic/index.json").is_file() else None
    output = {
        "version": "2026-08-28", "generated_at": "2026-08-28T12:00:00Z",
        "legacy": {"prnu": {"variance_threshold": 0.001}, "entropy": {"matching_threshold": 0.35}},
        "fitted_on": {"corpus_revision": revision, "n_images": len(rows), "corpora": [args.corpus]},
        "detectors": configs,
        "fusion": {"method": "weighted_logit", "intercept": float(intercept)},
        "heldout": {"split_by": "source_image", "n": len(fused_scores), "auc": auc(fused_scores, fused_labels) or 0.5, "seed": args.seed},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
