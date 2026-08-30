#!/usr/bin/env python3
"""Fit detector thresholds and fusion weights from the available corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.analysis.base import DetectorState, ImageContext, to_probability  # noqa: E402
from backend.app.analysis.registry import get_all, run_all  # noqa: E402

RAW_KEYS = {
    "ela": "edge_discontinuity",
    "prnu": "uniformity_score",
    "entropy": "matching_proportion",
    "qtable": "libjpeg_distance",
    "double_jpeg": "aggregate",
    "jpeg_ghosts": "distinct_modes",
    "copy_move": "verified_clusters",
    "cfa": "cfa_ratio",
    "zero": "foreign_grid_strength",
    "spectral": "peak_to_sigma",
    "learned": "deepfake_probability",
    "c2pa": "generative_assertion",
}
HIGHER_WORSE = {"entropy": False, "qtable": False}
FALSE_POSITIVE_FAMILIES = ("authentic_recompress", "resize_then_save")
FALSE_POSITIVE_LIMIT = 0.10
VERDICT_THRESHOLD = 0.55
L2 = 0.05


def auc(scores: list[float], labels: list[bool]) -> float | None:
    """Return tie-aware AUC, or None when the sample cannot support it."""
    if len(scores) < 2 or len(set(labels)) < 2:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(order)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and scores[order[end]] == scores[order[index]]:
            end += 1
        rank = (index + end + 1) / 2.0
        for position in range(index, end):
            ranks[order[position]] = rank
        index = end
    positives = sum(labels)
    negatives = len(labels) - positives
    return (sum(rank for rank, label in zip(ranks, labels) if label) - positives * (positives + 1) / 2) / (positives * negatives)


def within_source_auc(values: list[tuple[str, float, bool]]) -> float | None:
    """Return AUC using only positive/negative pairs from the same source image."""
    grouped: dict[str, tuple[list[float], list[float]]] = {}
    for source, score, label in values:
        positives, negatives = grouped.setdefault(source, ([], []))
        (positives if label else negatives).append(float(score))

    wins = ties = pairs = 0
    for positives, negatives in grouped.values():
        if not positives or not negatives:
            continue
        pairs += len(positives) * len(negatives)
        for positive in positives:
            for negative in negatives:
                if positive > negative:
                    wins += 1
                elif positive == negative:
                    ties += 1
    return (wins + ties / 2.0) / pairs if pairs else None


def _within_source_counts(values: list[tuple[str, float, bool]]) -> tuple[int, int]:
    grouped: dict[str, tuple[int, int]] = {}
    for source, _score, label in values:
        positives, negatives = grouped.setdefault(source, (0, 0))
        grouped[source] = (positives + int(label), negatives + int(not label))
    participating = [(positives, negatives) for positives, negatives in grouped.values() if positives and negatives]
    return sum(positives for positives, _ in participating), sum(negatives for _, negatives in participating)


def hanley_mcneil_se(auc_value: float | None, n_pos: int, n_neg: int) -> float | None:
    """Return Hanley-McNeil AUC standard error when both classes are present."""
    if auc_value is None or n_pos < 1 or n_neg < 1:
        return None
    q1 = auc_value / (2.0 - auc_value)
    q2 = 2.0 * auc_value * auc_value / (1.0 + auc_value)
    variance = (
        auc_value * (1.0 - auc_value)
        + (n_pos - 1) * (q1 - auc_value * auc_value)
        + (n_neg - 1) * (q2 - auc_value * auc_value)
    ) / (n_pos * n_neg)
    return math.sqrt(max(0.0, variance))


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end + 1) / 2.0
        for position in range(index, end):
            ranks[order[position]] = rank
        index = end
    return ranks


def spearman_rank_correlation(pairs: list[tuple[float, float]]) -> float | None:
    """Return tie-aware Spearman rho, or None when either rank is constant."""
    if len(pairs) < 2:
        return None
    weights, skills = zip(*pairs)
    weight_ranks = np.asarray(_rank(list(weights)), dtype=float)
    skill_ranks = np.asarray(_rank(list(skills)), dtype=float)
    if np.std(weight_ranks) == 0.0 or np.std(skill_ranks) == 0.0:
        return None
    return float(np.corrcoef(weight_ranks, skill_ranks)[0, 1])


def _manifest() -> dict:
    import yaml

    return yaml.safe_load((ROOT / "data/corpus/MANIFEST.yaml").read_text())


def _real_entries() -> list[dict]:
    real_dir = ROOT / "data/corpus/real"
    rows = []
    for item in _manifest().get("images", []):
        if item.get("path"):
            path = Path(item["path"])
            path = path if path.is_absolute() else ROOT / path
        else:
            if not real_dir.is_dir():
                continue
            suffix = (Path(urlparse(item["url"]).path).suffix or ".jpg").lower()
            path = real_dir / f"{item['id']}{suffix}"
        if path.is_file():
            rows.append({
                "id": item["id"],
                "path": path,
                "label": item["label"] != "authentic",
                "family": item["axis"],
                "source_image": item.get("source_group", item.get("source_image", str(path.relative_to(ROOT)))),
                "corpus": "real",
            })
    return rows


def entries(corpus: str) -> list[dict]:
    rows = []
    if corpus in {"synthetic", "all"}:
        index_path = ROOT / "data/corpus/synthetic/index.json"
        if index_path.is_file():
            for item in json.loads(index_path.read_text())["entries"]:
                path = ROOT / item["path"]
                if path.is_file():
                    rows.append({**item, "path": path, "label": item["label"] != "authentic", "corpus": "synthetic"})
    if corpus in {"real", "all"}:
        rows.extend(_real_entries())
    return rows


def threshold(values: list[tuple[float, bool]], higher: bool) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    candidates = sorted({value for value, _ in values})
    best = (-float("inf"), candidates[len(candidates) // 2])
    positives = sum(label for _, label in values)
    negatives = len(values) - positives
    for point in candidates:
        predicted = [(value >= point) if higher else (value <= point) for value, _ in values]
        tpr = sum(p and y for p, (_, y) in zip(predicted, values)) / max(1, positives)
        fpr = sum(p and not y for p, (_, y) in zip(predicted, values)) / max(1, negatives)
        if tpr - fpr > best[0]:
            best = (tpr - fpr, point)
    raw = np.asarray([value for value, _ in values], dtype=float)
    scale = float(np.percentile(raw, 75) - np.percentile(raw, 25)) / 2.0
    return float(best[1]), max(scale, float(np.finfo(float).eps))


def fit_logistic(
    features: np.ndarray, labels: np.ndarray, present: np.ndarray | None = None
) -> tuple[float, np.ndarray]:
    """Fit logistic weights with the catalog's non-negative direction constraint."""
    present = np.ones_like(features, dtype=bool) if present is None else present
    observed = np.maximum(1, present.sum(axis=0))
    weights = np.zeros(features.shape[1], dtype=float)
    intercept = 0.0
    for _ in range(350):
        logits = np.clip(intercept + (features * present) @ weights, -30, 30)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = probabilities - labels
        intercept -= 0.08 * float(error.mean())
        gradient = (features * present).T @ error / observed + L2 * weights
        update = 0.08 * gradient
        # A negative coefficient reverses the detector's documented suspicion
        # direction. Projecting after each step drops anti-correlated evidence
        # instead of silently teaching fusion to invert it.
        weights = np.maximum(0.0, weights - update)
    return intercept, weights


def fit_standardized_logistic(
    features: np.ndarray, labels: np.ndarray, present: np.ndarray | None = None
) -> tuple[float, np.ndarray]:
    """Fit on z-scored columns, then return coefficients for raw runtime features."""
    present = np.ones_like(features, dtype=bool) if present is None else present
    observed = np.maximum(1, present.sum(axis=0))
    means = (features * present).sum(axis=0) / observed
    centered = np.where(present, features - means, 0.0)
    scales = np.sqrt((centered * centered).sum(axis=0) / observed)
    scales = np.where(scales > np.finfo(float).eps, scales, 1.0)
    standardized = np.where(present, (features - means) / scales, 0.0)
    intercept, standardized_weights = fit_logistic(standardized, labels, present)
    return float(intercept - np.dot(standardized_weights, means / scales)), standardized_weights / scales


def _raw_value(detector_id: str, result) -> float | None:
    if result is None or result.state is not DetectorState.APPLICABLE:
        return None
    if detector_id == "exif":
        if "raw_score" in result.metrics:
            return float(result.metrics["raw_score"])
        values = [
            value for key, value in result.metrics.items()
            if key in {"thumbnail_difference", "editor_software", "missing_camera_block", "datetime_disagreement", "dimension_disagreement"}
        ]
        return max(values) if values else None
    key = RAW_KEYS.get(detector_id)
    value = result.metrics.get(key) if key else None
    return float(value) if value is not None else None


def _group_split(rows: list[dict], seed: int) -> tuple[list[int], list[int]]:
    groups = sorted({row["source_image"] for row in rows})
    paired_sources = _paired_sources(rows)
    order = [str(group) for group in np.random.default_rng(seed).permutation(groups)]
    paired_rows = sum(row["source_image"] in paired_sources for row in rows)
    target = max(1, math.ceil(paired_rows * 0.7))
    train_groups: list[str] = []
    train_count = 0
    for group in order:
        if train_count < target or not train_groups:
            train_groups.append(group)
            train_count += sum(row["source_image"] == group for row in rows)
    if len(train_groups) == len(order) and len(order) > 1:
        train_groups.pop()
    train = [index for index, row in enumerate(rows) if row["source_image"] in train_groups]
    test = [index for index, row in enumerate(rows) if row["source_image"] not in train_groups]
    return train, test


def _paired_sources(rows: list[dict]) -> set[str]:
    labels_by_group: dict[str, set[bool]] = {}
    for row in rows:
        labels_by_group.setdefault(row["source_image"], set()).add(bool(row["label"]))
    return {group for group, labels in labels_by_group.items() if len(labels) == 2}


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def _logit(value: float) -> float:
    clipped = min(0.99, max(0.01, value))
    return math.log(clipped / (1.0 - clipped))


def _fusion_score(row: dict, detector_ids: list[str], configs: dict[str, dict], intercept: float) -> tuple[float, int]:
    z = intercept
    applicable = 0
    for detector_id in detector_ids:
        score = row.get("scores", {}).get(detector_id)
        if score is None:
            continue
        applicable += 1
        z += configs[detector_id]["weight"] * _logit(score)
    return _sigmoid(z), applicable


def _apply_false_positive_gate(rows: list[dict], detector_ids: list[str], configs: dict[str, dict], intercept: float) -> float:
    trap_by_family = {
        family: [index for index, row in enumerate(rows) if row["family"] == family and not row["label"]]
        for family in FALSE_POSITIVE_FAMILIES
    }
    trap_by_family = {family: indices for family, indices in trap_by_family.items() if indices}
    if not trap_by_family:
        return intercept

    def allowed(family: str) -> int:
        return math.floor(FALSE_POSITIVE_LIMIT * len(trap_by_family[family]))

    def passes(candidate: float) -> bool:
        for family, indices in trap_by_family.items():
            flagged = 0
            for index in indices:
                score, applicable = _fusion_score(rows[index], detector_ids, configs, candidate)
                flagged += applicable >= 3 and score >= VERDICT_THRESHOLD
            if flagged > allowed(family):
                return False
        return True

    if passes(intercept):
        return intercept
    low, high = -60.0, intercept
    for _ in range(80):
        middle = (low + high) / 2.0
        if passes(middle):
            low = middle
        else:
            high = middle
    return float(np.nextafter(low, -math.inf))


def _revision(rows: list[dict]) -> str | None:
    paths = {ROOT / "data/corpus/synthetic/index.json", ROOT / "data/corpus/MANIFEST.yaml"}
    paths.update(row["path"] for row in rows)
    existing = sorted(path for path in paths if path.is_file())
    if not existing:
        return None
    digest = hashlib.sha256()
    for path in existing:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=("synthetic", "real", "all"), default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    rows = entries(args.corpus)
    detectors = list(get_all().values())
    detector_ids = [detector.id for detector in detectors]
    train_indices, test_indices = _group_split(rows, args.seed) if rows else ([], [])
    paired_sources = _paired_sources(rows)
    fit_indices = [index for index in train_indices if rows[index]["source_image"] in paired_sources]
    raw_by_image: list[dict[str, float]] = []
    for row in rows:
        results = {result.detector_id: result for result in run_all(ImageContext(row["path"].read_bytes()), detector_ids)}
        raw_by_image.append({
            detector_id: raw
            for detector_id in detector_ids
            if (raw := _raw_value(detector_id, results.get(detector_id))) is not None
        })

    configs: dict[str, dict] = {}
    for detector in detectors:
        values = [
            (raw_by_image[index][detector.id], rows[index]["label"])
            for index in fit_indices
            if detector.id in raw_by_image[index]
        ]
        higher = detector.id not in HIGHER_WORSE
        fitted = bool(values)
        t, scale = threshold(values, higher)
        score_by_index: dict[int, float] = {
            index: to_probability(raw_by_image[index][detector.id], t, scale, higher)
            for index in range(len(rows))
            if detector.id in raw_by_image[index] and fitted
        }
        all_within_source_auc = within_source_auc([
            (rows[index]["source_image"], score, rows[index]["label"])
            for index, score in score_by_index.items()
        ])
        all_within_values = [
            (rows[index]["source_image"], score, rows[index]["label"])
            for index, score in score_by_index.items()
        ]
        all_n_pos, all_n_neg = _within_source_counts(all_within_values)
        all_se = hanley_mcneil_se(all_within_source_auc, all_n_pos, all_n_neg)
        heldout_values = [
            (rows[index]["source_image"], score_by_index[index], rows[index]["label"])
            for index in test_indices
            if index in score_by_index
        ]
        heldout_within_source_auc = within_source_auc(heldout_values)
        heldout_n_pos, heldout_n_neg = _within_source_counts(heldout_values)
        heldout_se = hanley_mcneil_se(heldout_within_source_auc, heldout_n_pos, heldout_n_neg)
        drop = all_within_source_auc is None or all_se is None or all_within_source_auc <= 0.5 + all_se
        configs[detector.id] = {
            "threshold": t,
            "scale": scale,
            "weight": 0.0,
            "higher_is_worse": higher,
            "within_source_auc": all_within_source_auc,
            "heldout_auc": heldout_within_source_auc,
            "clipped": False,
            "weight_guard": {
                "metric": "within_source_auc",
                "auc": all_within_source_auc,
                "se": all_se,
                "n_pos": all_n_pos,
                "n_neg": all_n_neg,
                "drop": drop,
                "rule": "keep only when within_source_auc > 0.5 + Hanley-McNeil SE",
            },
        }
        for index, score in score_by_index.items():
            rows[index].setdefault("scores", {})[detector.id] = score
        if not fitted:
            configs[detector.id]["threshold"] = 0.0
            configs[detector.id]["scale"] = 1.0
            configs[detector.id]["weight_reason"] = "heldout_auc unavailable: no applicable observations"
        elif drop:
            if all_within_source_auc is None or all_se is None:
                configs[detector.id]["weight_reason"] = "within_source_auc unavailable for Hanley-McNeil guard"
            else:
                configs[detector.id]["weight_reason"] = f"within_source_auc={all_within_source_auc:.6f} <= 0.5 + SE={all_se:.6f}"

    fit_ids = [
        detector_id for detector_id in detector_ids
        if not configs[detector_id]["weight_guard"]["drop"]
        and any(detector_id in rows[index].get("scores", {}) for index in fit_indices)
    ]
    fitted_weights: dict[str, float] = {}
    if fit_ids and fit_indices:
        present = np.asarray(
            [[detector_id in rows[index].get("scores", {}) for detector_id in fit_ids] for index in fit_indices],
            dtype=bool,
        )
        features = np.asarray(
            [[_logit(rows[index].get("scores", {}).get(detector_id, 0.5)) if detector_id in rows[index].get("scores", {}) else 0.0 for detector_id in fit_ids] for index in fit_indices],
            dtype=float,
        )
        intercept, fitted_weights = fit_standardized_logistic(
            features, np.asarray([rows[index]["label"] for index in fit_indices], dtype=float), present
        )
        fitted_weights = dict(zip(fit_ids, (float(weight) for weight in fitted_weights)))
        for detector_id, fitted_weight in fitted_weights.items():
            configs[detector_id]["weight"] = fitted_weight
    else:
        intercept = 0.0

    intercept = _apply_false_positive_gate(rows, detector_ids, configs, float(intercept))
    skill_pairs = [
        (float(config["weight"]), float(config["heldout_auc"]))
        for config in configs.values()
        if config.get("heldout_auc") is not None
    ]
    weight_skill_spearman = spearman_rank_correlation(skill_pairs)
    if weight_skill_spearman is not None:
        assert weight_skill_spearman > 0.0, f"weight/heldout-skill Spearman correlation must be positive: {weight_skill_spearman}"
    print(f"weight/heldout-skill Spearman={weight_skill_spearman}")
    statistical_weights = [
        config["weight"] for detector_id, config in configs.items()
        if detector_id != "c2pa" and config["weight"] > 0.0
    ]
    if statistical_weights:
        cap = max(0.0, (_logit(0.80) - intercept) / _logit(0.99))
        factor = min(1.0, cap / max(statistical_weights))
        for detector_id, config in configs.items():
            if detector_id == "c2pa" or config["weight"] <= 0.0:
                continue
            weight = config["weight"] * factor
            if weight != config["weight"]:
                config["weight_unclipped"] = fitted_weights[detector_id]
                config["clipped"] = True
            config["weight"] = weight
    intercept = _apply_false_positive_gate(rows, detector_ids, configs, float(intercept))
    fused_scores = []
    fused_labels = []
    fused_within_source = []
    for index in test_indices:
        score, _ = _fusion_score(rows[index], detector_ids, configs, intercept)
        fused_scores.append(score)
        fused_labels.append(rows[index]["label"])
        fused_within_source.append((rows[index]["source_image"], score, rows[index]["label"]))

    corpora = sorted({row["corpus"] for row in rows})
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    output = {
        "version": "2026-08-28",
        "generated_at": generated_at,
        "legacy": {"prnu": {"variance_threshold": 0.001}, "entropy": {"matching_threshold": 0.35}},
        "fitted_on": {"corpus_revision": _revision(rows), "n_images": len(rows), "corpora": corpora},
        "detectors": configs,
        "weight_skill_spearman": weight_skill_spearman,
        "fusion": {"method": "weighted_logit", "intercept": float(intercept)},
        "heldout": {"split_by": "source_image", "n": len(fused_scores), "auc": within_source_auc(fused_within_source), "seed": args.seed},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
