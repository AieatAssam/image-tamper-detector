#!/usr/bin/env python3
"""Check whether image metadata alone predicts the corpus label."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FEATURE_GROUPS = ("all", "format", "dimensions", "file_size", "exif")
CURRENT_AI_AXES = frozenset({"sd35_flux", "synthbuster"})


def _auc(scores: list[float], labels: list[bool]) -> float | None:
    if len(set(labels)) < 2:
        return None
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=scores.__getitem__)
    rank_sum = 0.0
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and scores[order[end]] == scores[order[index]]:
            end += 1
        rank = (index + end + 1) / 2.0
        rank_sum += sum(labels[order[position]] for position in range(index, end)) * rank
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _se(value: float | None, n_positive: int, n_negative: int) -> float | None:
    if value is None or not n_positive or not n_negative:
        return None
    q1 = value / (2.0 - value)
    q2 = 2.0 * value * value / (1.0 + value)
    variance = (
        value * (1.0 - value)
        + (n_positive - 1) * (q1 - value * value)
        + (n_negative - 1) * (q2 - value * value)
    ) / (n_positive * n_negative)
    return float(np.sqrt(max(0.0, variance)))


def _metadata(path: Path) -> dict[str, float | str]:
    with Image.open(path) as image:
        return {
            "format": (image.format or "").upper(),
            "width": float(image.width),
            "height": float(image.height),
            "file_size": float(path.stat().st_size),
            "exif_present": float(bool(image.getexif())),
        }


def _features(rows: list[dict[str, Any]], feature_group: str = "all") -> tuple[list[str], np.ndarray]:
    if feature_group not in FEATURE_GROUPS:
        raise ValueError(f"unknown metadata feature group: {feature_group}")
    formats = sorted({str(row["metadata"]["format"]) for row in rows})
    names = []
    if feature_group in {"all", "format"}:
        names += [f"format={value}" for value in formats]
    if feature_group in {"all", "dimensions"}:
        names += ["width", "height"]
    if feature_group in {"all", "file_size"}:
        names += ["file_size"]
    if feature_group in {"all", "exif"}:
        names += ["exif_present"]
    matrix = np.zeros((len(rows), len(names)), dtype=float)
    for row_index, row in enumerate(rows):
        metadata = row["metadata"]
        values = []
        if feature_group in {"all", "format"}:
            values += [float(metadata["format"] == value) for value in formats]
        if feature_group in {"all", "dimensions"}:
            values += [float(metadata["width"]), float(metadata["height"])]
        if feature_group in {"all", "file_size"}:
            values += [float(metadata["file_size"])]
        if feature_group in {"all", "exif"}:
            values += [float(metadata["exif_present"])]
        matrix[row_index] = values
    return names, matrix


def _fit_stump(names: list[str], values: np.ndarray, labels: list[bool]) -> dict[str, Any]:
    """Fit the strongest one-feature threshold, using metadata only."""
    best: tuple[float, int, float, bool] | None = None
    for column, name in enumerate(names):
        candidates = sorted(set(float(value) for value in values[:, column]))
        for threshold in candidates:
            raw = values[:, column] >= threshold
            for higher in (True, False):
                scores = raw if higher else ~raw
                value = _auc(scores.astype(float).tolist(), labels)
                if value is None:
                    continue
                candidate = (value, -column, -threshold, higher)
                if best is None or candidate > best:
                    best = candidate
    if best is None:
        raise ValueError("metadata sample does not contain both labels")
    value, neg_column, neg_threshold, higher = best
    column = -neg_column
    return {
        "feature": names[column],
        "threshold": -neg_threshold,
        "higher_is_ai": higher,
        "train_auc": value,
    }


def _score(stump: dict[str, Any], names: list[str], values: np.ndarray) -> list[float]:
    column = names.index(stump["feature"])
    scores = values[:, column] >= stump["threshold"]
    if not stump["higher_is_ai"]:
        scores = ~scores
    return scores.astype(float).tolist()


def _split(rows: list[dict[str, Any]], seed: int) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row.get("source_image", row["path"])), []).append(index)
    rng = np.random.default_rng(seed)
    test: set[int] = set()
    strata: dict[tuple[bool, ...], list[str]] = {}
    for group, indices in groups.items():
        labels = tuple(sorted({bool(rows[index]["label"]) for index in indices}))
        strata.setdefault(labels, []).append(group)
    for group_names in strata.values():
        rng.shuffle(group_names)
        count = max(1, round(0.30 * len(group_names))) if len(group_names) > 1 else 0
        test.update(index for group in group_names[:count] for index in groups[group])
    if not test or len(test) == len(rows):
        ordered = [index for group in sorted(groups) for index in groups[group]]
        test.update(ordered[: max(1, len(ordered) // 3)])
    train = [index for index in range(len(rows)) if index not in test]
    if len({rows[index]["label"] for index in train}) < 2 or len({rows[index]["label"] for index in test}) < 2:
        raise ValueError("group split does not contain both labels; provide a larger sample")
    return train, sorted(test)


def _current_rows() -> list[dict[str, Any]]:
    from scripts import benchmark

    real, _manifest, _present = benchmark._real()
    rows = []
    for entry in real:
        if entry["axis"] in CURRENT_AI_AXES and entry["label"] == "ai_generated":
            rows.append({"path": entry["path"], "label": True, "axis": entry["axis"], "source_image": entry.get("source_image")})
        elif entry["axis"] == "real_camera" and entry["label"] == "authentic":
            rows.append({"path": entry["path"], "label": False, "axis": entry["axis"], "source_image": entry.get("source_image")})
    return rows


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    def label_value(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).lower() in {"1", "true", "ai", "ai_generated", "generated", "fake", "synthetic"}

    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        image = Path(item["path"])
        if not image.is_absolute():
            image = path.parent / image
        rows.append({"path": image, "label": label_value(item["label"]), "axis": item.get("axis", "unknown"), "source_image": item.get("source_image")})
    return rows


def evaluate(rows: list[dict[str, Any]], seed: int = 20260828, feature_group: str = "all") -> dict[str, Any]:
    if len(rows) < 2 or len({row["label"] for row in rows}) < 2:
        raise ValueError("need at least one authentic and one generated image")
    rows = [{**row, "metadata": _metadata(Path(row["path"]))} for row in rows]
    names, values = _features(rows, feature_group)
    labels = [bool(row["label"]) for row in rows]
    train_indices, test_indices = _split(rows, seed)
    stump = _fit_stump(names, values[train_indices], [labels[index] for index in train_indices])
    test_scores = _score(stump, names, values[test_indices])
    all_scores = _score(stump, names, values)

    def stats(scores: list[float], selected: list[int]) -> dict[str, Any]:
        selected_labels = [labels[index] for index in selected]
        value = _auc(scores, selected_labels)
        n_positive = sum(selected_labels)
        n_negative = len(selected_labels) - n_positive
        return {"auc": value, "se": _se(value, n_positive, n_negative), "n_positive": n_positive, "n_negative": n_negative}

    train_stats = stats(_score(stump, names, values[train_indices]), train_indices)
    test_stats = stats(test_scores, test_indices)
    pooled_stats = stats(all_scores, list(range(len(rows))))
    return {
        "n": len(rows),
        "axes": sorted({str(row.get("axis", "unknown")) for row in rows if row.get("label")}),
        "feature_group": feature_group,
        "features": names,
        "selected_feature": stump,
        "train": train_stats,
        "test": test_stats,
        "pooled": pooled_stats,
        "max_allowed_auc": 0.55,
        "parity_pass": (test_stats["auc"] or 0.5) <= 0.55,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=("ai",), default="ai")
    parser.add_argument("--manifest", type=Path, help="JSONL rows with path, label, and optional source_image")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--features", choices=FEATURE_GROUPS, default="all")
    parser.add_argument("--check", action="store_true", help="return failure when held-out metadata AUC exceeds 0.55")
    args = parser.parse_args()
    rows = _manifest_rows(args.manifest) if args.manifest else _current_rows()
    result = evaluate(rows, args.seed, args.features)
    axes = sorted({str(row.get("axis", "unknown")) for row in rows if row.get("label")})
    if axes:
        negatives = [row for row in rows if not row["label"]]
        result["per_axis"] = {
            axis: evaluate(
                [row for row in rows if row.get("axis", "unknown") == axis and row["label"]] + negatives,
                args.seed,
                args.features,
            )
            for axis in axes
        }
        result["parity_pass"] = result["parity_pass"] and all(
            value["parity_pass"] for value in result["per_axis"].values()
        )
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.write_text(payload)
    print(payload, end="")
    return 1 if args.check and not result["parity_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
