#!/usr/bin/env python3
"""Check whether blurred thumbnails predict corpus labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_format_shortcut import _auc, _se, _split  # noqa: E402

CURRENT_AI_AXES = frozenset({"real_ai", "sd35_flux", "synthbuster"})
POSITIVE_LABELS = frozenset({"ai_generated", "manipulated", "generated", "fake", "synthetic"})
NEGATIVE_LABELS = frozenset({"authentic", "real"})
DEFAULT_THUMBNAIL_SIZE = 32
BLUR_RADIUS = 1.0
MAX_ALLOWED_AUC = 0.55


def _thumbnail(path: Path, size: int = DEFAULT_THUMBNAIL_SIZE) -> np.ndarray:
    """Return a low-resolution, blurred RGB thumbnail in [0, 1]."""
    if size not in (16, 32):
        raise ValueError("thumbnail size must be 16 or 32")
    with Image.open(path) as image:
        thumbnail = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    thumbnail = thumbnail.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
    return np.asarray(thumbnail, dtype=np.float32) / 255.0


def _fit_centroid(values: np.ndarray, labels: list[bool]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a standardized nearest-centroid classifier on the training rows."""
    label_array = np.asarray(labels, dtype=bool)
    if label_array.all() or (~label_array).all():
        raise ValueError("content sample does not contain both labels")
    positive = values[label_array].mean(axis=0)
    negative = values[~label_array].mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return positive, negative, scale


def _score(model: tuple[np.ndarray, np.ndarray, np.ndarray], values: np.ndarray) -> list[float]:
    positive, negative, scale = model
    positive_distance = np.mean(((values - positive) / scale) ** 2, axis=1)
    negative_distance = np.mean(((values - negative) / scale) ** 2, axis=1)
    return (negative_distance - positive_distance).astype(float).tolist()


def _shortcut_auc(value: float | None) -> float | None:
    """Treat either classifier orientation as evidence of predictability."""
    return None if value is None else max(value, 1.0 - value)


def _label_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).lower() in {"1", "true", "ai", "ai_generated", "generated", "fake", "synthetic"}


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        image = Path(item["path"])
        if not image.is_absolute():
            image = path.parent / image
        rows.append({
            "path": image,
            "label": _label_value(item["label"]),
            "axis": item.get("axis", "unknown"),
            "source_image": item.get("source_image") or str(image),
        })
    return rows


def _parse_axes(value: str | None) -> set[str] | None:
    if value is None:
        return None
    axes = {axis.strip() for axis in value.split(",") if axis.strip()}
    if not axes:
        raise ValueError("--axes requires at least one axis")
    return axes


def _select_axes(rows: list[dict[str, Any]], axes: set[str] | None) -> list[dict[str, Any]]:
    if axes is None:
        return rows
    selected = set(axes)
    if selected - {"real_camera"}:
        selected.add("real_camera")
    return [row for row in rows if str(row.get("axis", "unknown")) in selected]


def _current_rows(axes: set[str] | None = None) -> list[dict[str, Any]]:
    from scripts import benchmark

    requested = CURRENT_AI_AXES if axes is None else axes
    selected = set(requested)
    if selected - {"real_camera"}:
        selected.add("real_camera")
    real, _manifest, _present = benchmark._real()
    rows = []
    for entry in real:
        axis = str(entry["axis"])
        if axis not in selected:
            continue
        source_label = str(entry["label"]).lower()
        if source_label in POSITIVE_LABELS:
            label = True
        elif source_label in NEGATIVE_LABELS:
            label = False
        else:
            continue
        path = Path(entry["path"])
        rows.append({
            "path": path,
            "label": label,
            "axis": axis,
            "source_image": entry.get("source_image") or str(path),
        })
    return rows


def _stats(scores: list[float], labels: list[bool]) -> dict[str, Any]:
    value = _auc(scores, labels)
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    return {
        "auc": value,
        "se": _se(value, n_positive, n_negative),
        "shortcut_auc": _shortcut_auc(value),
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def evaluate(rows: list[dict[str, Any]], seed: int = 20260828, size: int = DEFAULT_THUMBNAIL_SIZE) -> dict[str, Any]:
    if len(rows) < 2 or len({bool(row["label"]) for row in rows}) < 2:
        raise ValueError("need at least one authentic and one generated image")
    rows = [{**row, "path": Path(row["path"]), "source_image": row.get("source_image") or str(row["path"])} for row in rows]
    values = np.stack([_thumbnail(row["path"], size).reshape(-1) for row in rows])
    labels = [bool(row["label"]) for row in rows]
    train_indices, test_indices = _split(rows, seed)
    model = _fit_centroid(values[train_indices], [labels[index] for index in train_indices])

    def selected_stats(indices: list[int]) -> dict[str, Any]:
        scores = _score(model, values[indices])
        return _stats(scores, [labels[index] for index in indices])

    train_stats = selected_stats(train_indices)
    test_stats = selected_stats(test_indices)
    pooled_stats = selected_stats(list(range(len(rows))))
    return {
        "n": len(rows),
        "axes": sorted({str(row.get("axis", "unknown")) for row in rows}),
        "classifier": "standardized nearest centroid",
        "thumbnail_size": size,
        "blur_radius": BLUR_RADIUS,
        "seed": seed,
        "split": {"train_n": len(train_indices), "test_n": len(test_indices), "test_fraction": 0.30},
        "train": train_stats,
        "test": test_stats,
        "pooled": pooled_stats,
        "max_allowed_auc": MAX_ALLOWED_AUC,
        "content_pass": (test_stats["shortcut_auc"] or 0.5) <= MAX_ALLOWED_AUC,
    }


def _per_axis(rows: list[dict[str, Any]], seed: int, size: int) -> dict[str, Any]:
    negatives = [row for row in rows if not row["label"]]
    axes = sorted({str(row.get("axis", "unknown")) for row in rows if row["label"]})
    return {
        axis: evaluate(
            [row for row in rows if str(row.get("axis", "unknown")) == axis and row["label"]] + negatives,
            seed,
            size,
        )
        for axis in axes
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, help="JSONL rows with path, label, and optional axis/source_image")
    parser.add_argument("--axes", help="comma-separated axes; real_camera is added as the negative class")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--size", type=int, choices=(16, 32), default=DEFAULT_THUMBNAIL_SIZE)
    parser.add_argument("--check", action="store_true", help="return failure when held-out content AUC exceeds 0.55")
    args = parser.parse_args()
    axes = _parse_axes(args.axes)
    rows = _manifest_rows(args.manifest) if args.manifest else _current_rows(axes)
    if args.manifest:
        rows = _select_axes(rows, axes)
    result = evaluate(rows, args.seed, args.size)
    result["per_axis"] = _per_axis(rows, args.seed, args.size)
    result["content_pass"] = result["content_pass"] and all(item["content_pass"] for item in result["per_axis"].values())
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.write_text(payload)
    print(payload, end="")
    return 1 if args.check and not result["content_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
