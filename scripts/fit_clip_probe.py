#!/usr/bin/env python3
"""Fit a frozen CLIP linear probe with source- and generator-held-out splits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.analysis.clip_probe import MODEL_NAME, MODEL_REPO, _load_backbone  # noqa: E402
from scripts.calibrate import auc, fit_standardized_logistic, hanley_mcneil_se  # noqa: E402


AI_AXES = frozenset({"sd35_flux", "synthbuster"})
DEFAULT_BACKBONE_PATH = ROOT / "models/clip/open_clip_pytorch_model.safetensors"
DEFAULT_PROBE_PATH = ROOT / "models/clip/linear_probe.npz"


def _rows() -> list[dict]:
    manifest = yaml.safe_load((ROOT / "data/corpus/MANIFEST.yaml").read_text())
    rows = []
    real_dir = ROOT / "data/corpus/real"
    for item in manifest.get("images", []):
        if item["axis"] not in AI_AXES and item["axis"] != "real_camera":
            continue
        path = ROOT / item["path"] if item.get("path") else real_dir / f"{item['id']}{Path(item['url'].split('?', 1)[0]).suffix}"
        if path.is_file():
            rows.append({
                "path": path,
                "axis": item["axis"],
                "generator": item.get("generator"),
                "label": item["label"] != "authentic",
                "source_image": item.get("source_group", item.get("source_image", str(path.relative_to(ROOT)))),
            })
    return rows


def _split(rows: list[dict], seed: int, holdout_count: int) -> tuple[set[str], set[str], set[str]]:
    generators = sorted({row["generator"] for row in rows if row["label"]})
    if not 0 < holdout_count < len(generators):
        raise ValueError(f"holdout_generators must be between 1 and {len(generators) - 1}")
    rng = np.random.default_rng(seed)
    heldout = {generators[index] for index in rng.permutation(len(generators))[:holdout_count]}
    groups = sorted({row["source_image"] for row in rows})
    camera_groups = sorted({row["source_image"] for row in rows if not row["label"]})
    other_groups = sorted(set(groups) - set(camera_groups))
    test_count = max(1, math.ceil(len(groups) * 0.3))
    camera_test_count = max(1, math.ceil(len(camera_groups) * 0.3))
    test_groups = set(rng.permutation(camera_groups)[:camera_test_count])
    remaining = test_count - len(test_groups)
    test_groups.update(rng.permutation(other_groups)[:remaining])
    train_groups = set(groups) - test_groups
    if not test_groups:
        raise ValueError("source-image split produced no test groups")
    return heldout, train_groups, test_groups


def _scores(features: np.ndarray, intercept: float, weights: np.ndarray) -> np.ndarray:
    logits = np.clip(intercept + features @ weights, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _metric(features: np.ndarray, labels: np.ndarray, indices: list[int], intercept: float, weights: np.ndarray) -> dict:
    if not indices:
        return {"auc": None, "se": None, "n_positive": 0, "n_negative": 0}
    scores = _scores(features[indices], intercept, weights)
    labels = labels[indices]
    value = auc(scores.tolist(), labels.astype(bool).tolist())
    n_positive = int(labels.sum())
    n_negative = int(len(labels) - n_positive)
    return {
        "auc": value,
        "se": hanley_mcneil_se(value, n_positive, n_negative),
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def _generator_metrics(rows: list[dict], features: np.ndarray, labels: np.ndarray, indices: list[int], intercept: float, weights: np.ndarray, split: str) -> dict:
    output = {}
    for generator in sorted({rows[index]["generator"] for index in indices if rows[index]["label"]}):
        generator_indices = [index for index in indices if rows[index]["generator"] == generator or (not rows[index]["label"])]
        output[generator] = {"split": split, **_metric(features, labels, generator_indices, intercept, weights)}
    return output


def _features(rows: list[dict], backbone_path: Path) -> np.ndarray:
    torch, model, preprocess = _load_backbone(str(backbone_path))
    values = []
    for start in range(0, len(rows), 16):
        tensors = []
        for row in rows[start:start + 16]:
            with Image.open(row["path"]) as image:
                tensors.append(preprocess(image.convert("RGB")))
        batch = torch.stack(tensors)
        with torch.inference_mode():
            feature = model.encode_image(batch)
        values.append(feature.detach().float().cpu().numpy())
    return np.concatenate(values, axis=0).astype(np.float32)


def _indices(rows: list[dict], train_groups: set[str], test_groups: set[str], heldout: set[str]) -> tuple[list[int], list[int], list[int]]:
    train = [
        index for index, row in enumerate(rows)
        if row["source_image"] in train_groups
        and ((row["label"] and row["generator"] not in heldout) or not row["label"])
    ]
    in_distribution = [
        index for index, row in enumerate(rows)
        if row["source_image"] in test_groups
        and ((row["label"] and row["generator"] not in heldout) or not row["label"])
    ]
    out_of_distribution = [
        index for index, row in enumerate(rows)
        if row["source_image"] in test_groups
        and ((row["label"] and row["generator"] in heldout) or not row["label"])
    ]
    if not train or not in_distribution or not out_of_distribution:
        raise ValueError("generator/source-image split produced an empty partition")
    return train, in_distribution, out_of_distribution


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-path", type=Path, default=DEFAULT_BACKBONE_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_PROBE_PATH)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--holdout-generators", type=int, default=4)
    args = parser.parse_args()

    rows = _rows()
    if not args.backbone_path.is_file():
        raise SystemExit(f"BLOCKED: CLIP backbone weights are missing: {args.backbone_path}")
    heldout, train_groups, test_groups = _split(rows, args.seed, args.holdout_generators)
    features = _features(rows, args.backbone_path)
    labels = np.asarray([row["label"] for row in rows], dtype=bool)
    train_indices, id_indices, ood_indices = _indices(rows, train_groups, test_groups, heldout)
    intercept, weights = fit_standardized_logistic(features[train_indices], labels[train_indices])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, weight=weights.astype(np.float32), bias=np.asarray([intercept], dtype=np.float32), model_name=MODEL_NAME, model_repo=MODEL_REPO, seed=args.seed, heldout_generators=np.asarray(sorted(heldout)))

    per_generator = {}
    per_generator.update(_generator_metrics(rows, features, labels, id_indices, intercept, weights, "in_distribution"))
    per_generator.update(_generator_metrics(rows, features, labels, ood_indices, intercept, weights, "out_of_distribution"))
    report = {
        "model": {"name": MODEL_NAME, "repo": MODEL_REPO, "backbone_frozen": True, "probe": "linear"},
        "seed": args.seed,
        "heldout_generators": sorted(heldout),
        "train_generators": sorted({row["generator"] for row in rows if row["label"] and row["generator"] not in heldout}),
        "split": {"group_key": "source_image", "train_groups": len(train_groups), "test_groups": len(test_groups)},
        "in_distribution": _metric(features, labels, id_indices, intercept, weights),
        "out_of_distribution": _metric(features, labels, ood_indices, intercept, weights),
        "per_generator": per_generator,
        "n_rows": len(rows),
        "n_train": len(train_indices),
        "n_in_distribution": len(id_indices),
        "n_out_of_distribution": len(ood_indices),
    }
    report_path = args.out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
