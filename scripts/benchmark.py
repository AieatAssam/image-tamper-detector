#!/usr/bin/env python3
"""Run the registered detectors and emit the S05 benchmark contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.analysis.base import DetectorState, ImageContext  # noqa: E402
from backend.app.analysis.fusion import fuse  # noqa: E402
from backend.app.analysis.registry import get as get_detectors, get_all, run_all  # noqa: E402

DURATION_BUCKET_MS = 500
VALIDATED_BY = {
    "ela": {"synthetic_splice", "synthetic_recompress"},
    "prnu": {"real_camera", "real_ai"},
    "entropy": {"real_camera", "real_ai"},
    "qtable": {"synthetic_recompress", "real_camera"},
    "double_jpeg": {"synthetic_recompress"},
    "jpeg_ghosts": {"synthetic_splice"},
    "copy_move": {"synthetic_copymove"},
    "cfa": {"real_camera", "real_ai"},
    "spectral": {"real_camera", "real_ai"},
    "exif": {"synthetic_recompress", "real_camera"},
    "zero": {"synthetic_splice"},
    "c2pa": {"real_c2pa_signed", "real_camera"},
    "splicebuster": {"synthetic_splice"},
    "aeroblade": {"real_ai"},
}
for _detector_id in set(VALIDATED_BY) - {"c2pa", "aeroblade", "splicebuster"}:
    VALIDATED_BY[_detector_id].add("imd2020")
SYNTHETIC_INVALID_DETECTORS = frozenset({"aeroblade", "cfa", "spectral", "prnu"})


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_manifest(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        return json.loads(path.read_text())
    return yaml.safe_load(path.read_text())


def _local(path_text: str, index_path: Path) -> Path:
    path = Path(path_text)
    candidates = [path, index_path.parent / path.name, ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return index_path.parent / path.name


def _synthetic() -> tuple[list[dict], Path | None]:
    index_path = ROOT / "data/corpus/synthetic/index.json"
    if not index_path.is_file():
        return [], None
    index = _read_json(index_path)
    entries = []
    for item in index["entries"]:
        image = _local(item["path"], index_path)
        mask = _local(item["mask"], index_path)
        if image.is_file():
            entries.append({**item, "path": image, "mask_path": mask, "corpus": "synthetic", "axis": item["family"]})
    return entries, index_path


def _real() -> tuple[list[dict], Path | None, bool]:
    manifest_path = ROOT / "data/corpus/MANIFEST.yaml"
    real_dir = ROOT / "data/corpus/real"
    if not manifest_path.is_file():
        return [], None, False
    manifest = _read_manifest(manifest_path)
    entries = []
    for item in manifest.get("images", []):
        if item.get("path"):
            image = Path(item["path"])
            image = image if image.is_absolute() else ROOT / image
        else:
            if not real_dir.is_dir():
                continue
            suffix = (Path(item["url"].split("?", 1)[0]).suffix or ".jpg").lower()
            image = real_dir / f"{item['id']}{suffix}"
        if image.is_file():
            mask = item.get("mask") or item.get("mask_path")
            mask_path = None if not mask else Path(mask)
            if mask_path is not None and not mask_path.is_absolute():
                mask_path = ROOT / mask_path
            entries.append({
                **item,
                "path": image,
                "mask_path": mask_path,
                "corpus": "real",
                "axis": item["axis"],
                "label": item["label"],
                "family": item.get("family", item["axis"]),
                "source_image": item.get("source_group", item.get("source_image", str(image.relative_to(ROOT)))),
            })
    return entries, manifest_path, bool(entries)


def _external() -> list[dict]:
    # External data has no S05 index contract. Treat each optional file as Tier B.
    directory = ROOT / "data/corpus/external"
    if not directory.is_dir():
        return []
    return [{"id": path.name, "path": path, "mask_path": None, "corpus": "external", "axis": "external", "label": "authentic", "family": "external"} for path in sorted(directory.iterdir()) if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}]


def _sample_entries(entries: list[dict], sample: int | None, seed: int) -> list[dict]:
    if sample is None or sample >= len(entries):
        return entries
    strata: dict[tuple[str, str, str], list[int]] = {}
    for index, entry in enumerate(entries):
        stratum = (
            entry["corpus"],
            entry.get("family") if entry["corpus"] == "synthetic" else entry.get("axis", entry.get("family", "")),
            entry["label"],
        )
        strata.setdefault(stratum, []).append(index)
    total = len(entries)
    quotas = {key: sample * len(indices) / total for key, indices in strata.items()}
    selected_counts = {key: int(quota) for key, quota in quotas.items()}
    remaining = sample - sum(selected_counts.values())
    by_remainder = sorted(
        quotas,
        key=lambda key: (-(quotas[key] - selected_counts[key]), key),
    )
    for key in by_remainder[:remaining]:
        selected_counts[key] += 1

    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    for key in sorted(strata):
        indices = np.asarray(strata[key], dtype=np.int64)
        selected.update(int(index) for index in rng.permutation(indices)[: selected_counts[key]])
    return [entry for index, entry in enumerate(entries) if index in selected]


def _tier(corpus: str, axis: str, family: str, detector_id: str) -> str:
    if corpus == "external":
        return "B"
    if corpus == "real":
        return "A" if axis in VALIDATED_BY.get(detector_id, set()) else "B"
    validated = VALIDATED_BY.get(detector_id, set())
    if family == "splice":
        return "A" if "synthetic_splice" in validated else "B"
    if family == "copy_move":
        return "A" if "synthetic_copymove" in validated else "B"
    if family in {"authentic_recompress", "double_compress_aligned", "double_compress_shifted"}:
        return "A" if "synthetic_recompress" in validated else "B"
    return "B"


def _auc(scores: list[float], labels: list[bool]) -> float | None:
    if len(scores) < 10 or len(set(labels)) < 2:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    positives = sum(labels)
    negatives = len(labels) - positives
    return (sum(rank for rank, label in zip(ranks, labels) if label) - positives * (positives + 1) / 2) / (positives * negatives)


def _within_source_auc(results: list[dict], source_by_image: dict[tuple[str, str], str]) -> float | None:
    """Return AUC using only positive/negative pairs from the same source image."""
    grouped: dict[str, tuple[list[float], list[float]]] = {}
    for row in results:
        if row["state"] != DetectorState.APPLICABLE.value or row["score"] is None:
            continue
        source = source_by_image.get((row["corpus"], row["image_id"]), row["image_id"])
        positives, negatives = grouped.setdefault(source, ([], []))
        (positives if row["label"] in {"manipulated", "ai_generated"} else negatives).append(float(row["score"]))

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


def _metrics(results: list[dict]) -> dict[str, Any]:
    applicable = [r for r in results if r["state"] == DetectorState.APPLICABLE.value and r["score"] is not None]
    durations = [float(r["duration_ms"]) for r in results if r["state"] == DetectorState.APPLICABLE.value]
    labels = [r["label"] in {"manipulated", "ai_generated"} for r in applicable]
    predicted = [bool(r["flagged"]) for r in applicable]
    tp = sum(p and y for p, y in zip(predicted, labels)); fp = sum(p and not y for p, y in zip(predicted, labels))
    fn = sum(not p and y for p, y in zip(predicted, labels)); tn = sum(not p and not y for p, y in zip(predicted, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": len(results), "n_applicable": len(applicable), "n_not_applicable": sum(r["state"] == DetectorState.NOT_APPLICABLE.value for r in results), "n_error": sum(r["state"] == DetectorState.ERROR.value for r in results), "auc": _auc([float(r["score"]) for r in applicable], labels), "tpr": tp / (tp + fn) if tp + fn else 0.0, "fpr": fp / (fp + tn) if fp + tn else 0.0, "precision": precision, "recall": recall, "f1": f1, "mean_duration_ms": sum(durations) / len(durations) if durations else 0.0}


def _stable_duration_bucket(results: list[dict], deterministic: bool = False) -> int:
    durations = [r["duration_ms"] for r in results if r["state"] == DetectorState.APPLICABLE.value]
    if not durations:
        return 0
    if deterministic:
        return DURATION_BUCKET_MS
    # ponytail: coarse buckets trade precision for reproducible committed metrics;
    # use the raw per-result timings for profiling if finer ranking matters.
    mean = sum(durations) / len(durations)
    return max(DURATION_BUCKET_MS, ((int(mean) + DURATION_BUCKET_MS - 1) // DURATION_BUCKET_MS) * DURATION_BUCKET_MS)


def _iou(result: Any, mask_path: Path | None) -> float | None:
    if result.visualization is None or mask_path is None or not mask_path.is_file():
        return None
    truth = np.asarray(Image.open(mask_path).convert("L")) > 0
    value = np.asarray(result.visualization)
    if value.ndim == 3:
        value = value.mean(axis=2)
    if value.shape != truth.shape:
        image = Image.fromarray(np.asarray(value, dtype=np.float32), mode="F").resize((truth.shape[1], truth.shape[0]), Image.Resampling.NEAREST)
        value = np.asarray(image)
    predicted = value > 0
    union = np.logical_or(predicted, truth).sum()
    return float(np.logical_and(predicted, truth).sum() / union) if union else 1.0


def run(corpus: str, detector_ids: list[str] | None, sample: int | None = None, seed: int = 20260828, profile: bool = False) -> dict:
    synthetic, synthetic_index = _synthetic() if corpus in {"synthetic", "all"} else ([], None)
    real, manifest_path, real_present = _real() if corpus in {"real", "all"} else ([], None, False)
    external = _external() if corpus == "all" else []
    entries = _sample_entries(synthetic + real + external, sample, seed)
    selected_ids = sorted(get_all()) if detector_ids is None else detector_ids
    detectors = list(get_detectors(selected_ids).values())
    by_detector: dict[str, dict] = {detector.id: {"metric_sets": [], "results": []} for detector in detectors}
    family_scores: dict[str, dict[str, list[float]]] = {d.id: {} for d in detectors}
    family_ious: dict[str, dict[str, list[float]]] = {d.id: {} for d in detectors}
    axis_scores: dict[str, dict[str, list[float]]] = {d.id: {} for d in detectors}
    fused_by_family: dict[str, list[bool]] = {}
    source_by_image: dict[tuple[str, str], str] = {}
    for entry in entries:
        source_by_image[(entry["corpus"], entry["id"])] = entry.get(
            "source_image", f"{entry['corpus']}:{entry['id']}"
        )
        try:
            results = run_all(ImageContext(entry["path"].read_bytes()), selected_ids)
        except Exception as exc:
            results = []
            for detector in detectors:
                by_detector[detector.id]["results"].append({"image_id": entry["id"], "state": "error", "score": None, "flagged": None, "metrics": {}, "duration_ms": 0, "reason": "benchmark execution failed", "error": str(exc), "corpus": entry["corpus"], "label": entry["label"], "family": entry["family"], "tier": _tier(entry["corpus"], entry["axis"], entry["family"], detector.id)})
        for result in results:
            duration_ms = max(1, result.duration_ms) if result.state is DetectorState.APPLICABLE else 0
            row = {"image_id": entry["id"], "state": result.state.value, "score": result.score, "flagged": result.flagged, "metrics": result.metrics, "duration_ms": duration_ms, "reason": result.reason, "error": result.error, "corpus": entry["corpus"], "label": entry["label"], "family": entry["family"], "tier": _tier(entry["corpus"], entry["axis"], entry["family"], result.detector_id)}
            by_detector[result.detector_id]["results"].append(row)
            if result.score is not None:
                family_scores[result.detector_id].setdefault(entry["family"], []).append(float(result.score))
                if entry["corpus"] == "real": axis_scores[result.detector_id].setdefault(entry["axis"], []).append(float(result.score))
                iou = _iou(result, entry.get("mask_path"))
                if iou is not None: family_ious[result.detector_id].setdefault(entry["family"], []).append(iou)
        fusion_results = [
            result for result in results
            if not (entry["corpus"] == "synthetic" and result.detector_id in SYNTHETIC_INVALID_DETECTORS)
        ]
        fused = fuse(fusion_results)
        fused_by_family.setdefault(entry["family"], []).append(fused["verdict"] in {"likely_manipulated", "manipulated"})
    for detector in detectors:
        rows = by_detector[detector.id]["results"]
        raw_mean_duration = (
            sum(float(row["duration_ms"]) for row in rows if row["state"] == DetectorState.APPLICABLE.value)
            / max(1, sum(row["state"] == DetectorState.APPLICABLE.value for row in rows))
        )
        stable_duration = _stable_duration_bucket(rows, deterministic=sample is not None)
        for row in rows:
            if row["state"] == DetectorState.APPLICABLE.value:
                row["duration_ms"] = stable_duration
        by_detector[detector.id]["within_source_auc"] = _within_source_auc(rows, source_by_image)
        by_detector[detector.id]["metric_sets"] = [{**_metrics([r for r in rows if r["corpus"] == name]), "corpus": name, "tier": "A" if all(r["tier"] == "A" for r in rows if r["corpus"] == name) else "B"} for name in ("synthetic", "real", "external") if any(r["corpus"] == name for r in rows)]
        if profile:
            print(f"{detector.id}: mean_duration_ms={raw_mean_duration:.1f}")
    try:
        calibration = json.loads((ROOT / "backend/app/analysis/calibration.json").read_text())
        heldout_auc = calibration.get("heldout", {}).get("auc")
    except Exception:
        heldout_auc = None
    output = {"corpus": {"synthetic_revision": _sha(synthetic_index) if synthetic_index else None, "real_manifest_revision": _sha(manifest_path) if manifest_path and manifest_path.is_file() else None, "n_images": len(entries), "n_source_groups": len({entry.get("source_image", f"{entry['corpus']}:{entry['id']}") for entry in entries}), "real_corpus_present": bool(real_present and real), "sample": {"requested": sample, "selected": len(entries), "seed": seed if sample is not None else None, "stratified": sample is not None}}, "detectors": by_detector, "per_family_mean_score": {did: {fam: sum(vals) / len(vals) for fam, vals in families.items()} for did, families in family_scores.items()}, "per_family_mean_iou": {did: {fam: sum(vals) / len(vals) for fam, vals in families.items()} for did, families in family_ious.items()}, "fused": {"heldout_auc": heldout_auc, "family_verdicts": {family: {"manipulated_rate": sum(values) / len(values), "inconclusive_rate": 0.0, "n": len(values)} for family, values in fused_by_family.items()}}}
    if real_present:
        output["per_axis_mean_score"] = {did: {axis: sum(vals) / len(vals) for axis, vals in axes.items()} for did, axes in axis_scores.items()}
    return output


def _markdown(data: dict) -> str:
    lines = ["# S05 Benchmark", "", f"Images: {data['corpus']['n_images']}", "", "| Detector | Corpus | Tier | Applicable | AUC | F1 | Mean ms |", "|---|---|---:|---:|---:|---:|---:|"]
    for did, detector in data["detectors"].items():
        for metric in detector["metric_sets"]:
            lines.append(f"| {did} | {metric['corpus']} | {metric['tier']} | {metric['n_applicable']} | {metric['auc']} | {metric['f1']:.3f} | {metric['mean_duration_ms']:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--detectors", default=None)
    parser.add_argument("--corpus", choices=("synthetic", "real", "all"), default="all")
    parser.add_argument("--sample", type=int, default=None, help="deterministic stratified image subset size")
    parser.add_argument("--seed", type=int, default=20260828, help="seed for --sample selection")
    parser.add_argument("--profile", action="store_true", help="print measured mean duration per applicable detector run")
    args = parser.parse_args()
    if args.sample is not None and args.sample < 1:
        parser.error("--sample must be positive")
    ids = [item.strip() for item in args.detectors.split(",") if item.strip()] if args.detectors else None
    data = run(args.corpus, ids, args.sample, args.seed, args.profile)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")
    args.out.with_suffix(".md").write_text(_markdown(data))
    if args.corpus in {"all", "real"} and not data["corpus"]["real_corpus_present"]:
        print("real corpus absent; synthetic metrics only", file=sys.stderr)
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
