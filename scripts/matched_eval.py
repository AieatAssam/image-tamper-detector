#!/usr/bin/env python3
"""Build a deterministic, metadata-matched AI versus authentic subset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 20260828
DEFAULT_TOLERANCE = 4.0
AI_AXES = frozenset({"sd35_flux", "synthbuster"})
REAL_AXES = frozenset({"imd2020", "real_camera"})


def _metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        return {
            "format": (image.format or "").upper(),
            "width": image.width,
            "height": image.height,
            "file_size": path.stat().st_size,
            "exif_present": bool(image.getexif()),
        }


def _distance(ai: dict[str, Any], real: dict[str, Any]) -> float | None:
    """Return a weighted log-distance, or None for hard metadata mismatches.

    Format and EXIF presence are exact constraints. File-size distance is
    weighted four times, dimensions once, and aspect ratio twice so the
    proven file-size shortcut dominates nearest-neighbour selection.
    """
    if ai["format"] != real["format"] or ai["exif_present"] != real["exif_present"]:
        return None
    size = abs(math.log2(ai["file_size"] / real["file_size"]))
    dimensions = max(
        abs(math.log2(ai["width"] / real["width"])),
        abs(math.log2(ai["height"] / real["height"])),
    )
    ai_aspect = ai["width"] / ai["height"]
    real_aspect = real["width"] / real["height"]
    aspect = abs(math.log2(ai_aspect / real_aspect))
    return 4.0 * size + dimensions + 2.0 * aspect


def _entries() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from scripts import benchmark

    entries, _manifest, _present = benchmark._real()
    ai = [
        entry
        for entry in entries
        if entry["axis"] in AI_AXES and entry["label"] == "ai_generated"
    ]
    real = [
        entry
        for entry in entries
        if entry["axis"] in REAL_AXES and entry["label"] == "authentic"
    ]
    return ai, real


def match_entries(
    ai_entries: list[dict[str, Any]],
    real_entries: list[dict[str, Any]],
    seed: int = SEED,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Greedily match shuffled AI rows to unused nearest real rows."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    ai_rows = [
        {**entry, "metadata": _metadata(Path(entry["path"]))}
        for entry in sorted(ai_entries, key=lambda entry: entry["id"])
    ]
    real_rows = [
        {**entry, "metadata": _metadata(Path(entry["path"]))}
        for entry in sorted(real_entries, key=lambda entry: entry["id"])
    ]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ai_rows))
    unused = {entry["id"] for entry in real_rows}
    pairs: list[dict[str, Any]] = []
    for index in order:
        ai = ai_rows[int(index)]
        candidates = []
        for real in real_rows:
            if real["id"] not in unused:
                continue
            distance = _distance(ai["metadata"], real["metadata"])
            if distance is not None:
                candidates.append((distance, real["id"], real))
        if not candidates:
            continue
        distance, _real_id, real = min(candidates, key=lambda item: (item[0], item[1]))
        if distance > tolerance:
            continue
        unused.remove(real["id"])
        pair_id = f"matched_{len(pairs):04d}_{ai['id']}_{real['id']}"
        pairs.append(
            {
                "pair_id": pair_id,
                "distance": distance,
                "ai": ai,
                "real": real,
            }
        )
    pairs.sort(key=lambda pair: pair["pair_id"])
    return {
        "seed": seed,
        "tolerance": tolerance,
        "distance": "4*abs(log2(file_size_ratio)) + max(abs(log2(width_ratio)), abs(log2(height_ratio))) + 2*abs(log2(aspect_ratio_ratio)); exact format and EXIF presence",
        "ai_axes": sorted(AI_AXES),
        "real_axes": sorted(REAL_AXES),
        "candidate_ai": len(ai_rows),
        "candidate_real": len(real_rows),
        "n_pairs": len(pairs),
        "pairs": pairs,
    }


def _row(pair: dict[str, Any], entry: dict[str, Any], label: str) -> dict[str, Any]:
    return {
        "id": entry["id"],
        "path": str(entry["path"]),
        "label": label,
        "axis": entry["axis"],
        "family": entry.get("family", entry["axis"]),
        "generator": entry.get("generator"),
        "source_image": pair["pair_id"],
        "matched_pair_id": pair["pair_id"],
        "matched_distance": pair["distance"],
    }


def rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for pair in result["pairs"]:
        output.append(_row(pair, pair["ai"], "ai_generated"))
        output.append(_row(pair, pair["real"], "authentic"))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="matched JSONL output")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()
    result = match_entries(*_entries(), args.seed, args.tolerance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows(result)))
    summary = {key: value for key, value in result.items() if key != "pairs"}
    print(json.dumps(summary, sort_keys=True, indent=2))
    print(f"wrote {len(rows(result))} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
