#!/usr/bin/env python3
"""Create a deterministic fixed-canvas JPEG variant at a byte budget."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from io import BytesIO
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MIN_QUALITY = 1
MAX_QUALITY = 100
DEFAULT_CANVAS_SIZE = 1024
DEFAULT_TOLERANCE_BYTES = 10_000


def _jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    stream = BytesIO()
    image.save(
        stream,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=False,
        subsampling=2,
        exif=b"",
    )
    return stream.getvalue()


def _canvas(path: Path, size: int) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.contain(source.convert("RGB"), (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def encode_to_budget(
    image: Image.Image,
    target_bytes: int,
    tolerance_bytes: int = DEFAULT_TOLERANCE_BYTES,
) -> tuple[bytes, int]:
    """Return exact-budget JPEG bytes and the selected integer quality."""
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    if tolerance_bytes < 0:
        raise ValueError("tolerance_bytes must be non-negative")

    cache: dict[int, bytes] = {}

    def encoded(quality: int) -> bytes:
        if quality not in cache:
            cache[quality] = _jpeg_bytes(image, quality)
        return cache[quality]

    low, high = MIN_QUALITY, MAX_QUALITY
    while low < high:
        quality = (low + high + 1) // 2
        if len(encoded(quality)) <= target_bytes:
            low = quality
        else:
            high = quality - 1

    quality = low
    data = encoded(quality)
    if len(data) > target_bytes:
        raise ValueError(f"target {target_bytes} is below the minimum JPEG size {len(data)}")
    if target_bytes - len(data) > tolerance_bytes:
        raise ValueError(
            f"target {target_bytes} is not reachable within {tolerance_bytes} bytes "
            f"(nearest quality={quality}, size={len(data)})"
        )
    # JPEG decoders ignore bytes after EOI; tail padding gives an exact budget.
    return data + b"\0" * (target_bytes - len(data)), quality


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".json", ".jsonl"}:
        text = path.read_text()
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                rows = value["images"] if isinstance(value.get("images"), list) else [value]
            else:
                rows = value
        except json.JSONDecodeError:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read a YAML manifest") from exc
        value = yaml.safe_load(path.read_text())
        rows = value.get("images", [])
    if not isinstance(rows, list):
        raise ValueError("manifest must contain an images list")
    return [dict(row) for row in rows]


def _image_path(row: dict[str, Any], manifest: Path) -> Path:
    if row.get("path"):
        path = Path(str(row["path"]))
        return path if path.is_absolute() else ROOT / path
    suffix = Path(str(row["url"]).split("?", 1)[0]).suffix.lower() or ".jpg"
    return ROOT / "data/corpus/real" / f"{row['id']}{suffix}"


def _distribution(values: list[int]) -> dict[str, Any]:
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
        "counts": {str(key): count for key, count in sorted(Counter(values).items())},
    }


def encode_manifest(
    manifest: Path,
    out: Path,
    target_bytes: int,
    axes: set[str] | None = None,
    seed: int = 20260831,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    tolerance_bytes: int = DEFAULT_TOLERANCE_BYTES,
) -> dict[str, Any]:
    """Encode selected manifest rows and write a checker-compatible JSONL sidecar."""
    if canvas_size <= 0:
        raise ValueError("canvas_size must be positive")
    rows = _manifest_rows(manifest)
    selected = [row for row in rows if axes is None or row.get("axis") in axes]
    if not selected:
        raise ValueError("no manifest rows matched the requested axes")

    ordered = sorted(selected, key=lambda row: str(row.get("id", row.get("path", ""))))
    random.Random(seed).shuffle(ordered)
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    sidecar = out / "manifest.jsonl"
    qualities: dict[str, list[int]] = {}
    sizes: list[int] = []
    with sidecar.open("w") as stream:
        for row in ordered:
            source = _image_path(row, manifest)
            image = _canvas(source, canvas_size)
            data, quality = encode_to_budget(image, target_bytes, tolerance_bytes)
            identifier = str(row.get("id", source.stem)).replace("/", "_")
            output = image_dir / f"{identifier}.jpg"
            output.write_bytes(data)
            label = str(row.get("label", "unknown"))
            qualities.setdefault(label, []).append(quality)
            sizes.append(len(data))
            item = {
                "id": identifier,
                "path": str(output.relative_to(out)),
                "label": row.get("label"),
                "axis": row.get("axis", "unknown"),
                "generator": row.get("generator"),
                "source_image": row.get("source_group", row.get("source_image", identifier)),
                "native_path": str(source),
                "native_file_size": source.stat().st_size,
                "parity_quality": quality,
                "parity_file_size": len(data),
                "target_bytes": target_bytes,
            }
            stream.write(json.dumps(item, sort_keys=True) + "\n")

    summary = {
        "seed": seed,
        "n": len(ordered),
        "canvas": [canvas_size, canvas_size],
        "target_bytes": target_bytes,
        "tolerance_bytes": tolerance_bytes,
        "size": {"min": min(sizes), "max": max(sizes), "mean": mean(sizes), "median": median(sizes)},
        "quality_distribution": {label: _distribution(values) for label, values in sorted(qualities.items())},
        "manifest": str(sidecar),
    }
    (out / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-bytes", type=int, required=True)
    parser.add_argument("--axes", help="comma-separated axes to encode")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--canvas-size", type=int, default=DEFAULT_CANVAS_SIZE)
    parser.add_argument("--tolerance-bytes", type=int, default=DEFAULT_TOLERANCE_BYTES)
    args = parser.parse_args()
    axes = {value.strip() for value in args.axes.split(",") if value.strip()} if args.axes else None
    summary = encode_manifest(
        args.manifest,
        args.out,
        args.target_bytes,
        axes,
        args.seed,
        args.canvas_size,
        args.tolerance_bytes,
    )
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
