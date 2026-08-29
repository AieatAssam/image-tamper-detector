#!/usr/bin/env python3
"""Capture raw analyzer outputs for the checked-in sample images."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import scipy
import skimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.analysis.c2pa import C2PAAnalyzer  # noqa: E402
from backend.app.analysis.ela import ELAAnalyzer  # noqa: E402
from backend.app.analysis.entropy import EntropyAnalyzer  # noqa: E402
from backend.app.analysis.prnu import PRNUAnalyzer  # noqa: E402


def _float(value: float) -> float:
    return float(f"{float(value):.9g}")


def _sha_array(value: object) -> str | None:
    if not isinstance(value, np.ndarray):
        return None
    return hashlib.sha256(value.tobytes()).hexdigest()


def _image(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    with Image.open(path) as image:
        width, height, image_format = image.width, image.height, image.format
    row: dict[str, object] = {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "width": width,
        "height": height,
        "format": image_format,
    }
    try:
        flagged, visual, features = ELAAnalyzer().detect_tampering(path)
        row["ela"] = {
            "is_tampered": flagged,
            "edge_discontinuity": _float(features.edge_discontinuity),
            "texture_variance": _float(features.texture_variance),
            "noise_consistency": _float(features.noise_consistency),
            "compression_artifacts": _float(features.compression_artifacts),
            "visualization_sha256": _sha_array(visual),
        }
    except Exception as exc:
        row["ela"] = {"error": repr(exc)}
    try:
        flagged, visual, score = PRNUAnalyzer(variance_threshold=300).detect_tampering(path)
        row["prnu"] = {"is_tampered": flagged, "uniformity_score": _float(score), "visualization_sha256": _sha_array(visual)}
    except Exception as exc:
        row["prnu"] = {"error": repr(exc)}
    try:
        flagged, visual, score = EntropyAnalyzer().detect_ai_generated(path)
        row["entropy"] = {"is_ai_generated": flagged, "matching_proportion": _float(score), "visualization_sha256": _sha_array(visual)}
    except Exception as exc:
        row["entropy"] = {"error": repr(exc)}
    try:
        row["c2pa"] = C2PAAnalyzer().analyze_image(path)
    except Exception as exc:
        row["c2pa"] = {"error": repr(exc)}
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(
        path for path in (ROOT / "data/samples").rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    output = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cv2": cv2.__version__,
            "skimage": skimage.__version__,
            "PIL": Image.__version__,
            "scipy": scipy.__version__,
        },
        "images": [_image(path) for path in paths],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
