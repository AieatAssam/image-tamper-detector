#!/usr/bin/env python3
"""Fetch pinned optional model artifacts and verify their checksums/configs."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]

FACE_REPO = "onnx-community/Deep-Fake-Detector-v2-Model-ONNX"
FACE_REVISION = "4ea3d66dfb1bedca29727c6a0c6fa061d5f3f9c9"
TAESD_REPO = "madebyollin/taesd"
TAESD_REVISION = "614f76814bbe30edbe2e627ace1c2234c81a2c0e"
CLIP_REPO = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
CLIP_REVISION = "1627032197142fbe2a7cfec626f4ced3ae60d07a"

ARTIFACTS = {
    "taesd": [
        (TAESD_REPO, "config.json", TAESD_REVISION, "3ebefd1aa96ac474cc6430fe6ea81febccf7ee82461715691ee9315c884adeef"),
        (TAESD_REPO, "diffusion_pytorch_model.safetensors", TAESD_REVISION, "db169d69145ec4ff064e49d99c95fa05d3eb04ee453de35824a6d0f325513549"),
    ],
    "clip": [
        (CLIP_REPO, "open_clip_pytorch_model.safetensors", CLIP_REVISION, "7d129ed747e0ed53e82dfcc140382b51be66b56e6a9bdc3258afd2846e3bb019"),
    ],
}
LPIPS_URL = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
LPIPS_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(repo: str, filename: str, revision: str, target: Path, expected: str | None = None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface-hub is required only for model fetching") from exc
    path = Path(hf_hub_download(repo, filename, revision=revision, local_dir=target))
    if expected and _sha256(path) != expected:
        raise SystemExit(f"checksum mismatch for {filename}")
    return path


def _fetch_lpips() -> dict[str, str]:
    path = ROOT / "models/lpips/checkpoints/alexnet-owt-7be5be79.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        urlretrieve(LPIPS_URL, path)
    actual = _sha256(path)
    if actual != LPIPS_SHA256:
        raise SystemExit(f"checksum mismatch for LPIPS AlexNet: {actual}")
    return {"path": str(path), "url": LPIPS_URL, "sha256": actual}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=("face", "taesd", "clip", "all"), nargs="?", default="face")
    args = parser.parse_args()
    output = {}
    if args.target in {"face", "all"}:
        target = ROOT / "models"
        target.mkdir(parents=True, exist_ok=True)
        model = _download(FACE_REPO, "onnx/model_quantized.onnx", FACE_REVISION, target)
        config = _download(FACE_REPO, "config.json", FACE_REVISION, target)
        preprocessor = _download(FACE_REPO, "preprocessor_config.json", FACE_REVISION, target)
        model_config = json.loads(config.read_text())
        processor_config = json.loads(preprocessor.read_text())
        assert processor_config.get("size") == {"height": 224, "width": 224}
        assert processor_config.get("resample") == 2
        assert processor_config.get("rescale_factor") in (1 / 255, 0.00392156862745098)
        assert processor_config.get("image_mean") == [0.5, 0.5, 0.5]
        assert processor_config.get("image_std") == [0.5, 0.5, 0.5]
        assert model_config.get("image_size") == 224
        assert model_config.get("id2label") == {"0": "Realism", "1": "Deepfake"}
        output["face"] = {"model": str(model), "sha256": _sha256(model)}
    for family in ("taesd", "clip"):
        if args.target not in {family, "all"}:
            continue
        target = ROOT / "models" / family
        target.mkdir(parents=True, exist_ok=True)
        output[family] = {}
        for repo, filename, revision, expected in ARTIFACTS[family]:
            path = _download(repo, filename, revision, target, expected)
            output[family][filename] = {"path": str(path), "sha256": expected, "revision": revision}
    if args.target in {"taesd", "all"}:
        output["lpips"] = _fetch_lpips()
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
