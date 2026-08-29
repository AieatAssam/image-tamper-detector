#!/usr/bin/env python3
"""Fetch the pinned optional ONNX face-deepfake model and verify its configs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "onnx-community/Deep-Fake-Detector-v2-Model-ONNX"
REVISION = "4ea3d66dfb1bedca29727c6a0c6fa061d5f3f9c9"
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    target = ROOT / "models/onnx"
    target.mkdir(parents=True, exist_ok=True)
    model = Path(hf_hub_download(REPO, "onnx/model_quantized.onnx", revision=REVISION, local_dir=ROOT / "models"))
    config = Path(hf_hub_download(REPO, "config.json", revision=REVISION, local_dir=ROOT / "models"))
    preprocessor = Path(hf_hub_download(REPO, "preprocessor_config.json", revision=REVISION, local_dir=ROOT / "models"))
    model_config = json.loads(config.read_text())
    processor_config = json.loads(preprocessor.read_text())
    assert processor_config.get("size") == {"height": 224, "width": 224}
    assert processor_config.get("resample") == 2
    assert processor_config.get("rescale_factor") in (1 / 255, 0.00392156862745098)
    assert processor_config.get("image_mean") == [0.5, 0.5, 0.5]
    assert processor_config.get("image_std") == [0.5, 0.5, 0.5]
    assert model_config.get("image_size") == 224
    assert model_config.get("id2label") == {"0": "Realism", "1": "Deepfake"}
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    print(json.dumps({"model": str(model), "sha256": digest, "config": str(config), "preprocessor": str(preprocessor)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
