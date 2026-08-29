from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_synthetic_corpus_is_reproducible_and_balanced(tmp_path: Path) -> None:
    generator = _load("make_corpus")
    left, right = tmp_path / "left", tmp_path / "right"
    generator.generate(20260828, left, ROOT / "data/samples")
    generator.generate(20260828, right, ROOT / "data/samples")
    left_files = sorted(p.relative_to(left) for p in left.rglob("*"))
    right_files = sorted(p.relative_to(right) for p in right.rglob("*"))
    assert left_files == right_files
    assert all((left / p).read_bytes() == (right / p).read_bytes() for p in left_files)

    index = json.loads((left / "index.json").read_text())
    assert len(index["entries"]) == 100
    counts = defaultdict(Counter)
    for entry in index["entries"]:
        counts[entry["source_image"]][entry["label"]] += 1
    assert len(counts) >= 3
    assert sum(e["label"] == "authentic" for e in index["entries"]) >= 40
    assert sum(e["label"] == "manipulated" for e in index["entries"]) >= 60
    assert {e["source_label"] for e in index["entries"]} == {"authentic"}
    totals = Counter(e["label"] for e in index["entries"])
    assert all(
        values["authentic"] and values["manipulated"]
        and all(values[label] <= totals[label] * 0.4 for label in totals)
        for values in counts.values()
    )
    required = {"authentic_recompress", "splice", "copy_move", "double_compress_aligned", "double_compress_shifted", "local_retouch", "resize_then_save"}
    assert required <= {e["family"] for e in index["entries"]}
    for entry in index["entries"]:
        sidecar = json.loads((left / f"{entry['id']}.json").read_text())
        assert sidecar["source_label"] == entry["source_label"]
        with Image.open(left / Path(entry["path"]).name) as image:
            assert image.format == "JPEG"
            assert image.getexif()
        mask = np.asarray(Image.open(left / Path(entry["mask"]).name).convert("L"))
        if entry["label"] == "manipulated" and not entry["family"].startswith("double_compress"):
            assert mask.max() > 0


def test_rank_auc_contract_edges() -> None:
    benchmark = _load("benchmark")
    assert benchmark._auc([0.1] * 10, [False] * 5 + [True] * 5) == 0.5
    assert benchmark._auc(list(np.linspace(0.1, 0.9, 10)), [False] * 5 + [True] * 5) == 1.0
