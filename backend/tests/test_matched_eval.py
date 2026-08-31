from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).parents[2]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _image(path: Path, image_format: str, size: tuple[int, int]) -> None:
    Image.new("RGB", size, "gray").save(path, format=image_format)


def _entry(identifier: str, path: Path, label: str, axis: str) -> dict:
    return {"id": identifier, "path": path, "label": label, "axis": axis}


def test_matching_is_seeded_and_without_replacement(tmp_path: Path) -> None:
    matcher = _load("matched_eval")
    ai = []
    real = []
    for index in range(3):
        ai_path = tmp_path / f"ai_{index}.jpg"
        real_path = tmp_path / f"real_{index}.jpg"
        _image(ai_path, "JPEG", (100 + index, 100))
        _image(real_path, "JPEG", (100 + index, 100))
        ai.append(_entry(f"ai_{index}", ai_path, "ai_generated", "sd35_flux"))
        real.append(_entry(f"real_{index}", real_path, "authentic", "imd2020"))

    first = matcher.match_entries(ai, real, seed=20260828, tolerance=1)
    second = matcher.match_entries(ai, real, seed=20260828, tolerance=1)

    assert first["n_pairs"] == 3
    assert [(pair["ai"]["id"], pair["real"]["id"]) for pair in first["pairs"]] == [
        (pair["ai"]["id"], pair["real"]["id"]) for pair in second["pairs"]
    ]
    assert len({pair["real"]["id"] for pair in first["pairs"]}) == 3
    assert all(row["source_image"] == row["matched_pair_id"] for row in matcher.rows(first))


def test_matching_rejects_format_and_exif_mismatches(tmp_path: Path) -> None:
    matcher = _load("matched_eval")
    ai_path = tmp_path / "ai.png"
    real_path = tmp_path / "real.jpg"
    _image(ai_path, "PNG", (100, 100))
    _image(real_path, "JPEG", (100, 100))
    ai = _entry("ai", ai_path, "ai_generated", "sd35_flux")
    real = _entry("real", real_path, "authentic", "imd2020")

    assert matcher._distance(matcher._metadata(ai_path), matcher._metadata(real_path)) is None
    assert matcher.match_entries([ai], [real], tolerance=100)["n_pairs"] == 0


def test_benchmark_reads_matched_jsonl(tmp_path: Path) -> None:
    benchmark = _load("benchmark")
    path = tmp_path / "matched.jsonl"
    rows = [
        {
            "id": "ai",
            "path": str(tmp_path / "ai.jpg"),
            "label": "ai_generated",
            "axis": "sd35_flux",
            "source_image": "pair",
        },
        {
            "id": "real",
            "path": str(tmp_path / "real.jpg"),
            "label": "authentic",
            "axis": "imd2020",
            "source_image": "pair",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    loaded = benchmark._matched(path)

    assert {row["corpus"] for row in loaded} == {"matched"}
    assert {row["source_image"] for row in loaded} == {"pair"}
