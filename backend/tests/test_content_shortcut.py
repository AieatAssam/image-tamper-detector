import importlib.util
from pathlib import Path

from PIL import Image


def _module():
    path = Path("scripts/check_content_shortcut.py")
    spec = importlib.util.spec_from_file_location("check_content_shortcut", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _rows(tmp_path, colors):
    rows = []
    for index, (label, color) in enumerate(colors):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (64, 64), color).save(path)
        rows.append({"path": path, "label": label, "axis": "synthetic", "source_image": f"group-{index}"})
    return rows


def test_content_shortcut_fires_on_obvious_thumbnail_difference(tmp_path):
    module = _module()
    rows = _rows(tmp_path, [(True, "red")] * 20 + [(False, "blue")] * 20)
    result = module.evaluate(rows)
    assert result["test"]["auc"] > 0.99
    assert result["test"]["se"] == 0.0
    assert not result["content_pass"]


def test_content_shortcut_stays_at_chance_when_images_are_identical(tmp_path):
    module = _module()
    rows = _rows(tmp_path, [(True, "gray")] * 20 + [(False, "gray")] * 20)
    result = module.evaluate(rows)
    assert result["test"]["auc"] == 0.5
    assert result["test"]["shortcut_auc"] == 0.5
    assert result["content_pass"]


def test_axes_select_requested_axis_and_camera_negative(tmp_path):
    module = _module()
    rows = [
        {"path": tmp_path / "a", "label": True, "axis": "axis_a"},
        {"path": tmp_path / "b", "label": True, "axis": "axis_b"},
        {"path": tmp_path / "r", "label": False, "axis": "real_camera"},
    ]
    selected = module._select_axes(rows, {"axis_a"})
    assert {row["axis"] for row in selected} == {"axis_a", "real_camera"}
