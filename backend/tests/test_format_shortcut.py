import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path("scripts/check_format_shortcut.py")
    spec = importlib.util.spec_from_file_location("check_format_shortcut", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_metadata_stump_detects_a_format_shortcut():
    module = _module()
    names = ["format=JPEG", "format=PNG", "width", "height", "file_size", "exif_present"]
    values = np.asarray([
        [1, 0, 100, 100, 1000, 1],
        [1, 0, 120, 100, 1100, 1],
        [0, 1, 100, 100, 2000, 0],
        [0, 1, 120, 100, 2100, 0],
    ], dtype=float)
    stump = module._fit_stump(names, values, [False, False, True, True])
    assert stump["feature"] in {"format=JPEG", "format=PNG"}
    assert module._auc([0, 0, 1, 1], [False, False, True, True]) == 1.0


def test_metadata_feature_groups_are_independently_selectable():
    module = _module()
    rows = [
        {"metadata": {"format": "JPEG", "width": 100, "height": 100, "file_size": 1000, "exif_present": 1}},
        {"metadata": {"format": "PNG", "width": 120, "height": 100, "file_size": 2000, "exif_present": 0}},
    ]
    assert module._features(rows, "format")[0] == ["format=JPEG", "format=PNG"]
    assert module._features(rows, "dimensions")[0] == ["width", "height"]
    assert module._features(rows, "file_size")[0] == ["file_size"]
    assert module._features(rows, "exif")[0] == ["exif_present"]
