#!/usr/bin/env python3
"""Select a deterministic, source-stratified IMD2020 pair sample.

The extracted archive is intentionally not downloaded by this script. It
expects each source directory to contain one original and one mask for every
manipulated image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

SEED = 20260828
DEFAULT_ROOT = Path("data/corpus/imd2020")
DEFAULT_OUT = DEFAULT_ROOT / "sample-selection.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
MASK_MARKERS = {"annotation", "annotations", "binary", "ground", "groundtruth", "gt", "mask", "masks", "truth"}
REAL_MARKERS = {"authentic", "original", "orig", "pristine", "real", "source"}
ROLE_MARKERS = MASK_MARKERS | REAL_MARKERS | {"edited", "fake", "forged", "manipulated", "tampered"}


@dataclass(frozen=True)
class Pair:
    key: str
    source_group: str
    manipulated: Path
    real: Path
    mask: Path


def _tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]


def _role(path: Path, root: Path) -> str:
    tokens = [token for part in path.relative_to(root).parts for token in _tokens(part)]
    if MASK_MARKERS & set(tokens):
        return "mask"
    if REAL_MARKERS & set(tokens):
        return "real"
    return "manipulated"


def _pair_key(path: Path) -> str:
    stem_tokens = [token for token in _tokens(path.stem) if token not in ROLE_MARKERS]
    if not stem_tokens:
        raise ValueError(f"cannot derive a pair id from {path}")
    return "_".join(stem_tokens)


def discover_pairs(root: Path) -> list[Pair]:
    """Discover complete manipulated/mask/real triples below *root*."""
    if not root.is_dir():
        raise ValueError(f"extracted IMD2020 directory does not exist: {root}")

    pairs = []
    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        files = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        roles: dict[str, list[Path]] = {"manipulated": [], "real": [], "mask": []}
        for path in files:
            roles[_role(path, root)].append(path)
        if not roles["real"] or not roles["manipulated"] or not roles["mask"]:
            continue
        if len(roles["real"]) != 1:
            raise ValueError(f"expected one real counterpart in {directory}, found {len(roles['real'])}")
        source_group = directory.relative_to(root).as_posix()
        for manipulated in roles["manipulated"]:
            mask_candidates = [path for path in roles["mask"] if _pair_key(path) == _pair_key(manipulated)]
            key = _pair_key(manipulated)
            if len(mask_candidates) != 1:
                raise ValueError(
                    f"expected one mask for {source_group}:{key}, found {len(mask_candidates)}"
                )
            pairs.append(Pair(key, source_group, manipulated, roles["real"][0], mask_candidates[0]))

    if not pairs:
        raise ValueError(
            f"no complete IMD2020 triples found below {root}; expected one "
            "manipulated image, one real counterpart, and one mask per id"
        )
    return pairs


def select_pairs(pairs: list[Pair], count: int, seed: int = SEED, max_per_source: int = 2) -> list[Pair]:
    """Return *count* pairs breadth-first with a per-source cap."""
    if not 100 <= count <= 500:
        raise ValueError("count must be between 100 and 500 manipulated images")
    if max_per_source < 1:
        raise ValueError("max_per_source must be at least 1")
    by_source: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_source.setdefault(pair.source_group, []).append(pair)
    source_groups = sorted(by_source)
    capacity = sum(min(max_per_source, len(values)) for values in by_source.values())
    if count > capacity:
        raise ValueError(f"cannot select {count} pairs with max_per_source={max_per_source}; capacity is {capacity}")

    rng = random.Random(seed)
    for values in by_source.values():
        rng.shuffle(values)

    selected = []
    for round_index in range(max_per_source):
        for source_group in source_groups:
            if len(selected) >= count:
                break
            values = by_source[source_group]
            if round_index < len(values):
                selected.append(values[round_index])
    if len({(pair.source_group, pair.key) for pair in selected}) != len(selected):
        raise ValueError("pair ids are not unique; refusing an ambiguous sample")
    return sorted(selected, key=lambda pair: (pair.source_group, pair.key))


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _split_groups(source_groups: list[str], seed: int) -> dict[str, str]:
    order = sorted(source_groups)
    random.Random(seed).shuffle(order)
    train_count = max(1, int(len(order) * 0.7 + 0.999999))
    train = set(order[:train_count])
    return {source_group: "train" if source_group in train else "heldout" for source_group in order}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_row(row_id: str, path: str, label: str, mask: str | None, source_group: str, split: str, root: Path) -> str:
    local = root / Path(path)
    fields = [
        f"  - id: {json.dumps(row_id)}",
        "    axis: imd2020",
        f"    path: {json.dumps(f'data/corpus/imd2020/extracted/{path}')}",
        f"    sha256: {_sha256(local)}",
        f"    size: {local.stat().st_size}",
        f"    label: {label}",
        f"    mask: {json.dumps(f'data/corpus/imd2020/extracted/{mask}') if mask else 'null'}",
        f"    source_group: {json.dumps(f'imd2020/{source_group}')}",
        f"    split: {split}",
    ]
    return "\n".join(fields)


def append_manifest(manifest_path: Path, root: Path, selection: dict) -> int:
    """Append selected image rows while leaving the existing manifest text intact."""
    if not manifest_path.is_file():
        raise ValueError(f"manifest does not exist: {manifest_path}")
    blocks = re.split(r"(?=^  - id: )", manifest_path.read_text(), flags=re.MULTILINE)
    base = blocks[0] + "".join(block for block in blocks[1:] if "\n    axis: imd2020\n" not in block)
    rows = []
    for pair in selection["pairs"]:
        pair_id = pair["id"]
        source_group = pair["source_group"]
        split = pair["split"]
        rows.append(_manifest_row(f"{pair_id}_manipulated", pair["manipulated"]["path"], "manipulated", pair["manipulated"]["mask"], source_group, split, root))
        rows.append(_manifest_row(f"{pair_id}_authentic", pair["real"]["path"], "authentic", None, source_group, split, root))
    manifest_path.write_text(base.rstrip() + "\n" + "\n".join(rows) + "\n")
    return len(rows)


def build_selection(root: Path, pairs: list[Pair], count: int, seed: int, max_per_source: int) -> dict:
    selected = select_pairs(pairs, count, seed, max_per_source)
    split_by_source = _split_groups(sorted({pair.source_group for pair in selected}), seed)
    by_source = {}
    for pair in selected:
        by_source[pair.source_group] = by_source.get(pair.source_group, 0) + 1
    selected_source_groups = sorted(by_source)
    return {
        "dataset": "IMD2020",
        "seed": seed,
        "requested_manipulated_images": count,
        "max_per_source": max_per_source,
        "n_pairs": len(selected),
        "n_images": len(selected) * 2,
        "candidate_pairs": len(pairs),
        "candidate_source_groups": len({pair.source_group for pair in pairs}),
        "selected_source_groups": len(selected_source_groups),
        "candidate_pairs_by_source_group": {
            name: sum(pair.source_group == name for pair in pairs) for name in sorted({pair.source_group for pair in pairs})
        },
        "selected_pairs_by_source_group": by_source,
        "pairs": [
            {
                "id": f"imd2020_{pair.source_group.replace('/', '_')}_{pair.key}",
                "source_group": pair.source_group,
                "split": split_by_source[pair.source_group],
                "manipulated": {"path": _relative(pair.manipulated, root), "label": "manipulated", "mask": _relative(pair.mask, root)},
                "real": {"path": _relative(pair.real, root), "label": "authentic"},
            }
            for pair in selected
        ],
    }


def _self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for source_index in range(250):
            source = root / f"group_{source_index:03d}"
            source.mkdir(parents=True, exist_ok=True)
            (source / f"base_{source_index:03d}_original.jpg").touch()
            for pair_index in range(2):
                pair_id = f"pair_{source_index:03d}_{pair_index}"
                (source / f"{pair_id}.jpg").touch()
                (source / f"{pair_id}_mask.png").touch()
        pairs = discover_pairs(root)
        assert len(pairs) == 500
        selected_a = select_pairs(pairs, 300, SEED)
        selected_b = select_pairs(pairs, 300, SEED)
        assert [(p.key, p.source_group) for p in selected_a] == [(p.key, p.source_group) for p in selected_b]
        counts = {source: sum(pair.source_group == source for pair in selected_a) for source in {pair.source_group for pair in selected_a}}
        assert len(counts) == 250 and max(counts.values()) == 2
        assert len(selected_a) == 300


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="extracted IMD2020 directory")
    parser.add_argument("--count", type=int, default=400, help="number of manipulated images/pairs to select")
    parser.add_argument("--max-per-source", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="selection JSON path")
    parser.add_argument("--manifest", type=Path, help="append selected image rows to this manifest")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print("sample_imd2020 self-test: PASS")
        return 0

    root = args.root.resolve()
    selection = build_selection(root, discover_pairs(root), args.count, args.seed, args.max_per_source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    if args.manifest:
        added = append_manifest(args.manifest, root, selection)
        print(f"manifest rows added: {added} -> {args.manifest}")
    print(f"selected {selection['n_pairs']} IMD2020 pairs ({selection['n_images']} images) -> {args.out}")
    print(
        f"selected {selection['selected_source_groups']} source groups with "
        f"max_per_source={selection['max_per_source']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
