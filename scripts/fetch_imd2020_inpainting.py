#!/usr/bin/env python3
"""Fetch and sample paired IMD2020 Yu et al. 2018 inpainting images."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import zipfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_imd2020 import CHUNK_SIZE, USER_AGENT, _ssl_context  # noqa: E402

CACHE = ROOT / "data/corpus/real/r15b-imd2020-inpainting-download"
EXTRACTED = CACHE / "extracted"
ARCHIVES = {
    "01": {
        "url": "https://staff.utia.cas.cz/novozada/db/IMD2020_Generative_Image_Inpainting_yu2018_01.zip",
        "filename": "IMD2020_Generative_Image_Inpainting_yu2018_01.zip",
        "bytes": 1722669427,
        "sha256": "4cefd85107326757c0ec3e6db4eb573ad2fb5cd4aa324e3d4a49fcb51dcd9aa9",
    },
    "real01": {
        "url": "https://staff.utia.cas.cz/novozada/db/IMD2020_real_01.zip",
        "filename": "IMD2020_real_01.zip",
        "bytes": 1960999112,
        "sha256": "158e98cf8923b9eff7b4b3ff44d5b303e6b19dd30b1006d285a664698ce7add1",
    },
    "mask": {
        "url": "https://staff.utia.cas.cz/novozada/db/IMD2020_Generative_Image_Inpainting_yu2018_mask.zip",
        "filename": "IMD2020_Generative_Image_Inpainting_yu2018_mask.zip",
        "bytes": 126008215,
        "sha256": "08ab044b56930066955853a14f88710023ee03d092f1bb4f7af1db93000a4573",
    },
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
DEFAULT_COUNT = 200
DEFAULT_SEED = 20260828
METHOD = "generative inpainting via Yu et al. 2018"
CITATION = (
    'Novozamsky, Mahdian, and Saic, "IMD2020: A Large-Scale Annotated Dataset '
    'Tailored for Detecting Manipulated Images," IEEE WACV Workshops 2020.'
)


class Pair:
    __slots__ = ("key", "manipulated", "real", "mask", "camera_group")

    def __init__(self, key: str, manipulated: Path, real: Path, mask: Path, camera_group: str) -> None:
        self.key = key
        self.manipulated = manipulated
        self.real = real
        self.mask = mask
        self.camera_group = camera_group


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(key: str) -> dict[str, str | int]:
    spec = ARCHIVES[key]
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / spec["filename"]
    partial = archive.with_suffix(archive.suffix + ".part")
    if archive.is_file() and archive.stat().st_size == spec["bytes"]:
        digest = _sha256(archive)
        if digest != spec["sha256"]:
            raise RuntimeError(f"{key} archive checksum mismatch: expected {spec['sha256']}, got {digest}")
        return {"key": key, "url": spec["url"], "bytes": archive.stat().st_size, "sha256": digest}
    request = Request(spec["url"], headers={"User-Agent": USER_AGENT})
    with urlopen(request, context=_ssl_context(), timeout=120) as response, partial.open("wb") as output:
        size = 0
        digest = hashlib.sha256()
        while chunk := response.read(CHUNK_SIZE):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if size != spec["bytes"]:
        raise RuntimeError(f"{key} archive size mismatch: expected {spec['bytes']}, got {size}")
    if digest.hexdigest() != spec["sha256"]:
        raise RuntimeError(f"{key} archive checksum mismatch: expected {spec['sha256']}, got {digest.hexdigest()}")
    partial.replace(archive)
    return {"key": key, "url": spec["url"], "bytes": size, "sha256": digest.hexdigest()}


def _safe_extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(archive) as source:
        if bad := source.testzip():
            raise RuntimeError(f"corrupt archive member: {bad}")
        for member in source.infolist():
            relative = PurePosixPath(member.filename)
            destination = (target / Path(*relative.parts)).resolve()
            if destination != root and root not in destination.parents:
                raise RuntimeError(f"archive path escapes extraction directory: {member.filename}")
            source.extract(member, target)


def _files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _unique_by_name(paths: list[Path], role: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in paths:
        if path.name in output:
            raise RuntimeError(f"duplicate {role} basename: {path.name}")
        output[path.name] = path
    return output


def discover_pairs() -> list[Pair]:
    fake_root = EXTRACTED / "01"
    real_root = EXTRACTED / "real01"
    mask_root = EXTRACTED / "mask"
    fake_by_name = _unique_by_name(_files(fake_root), "inpainting")
    real_by_name = _unique_by_name(_files(real_root), "real")
    mask_by_name: dict[str, Path] = {}
    for path in _files(mask_root):
        suffix = path.suffix
        if not path.name.endswith(f"_mask{suffix}"):
            continue
        key = f"{path.name[:-len(f'_mask{suffix}')]}{suffix}"
        if key in mask_by_name:
            raise RuntimeError(f"duplicate mask basename: {path.name}")
        mask_by_name[key] = path

    pairs = []
    for name, manipulated in fake_by_name.items():
        real = real_by_name.get(name)
        mask = mask_by_name.get(name)
        if real is None or mask is None:
            continue
        relative = real.relative_to(real_root)
        if len(relative.parts) < 2:
            raise RuntimeError(f"real image has no camera group: {real}")
        pairs.append(Pair(name, manipulated, real, mask, relative.parent.as_posix()))
    if not pairs:
        raise RuntimeError("no complete inpainting/real/mask basename triples found")
    return sorted(pairs, key=lambda pair: pair.key)


def select_pairs(pairs: list[Pair], count: int = DEFAULT_COUNT, seed: int = DEFAULT_SEED, max_per_camera_group: int = 1) -> list[Pair]:
    if not 150 <= count <= 250:
        raise ValueError("count must be between 150 and 250 paired images")
    if max_per_camera_group < 1:
        raise ValueError("max_per_camera_group must be at least 1")
    by_camera: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_camera.setdefault(pair.camera_group, []).append(pair)
    camera_groups = sorted(by_camera)
    capacity = sum(min(max_per_camera_group, len(values)) for values in by_camera.values())
    if count > capacity:
        raise ValueError(f"cannot select {count} pairs; camera-group capacity is {capacity}")
    rng = random.Random(seed)
    for values in by_camera.values():
        rng.shuffle(values)
    rng.shuffle(camera_groups)
    selected: list[Pair] = []
    for round_index in range(max_per_camera_group):
        for camera_group in camera_groups:
            if len(selected) >= count:
                break
            values = by_camera[camera_group]
            if round_index < len(values):
                selected.append(values[round_index])
    return sorted(selected, key=lambda pair: pair.key)


def _split_by_pair(pairs: list[Pair], seed: int) -> dict[str, str]:
    keys = [pair.key for pair in pairs]
    random.Random(seed).shuffle(keys)
    train_count = max(1, int(len(keys) * 0.7 + 0.999999))
    return {key: "train" if index < train_count else "heldout" for index, key in enumerate(keys)}


def _sha256_and_size(path: Path) -> tuple[str, int]:
    return _sha256(path), path.stat().st_size


def _verify_pair(pair: Pair) -> None:
    with Image.open(pair.manipulated) as manipulated, Image.open(pair.real) as real, Image.open(pair.mask) as mask:
        if manipulated.size != mask.size:
            raise RuntimeError(f"mask size mismatch: {pair.key}")
        if manipulated.format != real.format:
            raise RuntimeError(f"pair format mismatch: {pair.key}")
        manipulated.verify()
        real.verify()
        mask.verify()


def build_selection(pairs: list[Pair], count: int, seed: int, max_per_camera_group: int) -> dict[str, object]:
    selected = select_pairs(pairs, count, seed, max_per_camera_group)
    split_by_pair = _split_by_pair(selected, seed)
    return {
        "dataset": "IMD2020 Large-Scale Set of Inpainting Images",
        "method": METHOD,
        "citation": CITATION,
        "seed": seed,
        "requested_pairs": count,
        "max_per_camera_group": max_per_camera_group,
        "candidate_inpainting_images": len(pairs),
        "candidate_camera_groups": len({pair.camera_group for pair in pairs}),
        "selected_pairs": [
            {
                "id": f"imd2020_inpaint_{pair.key.rsplit('.', 1)[0]}",
                "key": pair.key,
                "camera_group": pair.camera_group,
                "split": split_by_pair[pair.key],
                "manipulated": str(pair.manipulated),
                "real": str(pair.real),
                "mask": str(pair.mask),
            }
            for pair in selected
        ],
    }


def write_candidate_jsonl(selection: dict[str, object], output: Path) -> None:
    rows = []
    for pair in selection["selected_pairs"]:
        source_group = f"imd2020_inpaint/{pair['key']}"
        for path, label in ((pair["manipulated"], "ai_generated"), (pair["real"], "authentic")):
            rows.append({"axis": "imd2020_inpaint", "label": label, "path": path, "source_image": source_group, "split": pair["split"]})
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _yaml_string(value: str) -> str:
    return json.dumps(value)


def _manifest_source(selection: dict[str, object], archives: list[dict[str, str | int]]) -> str:
    archive_lines = []
    for archive in archives:
        archive_lines.extend([
            f"      - url: {_yaml_string(str(archive['url']))}",
            f"        sha256: {archive['sha256']}",
            f"        size: {archive['bytes']}",
        ])
    pair_ids = [pair["id"] for pair in selection["selected_pairs"]]
    lines = [
        "  imd2020_inpaint:",
        '    title: "IMD2020 Large-Scale Set of Inpainting Images"',
        f"    method: {_yaml_string(METHOD)}",
        f"    citation: {_yaml_string(CITATION)}",
        '    license: "No explicit redistribution license published; local-only under D7b."',
        "    archives:",
        *archive_lines,
        f"    sample_seed: {selection['seed']}",
        f"    sample_pairs: {selection['requested_pairs']}",
        "    sample_stratified_by: camera_group",
        f"    max_per_camera_group: {selection['max_per_camera_group']}",
        f"    candidate_pairs: {selection['candidate_inpainting_images']}",
        f"    candidate_camera_groups: {selection['candidate_camera_groups']}",
        "    selected_pair_ids:",
        *[f"      - {_yaml_string(pair_id)}" for pair_id in pair_ids],
    ]
    return "\n".join(lines)


def append_manifest(manifest: Path, selection: dict[str, object], archives: list[dict[str, str | int]]) -> None:
    text = manifest.read_text()
    if "  imd2020_inpaint:" in text or "    axis: imd2020_inpaint\n" in text:
        raise RuntimeError("manifest already contains imd2020_inpaint")
    source, images = text.split("images:\n", 1)
    rows = []
    for pair in selection["selected_pairs"]:
        pair_id = pair["id"]
        source_group = f"imd2020_inpaint/{pair['key']}"
        for suffix, label, path_key, mask in (
            ("manipulated", "manipulated", "manipulated", pair["mask"]),
            ("authentic", "authentic", "real", None),
        ):
            path = Path(pair[path_key])
            sha, size = _sha256_and_size(path)
            relative = path.relative_to(ROOT).as_posix()
            mask_relative = Path(mask).relative_to(ROOT).as_posix() if mask else None
            rows.extend([
                f"  - id: {_yaml_string(f'{pair_id}_{suffix}')}",
                "    axis: imd2020_inpaint",
                f"    path: {_yaml_string(relative)}",
                f"    sha256: {sha}",
                f"    size: {size}",
                f"    label: {label}",
                f"    mask: {_yaml_string(mask_relative) if mask_relative else 'null'}",
                f"    source_group: {_yaml_string(source_group)}",
                f"    split: {pair['split']}",
                f"    camera_group: {_yaml_string(pair['camera_group'])}",
                f"    operation: {_yaml_string(METHOD)}",
            ])
            rows.append("")
    manifest.write_text(source.rstrip() + "\n" + _manifest_source(selection, archives) + "\nimages:\n" + images.rstrip() + "\n" + "\n".join(rows))


def _inventory() -> dict[str, object]:
    output: dict[str, object] = {}
    for key, spec in ARCHIVES.items():
        archive = CACHE / spec["filename"]
        with zipfile.ZipFile(archive) as source:
            members = [name for name in source.namelist() if Path(name).suffix.lower() in IMAGE_SUFFIXES]
        output[key] = {"archive": str(archive), "image_members": len(members), "first_members": members[:8]}
    pairs = discover_pairs()
    output["complete_pairs"] = len(pairs)
    output["camera_groups"] = len({pair.camera_group for pair in pairs})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--candidate", type=Path, default=Path("/tmp/r15b-imd2020-inpaint.jsonl"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if not args.download and not args.inspect and not args.sample:
        parser.error("choose --download, --inspect, or --sample")
    if args.download:
        print(json.dumps([_download(key) for key in ("01", "real01", "mask")], sort_keys=True))
        _safe_extract(CACHE / ARCHIVES["01"]["filename"], EXTRACTED / "01")
        _safe_extract(CACHE / ARCHIVES["real01"]["filename"], EXTRACTED / "real01")
        _safe_extract(CACHE / ARCHIVES["mask"]["filename"], EXTRACTED / "mask")
    if args.inspect:
        print(json.dumps(_inventory(), indent=2, sort_keys=True))
    if args.sample:
        pairs = discover_pairs()
        selected = select_pairs(pairs, args.count, args.seed)
        for pair in selected:
            _verify_pair(pair)
        selection = build_selection(pairs, args.count, args.seed, 1)
        write_candidate_jsonl(selection, args.candidate)
        if args.manifest:
            archives = [_download(key) for key in ("01", "real01", "mask")]
            append_manifest(args.manifest, selection, archives)
        print(json.dumps({"candidate": str(args.candidate), **{key: value for key, value in selection.items() if key != "selected_pairs"}, "selected_pair_ids": [pair["id"] for pair in selection["selected_pairs"]]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
