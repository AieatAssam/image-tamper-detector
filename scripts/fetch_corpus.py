#!/usr/bin/env python3
"""Fetch the optional real-image corpus and verify every byte."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from PIL import Image
from c2pa import Reader

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/corpus/MANIFEST.yaml"
REAL = ROOT / "data/corpus/real"
USER_AGENT = "image-tamper-detector-corpus/1.0 (+https://github.com/)"
ALLOWED_LICENSE = re.compile(
    r"^(?:CC0|CC BY(?:-SA)?(?: [0-9]+(?:\.[0-9]+)?)?|Public domain|PD|MIT OR Apache-2\.0)$"
)
EDITOR_SOFTWARE = ("gimp", "adobe", "photoshop", "lightroom", "paintshop", "affinity")


def _manifest() -> dict:
    text = MANIFEST.read_text()
    try:
        import yaml
    except ImportError:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit("PyYAML is required to read MANIFEST.yaml") from exc
    return yaml.safe_load(text)


def _extension(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"} else ".jpg"


def _path(entry: dict) -> Path:
    return REAL / f"{entry['id']}{_extension(entry['url'])}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exif_value(exif, tag: int):
    value = exif.get(tag)
    return value if value is not None else exif.get_ifd(0x8769).get(tag)


def _verify(entry: dict, path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing real corpus file: {path}")
    actual = _sha256(path)
    if actual != entry["sha256"]:
        raise RuntimeError(f"checksum mismatch for {entry['id']}: expected {entry['sha256']}, got {actual}")
    if path.stat().st_size != int(entry["bytes"]):
        raise RuntimeError(f"byte count mismatch for {entry['id']}: expected {entry['bytes']}, got {path.stat().st_size}")
    license_name = entry.get("license", "")
    if not ALLOWED_LICENSE.fullmatch(license_name):
        raise RuntimeError(f"unsupported license for {entry['id']}: {license_name!r}")
    if any(key.startswith("utm_") for key in parse_qs(urlparse(entry["url"]).query)):
        raise RuntimeError(f"tracking query string in URL for {entry['id']}")
    axis = entry.get("axis")
    if axis == "real_camera":
        with Image.open(path) as image:
            if image.format != "JPEG":
                raise RuntimeError(f"real_camera entry is not JPEG: {entry['id']}")
            exif = image.getexif()
            if not _exif_value(exif, 0x010F) or not _exif_value(exif, 0x0110):
                raise RuntimeError(f"real_camera entry lacks EXIF Make/Model: {entry['id']}")
            evidence = entry.get("unresized_evidence")
            if evidence not in {"strict", "relaxed"}:
                raise RuntimeError(f"real_camera entry lacks unresized_evidence: {entry['id']}")
            pixel_x = _exif_value(exif, 0xA002)
            if evidence == "strict" and pixel_x != image.width:
                raise RuntimeError(
                    f"real_camera PixelXDimension mismatch for {entry['id']}: "
                    f"{pixel_x!r} != {image.width}"
                )
            if evidence == "relaxed":
                software = _exif_value(exif, 0x0131)
                is_editor = software and any(name in str(software).lower() for name in EDITOR_SOFTWARE)
                if pixel_x is not None or is_editor:
                    raise RuntimeError(f"real_camera relaxed evidence is invalid for {entry['id']}")
    elif axis == "real_ai" and entry.get("label") != "ai_generated":
        raise RuntimeError(f"real_ai entry must be labelled ai_generated: {entry['id']}")
    elif axis == "real_c2pa_signed":
        expected = entry.get("c2pa_validation")
        if expected not in {"valid", "invalid"}:
            raise RuntimeError(f"real_c2pa_signed entry lacks c2pa_validation: {entry['id']}")
        try:
            with Reader(path) as reader:
                store = json.loads(reader.json())
        except Exception as exc:
            raise RuntimeError(f"C2PA manifest could not be parsed for {entry['id']}: {exc}") from exc
        actual = str(store.get("validation_state", "")).lower()
        if actual != expected:
            raise RuntimeError(
                f"C2PA validation mismatch for {entry['id']}: expected {expected}, got {actual or 'missing'}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify existing files without downloading")
    args = parser.parse_args()
    entries = _manifest().get("images", [])
    REAL.mkdir(parents=True, exist_ok=True)
    last_request = 0.0
    try:
        for entry in entries:
            path = _path(entry)
            if path.exists():
                _verify(entry, path)
                print(f"verified {entry['id']}")
                continue
            if args.check:
                raise RuntimeError(f"missing real corpus file: {path}")
            delay = 0.5 - (time.monotonic() - last_request)
            if delay > 0:
                time.sleep(delay)
            request = Request(entry["url"], headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=60) as response:
                data = response.read()
            path.write_bytes(data)
            last_request = time.monotonic()
            _verify(entry, path)
            print(f"fetched and verified {entry['id']}")
    except Exception as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1
    print(f"{len(entries)} manifest entries verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
