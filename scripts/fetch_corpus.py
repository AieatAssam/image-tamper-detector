#!/usr/bin/env python3
"""Fetch the optional real-image corpus and verify every byte."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/corpus/MANIFEST.yaml"
REAL = ROOT / "data/corpus/real"
USER_AGENT = "image-tamper-detector-corpus/1.0 (+https://github.com/)"


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


def _verify(entry: dict, path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing real corpus file: {path}")
    actual = _sha256(path)
    if actual != entry["sha256"]:
        raise RuntimeError(f"checksum mismatch for {entry['id']}: expected {entry['sha256']}, got {actual}")
    if path.stat().st_size != int(entry["bytes"]):
        raise RuntimeError(f"byte count mismatch for {entry['id']}: expected {entry['bytes']}, got {path.stat().st_size}")


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
