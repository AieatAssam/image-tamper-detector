#!/usr/bin/env python3
"""Generate the deterministic, processing-history image corpus for S05."""

from __future__ import annotations

import argparse
import json
import random
import struct
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, JpegPresets

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260828
GENERATED_AT = "2026-08-28T12:00:00Z"
IMAGE_SIZE = (768, 512)
JPEG_QUALITIES = (70, 80, 90, 95)
SOURCE_LABELS = {
    "landscape_original.jpg": "authentic",
    "landscape_copy_paste.jpg": "known_forgery",
    "gpt-4o-generated-receipt-01.png": "ai_generated",
    "gpt-4o-generated-receipt-02.png": "ai_generated",
}


def _source_label(source: Path) -> str:
    try:
        return SOURCE_LABELS[source.name]
    except KeyError as exc:
        raise ValueError(f"unclassified seed image: {source}") from exc


def _source_name(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _real_camera_paths() -> list[Path]:
    manifest_path = ROOT / "data/corpus/MANIFEST.yaml"
    real_dir = ROOT / "data/corpus/real"
    if not manifest_path.is_file() or not real_dir.is_dir():
        return []
    try:
        import yaml
        manifest = yaml.safe_load(manifest_path.read_text())
    except Exception:
        return []
    paths = []
    for entry in manifest.get("images", []):
        if entry.get("axis") != "real_camera":
            continue
        suffix = (Path(urlparse(entry["url"]).path).suffix or ".jpg").lower()
        path = real_dir / f"{entry['id']}{suffix}"
        if path.is_file():
            SOURCE_LABELS[path.name] = "authentic"
            paths.append(path)
    return paths


def _jpeg_bytes(image: Image.Image, quality: int, exif: bytes | None = None, variant: int = 0) -> bytes:
    out = BytesIO()
    options = {
        "format": "JPEG", "quality": quality, "optimize": variant % 2 == 0,
        "progressive": variant % 3 == 0, "subsampling": (0, 1, 2)[variant % 3],
    }
    if variant % 4 == 3:
        options.pop("quality")
        options["qtables"] = JpegPresets.presets["web_high"]["quantization"]
    if exif is not None:
        options["exif"] = exif
    image.convert("RGB").save(out, **options)
    return out.getvalue()


def _decode(data: bytes) -> Image.Image:
    with Image.open(BytesIO(data)) as image:
        return image.convert("RGB").copy()


def _save_image(image: Image.Image, path: Path, quality: int = 90, exif: bytes | None = None, variant: int = 0) -> None:
    path.write_bytes(_jpeg_bytes(image, quality, exif, variant))


def _mask(size: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    result = Image.new("L", size, 0)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(result)
    for box in boxes:
        draw.rectangle(box, fill=255)
    return result


def _region(rng: random.Random, size: tuple[int, int], fraction: float) -> tuple[int, int, int, int]:
    width, height = size
    area = max(32 * 32, int(width * height * fraction))
    rw = max(32, min(width // 2, int((area * rng.uniform(0.8, 1.2)) ** 0.5)))
    rh = max(32, min(height // 2, int(area / rw)))
    x = rng.randrange(0, max(1, width - rw + 1))
    y = rng.randrange(0, max(1, height - rh + 1))
    return x, y, x + rw, y + rh


def _different_region(rng: random.Random, size: tuple[int, int], fraction: float, source: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    for _ in range(30):
        candidate = _region(rng, size, fraction)
        if candidate[2] <= source[0] or candidate[0] >= source[2] or candidate[3] <= source[1] or candidate[1] >= source[3]:
            return candidate
    return (0, 0, max(32, size[0] // 5), max(32, size[1] // 5))


def _ascii(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.rstrip(b"\x00").decode("utf-8", "replace")
    return str(value)


def _ifd_entry(tag: int, kind: int, count: int, value: int | bytes) -> bytes:
    if isinstance(value, bytes):
        value = value[:4].ljust(4, b"\x00")
    else:
        value = struct.pack(">I", value)
    return struct.pack(">HHI", tag, kind, count) + value


def _synthetic_exif(source_exif: bytes, image: Image.Image, variant: int) -> bytes:
    source = Image.Exif()
    if source_exif:
        source.load(source_exif)
    software = (None, "Pillow", "GIMP 3.0", "Adobe Photoshop")[variant % 4]
    original = f"2024:01:{variant % 28 + 1:02d} 12:{variant % 60:02d}:00"
    modified = original if variant % 5 else "2025:02:01 12:00:00"
    values = [
        (0x010E, _ascii(source.get(0x010E)) or "synthetic corpus source"),
        (0x010F, _ascii(source.get(0x010F))),
        (0x0110, _ascii(source.get(0x0110))),
        (0x0131, software),
        (0x0132, modified),
    ]
    values = [(tag, value) for tag, value in values if value]
    thumbnail = None
    if variant % 4 == 1:
        thumbnail = _jpeg_bytes(image.resize((32, 32), Image.Resampling.LANCZOS), 80)
    elif variant % 4 == 2:
        thumbnail = _jpeg_bytes(ImageOps.mirror(image.resize((32, 32), Image.Resampling.LANCZOS)), 80)

    ifd0_count = len(values) + 1
    ifd0_size = 2 + ifd0_count * 12 + 4
    data_offset = 8 + ifd0_size
    entries = []
    data = bytearray()
    for tag, value in values:
        raw = value.encode("ascii", "replace") + b"\x00"
        if len(raw) <= 4:
            entries.append(_ifd_entry(tag, 2, len(raw), raw))
        else:
            entries.append(_ifd_entry(tag, 2, len(raw), data_offset + len(data)))
            data.extend(raw)

    exif_offset = data_offset + len(data)
    original_raw = original.encode("ascii") + b"\x00"
    exif_size = 2 + 12 + 4
    exif_data_offset = exif_offset + exif_size
    exif_entry = _ifd_entry(0x9003, 2, len(original_raw), exif_data_offset)
    exif_ifd = struct.pack(">H", 1) + exif_entry + struct.pack(">I", 0) + original_raw

    thumb_ifd = b""
    thumb_offset = 0
    if thumbnail:
        thumb_ifd_offset = exif_data_offset + len(original_raw)
        thumb_offset = thumb_ifd_offset + 2 + 2 * 12 + 4
        thumb_ifd = (
            struct.pack(">H", 2)
            + _ifd_entry(0x0201, 4, 1, thumb_offset)
            + _ifd_entry(0x0202, 4, 1, len(thumbnail))
            + struct.pack(">I", 0)
        )
    next_ifd = exif_offset + exif_size + len(original_raw) if thumbnail else 0
    ifd0 = struct.pack(">H", ifd0_count) + b"".join(entries) + _ifd_entry(0x8769, 4, 1, exif_offset) + struct.pack(">I", next_ifd)
    return b"Exif\x00\x00MM\x00*\x00\x00\x00\x08" + ifd0 + data + exif_ifd + thumb_ifd + (thumbnail or b"")


def _assert_source_balance(entries: list[dict]) -> None:
    counts: dict[str, dict[str, int]] = {}
    for entry in entries:
        source = entry["source_image"]
        counts.setdefault(source, {"authentic": 0, "manipulated": 0})[entry["label"]] += 1
    total = {label: sum(values[label] for values in counts.values()) for label in ("authentic", "manipulated")}
    for source, values in counts.items():
        if not all(values[label] for label in total):
            raise ValueError(f"source balance requires both classes: {source}")
        if any(values[label] > total[label] * 0.4 for label in total):
            raise ValueError(f"source balance exceeds 40 percent: {source} {values}")


def _entry(out: Path, entries: list[dict], image: Image.Image, mask: Image.Image, *, ident: str, label: str, family: str, source: str, source_exif: bytes, params: dict, quality: int = 90) -> None:
    source_label = _source_label(Path(source))
    if label == "authentic" and source_label != "authentic":
        raise ValueError(f"authentic output cannot derive from {source_label} source: {source}")
    image_path = out / f"{ident}.jpg"
    mask_path = out / f"{ident}_mask.png"
    variant = len(entries)
    _save_image(image, image_path, quality, _synthetic_exif(source_exif, image, variant), variant)
    mask.save(mask_path, format="PNG", optimize=False)
    sidecar = {
        "id": ident, "label": label, "family": family, "params": params,
        "source_image": source, "source_label": source_label,
        "seed": params.get("seed", SEED),
    }
    (out / f"{ident}.json").write_text(json.dumps(sidecar, sort_keys=True, indent=2) + "\n")
    entries.append({
        "id": ident,
        "path": f"data/corpus/synthetic/{image_path.name}",
        "mask": f"data/corpus/synthetic/{mask_path.name}",
        "label": label,
        "family": family,
        "source_image": source,
        "source_label": source_label,
        "params": params,
    })


def _load_seed_images(seed_dir: Path) -> list[tuple[Path, Image.Image, bytes]]:
    paths = sorted(p for p in seed_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"})
    if seed_dir.resolve() == (ROOT / "data/samples").resolve():
        paths.extend(_real_camera_paths())
    if not paths:
        raise SystemExit(f"no seed images found under {seed_dir}")
    images = []
    for path in paths:
        with Image.open(path) as image:
            exif = image.getexif()
            if not exif:
                exif = Image.Exif()
                exif[0x010E] = f"synthetic corpus source: {path.name}"
            images.append((path, ImageOps.fit(image.convert("RGB"), IMAGE_SIZE, method=Image.Resampling.LANCZOS), exif.tobytes()))
    return images


def generate(seed: int, out: Path, seed_dir: Path) -> dict:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    for path in out.glob("synthetic_*.jpg"):
        path.unlink()
    for path in out.glob("synthetic_*.json"):
        path.unlink()
    for path in out.glob("synthetic_*_mask.png"):
        path.unlink()
    index_path = out / "index.json"
    if index_path.exists():
        index_path.unlink()

    seeds = _load_seed_images(seed_dir)
    authentic_seeds = [seed for seed in seeds if _source_label(seed[0]) == "authentic"]
    if len(authentic_seeds) < 3:
        raise SystemExit(
            "source balance requires at least 3 genuine authentic sources; "
            f"found {len(authentic_seeds)}"
        )
    entries: list[dict] = []
    zero = lambda size: Image.new("L", size, 0)

    # 32 authentic JPEGs, balanced over the four encoder qualities.
    for n in range(32):
        host_path, host, source_exif = authentic_seeds[n % len(authentic_seeds)]
        quality = JPEG_QUALITIES[n % len(JPEG_QUALITIES)]
        jitter = np_rng.normal(0, 0.35, (IMAGE_SIZE[1], IMAGE_SIZE[0], 3))
        source = np.asarray(host, dtype=np.float32) + jitter
        image = Image.fromarray(np.clip(source, 0, 255).astype(np.uint8), "RGB")
        _entry(out, entries, image, zero(image.size), ident=f"synthetic_authentic_recompress_{n:03d}", label="authentic", family="authentic_recompress", source=_source_name(host_path), source_exif=source_exif, params={"quality": quality, "seed": seed}, quality=quality)

    # 14 splices from a different sample source, with donor recompression.
    for n in range(14):
        host_path, host, source_exif = authentic_seeds[n % len(authentic_seeds)]
        donor_path, donor, _ = seeds[(n + 1) % len(seeds)]
        fraction = (0.05, 0.10, 0.20)[n % 3]
        host_quality = JPEG_QUALITIES[n % 4]
        donor_quality = JPEG_QUALITIES[(n + 1) % 4]
        box = _region(rng, IMAGE_SIZE, fraction)
        patch = _decode(_jpeg_bytes(donor, donor_quality)).crop(box)
        image = host.copy()
        feathered = n % 2 == 1
        if feathered:
            alpha = Image.new("L", patch.size, 255).filter(ImageFilter.GaussianBlur(3))
            image.paste(patch, box[:2], alpha)
        else:
            image.paste(patch, box[:2])
        _entry(out, entries, image, _mask(image.size, [box]), ident=f"synthetic_splice_{n:03d}", label="manipulated", family="splice", source=_source_name(host_path), source_exif=source_exif, params={"region": list(box), "host_quality": host_quality, "donor_quality": donor_quality, "feathered": feathered, "donor_source": _source_name(donor_path), "seed": seed}, quality=host_quality)

    # 14 copy-moves. Ground truth includes both the source and destination.
    for n in range(14):
        host_path, host, source_exif = authentic_seeds[n % len(authentic_seeds)]
        fraction = (0.05, 0.10, 0.20)[n % 3]
        source_box = _region(rng, IMAGE_SIZE, fraction)
        dest_box = _different_region(rng, IMAGE_SIZE, fraction, source_box)
        patch = host.crop(source_box)
        scale = (1.0, 1.1)[n % 2]
        rotation = (0, 5)[n % 2]
        if scale != 1.0:
            patch = patch.resize((max(1, int(patch.width * scale)), max(1, int(patch.height * scale))), Image.Resampling.BICUBIC)
        if rotation:
            patch = patch.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
        target = (dest_box[0], dest_box[1], dest_box[0] + patch.width, dest_box[1] + patch.height)
        target = (min(target[0], IMAGE_SIZE[0] - patch.width), min(target[1], IMAGE_SIZE[1] - patch.height), min(target[0], IMAGE_SIZE[0] - patch.width) + patch.width, min(target[1], IMAGE_SIZE[1] - patch.height) + patch.height)
        image = host.copy()
        feathered = n % 3 == 2
        if feathered:
            image.paste(patch, target[:2], Image.new("L", patch.size, 255).filter(ImageFilter.GaussianBlur(2)))
        else:
            image.paste(patch, target[:2])
        _entry(out, entries, image, _mask(image.size, [source_box, target]), ident=f"synthetic_copy_move_{n:03d}", label="manipulated", family="copy_move", source=_source_name(host_path), source_exif=source_exif, params={"source_region": list(source_box), "destination_region": list(target), "scale": scale, "rotation": rotation, "feathered": feathered, "seed": seed})

    for family, count, shifted in [("double_compress_aligned", 10, False), ("double_compress_shifted", 10, True)]:
        for n in range(count):
            host_path, host, source_exif = authentic_seeds[n % len(authentic_seeds)]
            q1, q2 = JPEG_QUALITIES[n % 4], JPEG_QUALITIES[(n + 2) % 4]
            first = _decode(_jpeg_bytes(host, q1))
            dx, dy = ((1, 0), (4, 4), (7, 3))[n % 3] if shifted else (0, 0)
            image = first
            if shifted:
                image = first.crop((dx, dy, IMAGE_SIZE[0], IMAGE_SIZE[1])).resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            image = _decode(_jpeg_bytes(image, q2))
            _entry(out, entries, image, Image.new("L", image.size, 255), ident=f"synthetic_{family}_{n:03d}", label="manipulated", family=family, source=_source_name(host_path), source_exif=source_exif, params={"quality_first": q1, "quality_second": q2, "shift": [dx, dy], "seed": seed})

    for n in range(12):
        host_path, host, source_exif = authentic_seeds[n % len(authentic_seeds)]
        box = _region(rng, IMAGE_SIZE, (0.05, 0.10, 0.20)[n % 3])
        patch = host.crop(box)
        operation = ("blur", "brightness", "clone", "median")[n % 4]
        if operation == "blur":
            patch = patch.filter(ImageFilter.GaussianBlur(4))
        elif operation == "brightness":
            patch = ImageEnhance.Brightness(patch).enhance(0.65 if n % 2 else 1.35)
        elif operation == "clone":
            patch = host.crop(_region(rng, IMAGE_SIZE, 0.05)).resize(patch.size, Image.Resampling.BICUBIC)
        else:
            patch = patch.filter(ImageFilter.MedianFilter(5))
        image = host.copy()
        image.paste(patch, box[:2])
        _entry(out, entries, image, _mask(image.size, [box]), ident=f"synthetic_local_retouch_{n:03d}", label="manipulated", family="local_retouch", source=_source_name(host_path), source_exif=source_exif, params={"region": list(box), "operation": operation, "seed": seed})

    for n in range(8):
        host_path, host, source_exif = authentic_seeds[n % len(authentic_seeds)]
        ratio = (0.5, 0.75)[n % 2]
        image = host.resize((int(host.width * ratio), int(host.height * ratio)), Image.Resampling.LANCZOS)
        _entry(out, entries, image, zero(image.size), ident=f"synthetic_resize_then_save_{n:03d}", label="authentic", family="resize_then_save", source=_source_name(host_path), source_exif=source_exif, params={"scale": ratio, "seed": seed})

    _assert_source_balance(entries)
    index = {"seed": seed, "generated_at": GENERATED_AT, "entries": entries}
    index_path.write_text(json.dumps(index, sort_keys=True, indent=2) + "\n")
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=Path("data/corpus/synthetic"))
    parser.add_argument("--seed-images", type=Path, default=Path("data/samples"))
    args = parser.parse_args()
    index = generate(args.seed, args.out, args.seed_images)
    counts = {label: sum(entry["label"] == label for entry in index["entries"]) for label in ("authentic", "manipulated")}
    print(f"generated {len(index['entries'])} entries: {counts['authentic']} authentic, {counts['manipulated']} manipulated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
