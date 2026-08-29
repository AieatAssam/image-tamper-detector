#!/usr/bin/env python3
"""Generate the deterministic, processing-history image corpus for S05."""

from __future__ import annotations

import argparse
import json
import random
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

SEED = 20260828
GENERATED_AT = "2026-08-28T12:00:00Z"
IMAGE_SIZE = (768, 512)
JPEG_QUALITIES = (70, 80, 90, 95)


def _jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    out = BytesIO()
    image.convert("RGB").save(
        out, format="JPEG", quality=quality, optimize=False,
        progressive=False, subsampling=0
    )
    return out.getvalue()


def _decode(data: bytes) -> Image.Image:
    with Image.open(BytesIO(data)) as image:
        return image.convert("RGB").copy()


def _save_image(image: Image.Image, path: Path, quality: int = 90) -> None:
    path.write_bytes(_jpeg_bytes(image, quality))


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


def _entry(out: Path, entries: list[dict], image: Image.Image, mask: Image.Image, *, ident: str, label: str, family: str, source: str, params: dict, quality: int = 90) -> None:
    image_path = out / f"{ident}.jpg"
    mask_path = out / f"{ident}_mask.png"
    _save_image(image, image_path, quality)
    mask.save(mask_path, format="PNG", optimize=False)
    sidecar = {
        "id": ident, "label": label, "family": family, "params": params,
        "source_image": source, "seed": params.get("seed", SEED),
    }
    (out / f"{ident}.json").write_text(json.dumps(sidecar, sort_keys=True, indent=2) + "\n")
    entries.append({
        "id": ident,
        "path": f"data/corpus/synthetic/{image_path.name}",
        "mask": f"data/corpus/synthetic/{mask_path.name}",
        "label": label,
        "family": family,
        "source_image": source,
        "params": params,
    })


def _load_seed_images(seed_dir: Path) -> list[tuple[Path, Image.Image]]:
    paths = sorted(p for p in seed_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"})
    if not paths:
        raise SystemExit(f"no seed images found under {seed_dir}")
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append((path, ImageOps.fit(image.convert("RGB"), IMAGE_SIZE, method=Image.Resampling.LANCZOS)))
    return images


def _donor(seed_dir: Path, host: Image.Image) -> tuple[str, Image.Image]:
    candidates = sorted((seed_dir.parent / "tampered").glob("*.png"))
    if candidates:
        with Image.open(candidates[0]) as image:
            return str(candidates[0]), ImageOps.fit(image.convert("RGB"), IMAGE_SIZE, method=Image.Resampling.LANCZOS)
    # This fallback keeps the generator usable with a user-supplied seed folder.
    return str(seed_dir / "derived_donor_from_seed"), ImageOps.mirror(host)


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
    host_path, host = seeds[0]
    donor_path, donor = _donor(seed_dir, host)
    entries: list[dict] = []
    zero = lambda size: Image.new("L", size, 0)

    # 32 authentic JPEGs, balanced over the four encoder qualities.
    for n in range(32):
        quality = JPEG_QUALITIES[n % len(JPEG_QUALITIES)]
        jitter = np_rng.normal(0, 0.35, (IMAGE_SIZE[1], IMAGE_SIZE[0], 3))
        source = np.asarray(host, dtype=np.float32) + jitter
        image = Image.fromarray(np.clip(source, 0, 255).astype(np.uint8), "RGB")
        _entry(out, entries, image, zero(image.size), ident=f"synthetic_authentic_recompress_{n:03d}", label="authentic", family="authentic_recompress", source=str(host_path), params={"quality": quality, "seed": seed}, quality=quality)

    # 14 splices from a different sample source, with donor recompression.
    for n in range(14):
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
        _entry(out, entries, image, _mask(image.size, [box]), ident=f"synthetic_splice_{n:03d}", label="manipulated", family="splice", source=str(host_path), params={"region": list(box), "host_quality": host_quality, "donor_quality": donor_quality, "feathered": feathered, "donor_source": donor_path, "seed": seed}, quality=host_quality)

    # 14 copy-moves. Ground truth includes both the source and destination.
    for n in range(14):
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
        _entry(out, entries, image, _mask(image.size, [source_box, target]), ident=f"synthetic_copy_move_{n:03d}", label="manipulated", family="copy_move", source=str(host_path), params={"source_region": list(source_box), "destination_region": list(target), "scale": scale, "rotation": rotation, "feathered": feathered, "seed": seed})

    for family, count, shifted in [("double_compress_aligned", 10, False), ("double_compress_shifted", 10, True)]:
        for n in range(count):
            q1, q2 = JPEG_QUALITIES[n % 4], JPEG_QUALITIES[(n + 2) % 4]
            first = _decode(_jpeg_bytes(host, q1))
            dx, dy = ((1, 0), (4, 4), (7, 3))[n % 3] if shifted else (0, 0)
            image = first
            if shifted:
                image = first.crop((dx, dy, IMAGE_SIZE[0], IMAGE_SIZE[1])).resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            image = _decode(_jpeg_bytes(image, q2))
            _entry(out, entries, image, Image.new("L", image.size, 255), ident=f"synthetic_{family}_{n:03d}", label="manipulated", family=family, source=str(host_path), params={"quality_first": q1, "quality_second": q2, "shift": [dx, dy], "seed": seed})

    for n in range(12):
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
        _entry(out, entries, image, _mask(image.size, [box]), ident=f"synthetic_local_retouch_{n:03d}", label="manipulated", family="local_retouch", source=str(host_path), params={"region": list(box), "operation": operation, "seed": seed})

    for n in range(8):
        ratio = (0.5, 0.75)[n % 2]
        image = host.resize((int(host.width * ratio), int(host.height * ratio)), Image.Resampling.LANCZOS)
        _entry(out, entries, image, zero(image.size), ident=f"synthetic_resize_then_save_{n:03d}", label="authentic", family="resize_then_save", source=str(host_path), params={"scale": ratio, "seed": seed})

    index = {"seed": seed, "generated_at": GENERATED_AT, "entries": entries}
    index_path.write_text(json.dumps(index, sort_keys=True, indent=2) + "\n")
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=Path("data/corpus/synthetic"))
    parser.add_argument("--seed-images", type=Path, default=Path("data/samples/original"))
    args = parser.parse_args()
    index = generate(args.seed, args.out, args.seed_images)
    counts = {label: sum(entry["label"] == label for entry in index["entries"]) for label in ("authentic", "manipulated")}
    print(f"generated {len(index['entries'])} entries: {counts['authentic']} authentic, {counts['manipulated']} manipulated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
