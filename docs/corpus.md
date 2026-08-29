# Corpus and benchmark

S05 has two distinct corpus roles:

- `data/corpus/synthetic/` measures processing-history cues: authentic recompression, splices, copy-move, double JPEG compression, local retouching, and resized authentic content. Manipulation families may use the four existing files under `data/samples/`, but both negative families (`authentic_recompress` and `resize_then_save`) derive only from the genuinely authentic `landscape_original.jpg`. Each index entry and sidecar records `source_label` (`authentic`, `known_forgery`, or `ai_generated`) so parent provenance cannot be confused with the family-specific `label`. Source EXIF is preserved or a neutral source description is added to each JPEG. `index.json` and JSON sidecars are reviewable; image and mask bytes are reproducible and ignored by Git.
- `data/corpus/MANIFEST.yaml` defines optional real-image downloads. `scripts/fetch_corpus.py` verifies both SHA-256 and byte count, and refuses mismatches.

IMD2020 is an additional local-only corpus candidate from Novozamsky, Mahdian,
and Saic, “IMD2020: A Large-Scale Annotated Dataset Tailored for Detecting
Manipulated Images,” IEEE WACV Workshops 2020. The archive is available at
`https://staff.utia.cas.cz/novozada/db/IMD2020.zip`; it is downloaded only to
the gitignored `data/corpus/imd2020/` directory. The verified archive is
`592836398` bytes with SHA-256
`a1497d7cc21a20ee412c0758f1450ee87e35bf3da9aac114f044e8dedaec382f` and
contains `2010` manipulated images, `2010` masks, and `414` corresponding real
counterparts across `414` source groups. The publication does not state an
explicit redistribution license, so this repository commits no IMD2020 image
bytes. Run `.venv/bin/python scripts/fetch_imd2020.py --check` to verify the
local archive and all image/mask triples.

The archive's real-life split does not include machine-readable manipulation
type metadata: its source directories are opaque IDs and its files are image
and mask files only. The fixed-seed sampler therefore stratifies by source
directory, not by an invented operation taxonomy. With seed `20260828`, it
selects 200 source directories, one manipulated image per source, its mask,
and the source's `_orig.jpg` counterpart. The manifest records 400 rows with
`axis: imd2020`, `source_group`, `split`, SHA-256, and byte size; the image and
mask paths remain under the gitignored local archive directory.

The synthetic corpus cannot validate sensor provenance. A generator can faithfully synthesise PROCESSING HISTORY (splices, recompression, copy-move, quality changes) but CANNOT synthesise SENSOR PROVENANCE. Re-splicing one Unsplash JPEG creates no genuine Bayer interpolation structure and no genuine sensor noise. Therefore cfa_periodicity, spectral_peaks and the noise-residual detector MUST be validated against real images, never against generated splices.

For synthetic benchmark fusion, those three provenance detectors are omitted from the fused verdict because their synthetic scores are not valid evidence. Their calibrated weights remain available to the runtime ensemble for genuine uploaded images; this keeps validation scope separate from ensemble availability.

The manifest currently verifies `12` `real_camera`, `12` `real_ai`, and `2` `real_c2pa_signed` entries against requirements of `12`, `12`, and `2`. All 12 camera entries now meet the `strict` evidence rule: EXIF Make/Model is present and nested `PixelXDimension` equals the decoded width. The prior 11 entries were mislabeled `relaxed` because the checker read only Pillow's root IFD; the bytes contain `PixelXDimension` in EXIF sub-IFD `0x8769`. `cam_013` is the newly sourced Nikon D70 entry; its nested EXIF `PixelXDimension` is `1200`, equal to its decoded width of `1200`. The 12 AI entries are Commons files whose API metadata names DALL-E, Midjourney v4, Stable Diffusion 3.5 Large, or xAI Aurora; their bytes, SHA-256, decoded format and size, label, and license were checked before recording them. The two C2PA entries are pinned `contentauth/c2pa-rs` test fixtures under its recorded `MIT OR Apache-2.0` package license. `c2pa_001` parses with pinned `c2pa-python 0.37.8` as `validation_state=Valid`; `c2pa_002` parses as `validation_state=Invalid` with `assertion.dataHash.mismatch`. Real-camera and C2PA metrics are available for the downloaded entries; synthetic results are still valid Tier A only for the detector families they validate.

The strict real-camera verification command used for the downloaded candidates was:

```bash
.venv/bin/python - <<'PY'
import glob
from PIL import Image
for path in glob.glob('/tmp/itd-camera-bytes/cam_*.jpg'):
    with Image.open(path) as image:
        exif = image.getexif()
        print(path, image.format, image.size, exif.get(0x010F), exif.get(0x0110), exif.get(0xA002))
PY
```

The discovery and download commands were the S05 Wikimedia API query and direct upload URL fetch, for example:

```bash
curl -sS --fail --max-time 30 -H 'User-Agent: image-tamper-detector-corpus/1.0' \
  'https://commons.wikimedia.org/w/api.php?action=query&generator=categorymembers&gcmtitle=Category%3APhotographs%20taken%20with%20Google%20Pixel%203a%20by%20Gzen92&gcmlimit=20&gcmtype=file&prop=imageinfo&iiprop=url%7Csize%7Cmime%7Cextmetadata&format=json'
curl -sS --fail --location --max-time 90 -H 'User-Agent: image-tamper-detector-corpus/1.0' \
  'https://upload.wikimedia.org/wikipedia/commons/9/9d/Goutum.jpg'
```

The first command returned candidate metadata. The prior direct upload retry returned HTTP 429 from Wikimedia's upload host with `Retry-After: 600`; this round did not sleep through that interval. A later category metadata query returned a strict candidate, and its direct upload fetch returned `197670` bytes. The fetcher reads EXIF sub-IFD `0x8769` so nested `0xA002` is checked rather than rejected as absent.

The C2PA fixture checks were run before recording the entries:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from c2pa import Reader
for name in ("C.jpg", "XCA.jpg"):
    with Reader(Path("/tmp/itd-c2pa-cache") / name) as reader:
        store = json.loads(reader.json())
    print(name, store["validation_state"], store.get("validation_status"))
PY
```

This printed `C.jpg Valid` and `XCA.jpg Invalid`; the invalid fixture included `assertion.dataHash.mismatch`. The fixture source was pinned to commit `be7f5ea22b385ee1af6c327906ba002747687628`; its `make_test_images/Cargo.toml` records `MIT OR Apache-2.0`, and the generator source carries the corresponding Apache/MIT notices.

## Commands

```bash
pyenv local 3.14.7
.venv/bin/python scripts/make_corpus.py --seed 20260828 --out data/corpus/synthetic --seed-images data/samples
.venv/bin/python scripts/fetch_corpus.py
.venv/bin/python scripts/fetch_corpus.py --check
.venv/bin/python scripts/benchmark.py --out /tmp/bench.json --corpus all
```

Use `--corpus synthetic` for an offline benchmark. Use `--detectors ela,prnu,entropy` to select a subset. The benchmark writes a JSON contract and a matching Markdown table. Its output omits timestamps and uses deterministic timing buckets, so unchanged runs can be compared byte-for-byte; S06/S07 wall-clock caps are tested separately.
