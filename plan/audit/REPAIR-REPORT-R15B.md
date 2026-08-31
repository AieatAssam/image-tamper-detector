# Repair report R15B: IMD2020 inpainting axis

Date: 2026-08-31
Status: **Rejected by the metadata shortcut gate; no manifest ingestion.**

## Outcome

The IMD2020 generative-inpainting candidate does not satisfy the corpus
acceptance gate. A fixed-seed sample of 200 inpainted images, their 200 real
counterparts, and their 200 masks gives a metadata-only held-out AUC of
**0.9917 +/- 0.0084** and pooled AUC of **0.9975 +/- 0.0025**. The selected
feature is `exif_present`, with real counterparts carrying EXIF and the
inpainting outputs not carrying it. The candidate is therefore not ingested
into `data/corpus/MANIFEST.yaml`.

No calibration, detector benchmark, or `data/samples/` file changed. The
downloaded archives and extracted image bytes remain local under the ignored
`data/corpus/real/r15b-imd2020-inpainting-download/` directory.

## 1. D7b preflight and acquisition

Before the first byte:

```text
git check-ignore -v data/corpus/real/r15b-imd2020-inpainting-download
# .gitignore:8:data/corpus/real/* data/corpus/real/r15b-imd2020-inpainting-download
```

The two requested inpainting files returned HTTP 200 with these sizes:

| File | Bytes | SHA-256 |
|---|---:|---|
| `IMD2020_Generative_Image_Inpainting_yu2018_01.zip` | 1,722,669,427 | `4cefd85107326757c0ec3e6db4eb573ad2fb5cd4aa324e3d4a49fcb51dcd9aa9` |
| `IMD2020_Generative_Image_Inpainting_yu2018_mask.zip` | 126,008,215 | `08ab044b56930066955853a14f88710023ee03d092f1bb4f7af1db93000a4573` |

Archive 01 contains only inpainting outputs, so the same-dataset real Part01
archive was required to construct the requested pairs. It was fetched as the
implied real-side Part01, not as an inpainting Part02-07 archive:

| File | Bytes | SHA-256 |
|---|---:|---|
| `IMD2020_real_01.zip` | 1,960,999,112 | `158e98cf8923b9eff7b4b3ff44d5b303e6b19dd30b1006d285a664698ce7add1` |

No inpainting Parts 02-07 or real Parts 02-04 were requested.

The archive inventory passed `zipfile.testzip()` during extraction:

| Archive | Image members |
|---|---:|
| Inpainting Part01 | 5,000 |
| Real Part01 | 9,102 |
| Masks | 35,000 |

Exact basename matching found 1,339 complete inpainting/real/mask triples
across 249 camera groups. Camera groups came from the real archive's own
`<make>/<model>` directory paths. No counterpart was inferred from image
content or from the unrelated opaque IDs in the classical IMD2020 archive.

## 2. Reproducible candidate sample

The new fetcher is `scripts/fetch_imd2020_inpainting.py`. It selects with seed
`20260828`, takes 200 pairs, and limits the sample to one pair per camera
group. The selected pair is represented by the exact common basename; the
mask is the exact `_mask.jpg` basename. The candidate JSONL used for the gate
was generated with:

```text
.venv/bin/python scripts/fetch_imd2020_inpainting.py \
  --sample --candidate /tmp/r15b-imd2020-inpaint.jsonl
```

The result was 200 inpainted positives and 200 authentic negatives. The
deterministic group split contains 140 pairs / 280 rows in train and 60 pairs
/ 120 rows in held-out. Each candidate row records axis
`imd2020_inpaint`, its paired source image, and its split. Since the candidate
failed the gate, this selection was intentionally not copied into the
manifest.

## 3. Acceptance gate

The required gate was run before any manifest edit:

```text
.venv/bin/python scripts/check_format_shortcut.py \
  --manifest /tmp/r15b-imd2020-inpaint.jsonl --check \
  --out /tmp/r15b-format.json
# exit 1
```

| Feature group | Held-out AUC +/- SE | Pooled AUC +/- SE | Selected feature | Gate |
|---|---:|---:|---|---|
| all | 0.9917 +/- 0.0084 | 0.9975 +/- 0.0025 | `exif_present` | **FAIL** |
| format | 0.5000 +/- 0.0529 | 0.5000 +/- 0.0289 | constant JPEG | pass |
| dimensions | 0.5917 +/- 0.0518 | 0.5625 +/- 0.0286 | `height` | fail |
| file size | 0.7250 +/- 0.0462 | 0.7625 +/- 0.0238 | `file_size` | fail |
| EXIF | 0.9917 +/- 0.0084 | 0.9975 +/- 0.0025 | `exif_present` | fail |

The corpus gate allows held-out metadata AUC no higher than `0.55`. The
format-only result is near chance, but the full required feature set is not.
The measured failure is exactly the kind of save-pipeline shortcut this axis
was intended to rule out.

## 4. Licence, citation, and limitation

The official IMD2020 download page publishes no explicit redistribution
license for these image archives. Under plan decision D7b, the files are
local-only in a gitignored directory; no image or mask bytes are committed.

Citation recorded for the candidate:

> Novozamsky, Mahdian, and Saic, “IMD2020: A Large-Scale Annotated Dataset
> Tailored for Detecting Manipulated Images,” IEEE WACV Workshops 2020.

The intended `imd2020_inpaint` question is narrower than the
`sd35_flux`/`synthbuster` question: it measures detection of **AI-edited
regions**, not detection of wholly generated images. Yu et al. 2018 is a GAN
inpainter, not a diffusion model. This candidate could not answer even that
narrow question validly because it failed metadata parity, so no detector AUC
is reported.
