# Repair report — round 2 — 2026-08-29

Round-2 statuses are `fixed` for G1, G2, G3, G5, G7, and G8; `partially-fixed` for G4 and G6. No finding is reported as blocked: the remaining G6 shortfall is documented with the exact evidence available.

Final gates:

- `.venv/bin/python -m pytest backend/tests -q`: `54 passed, 1 warning in 139.28s`.
- `.venv/bin/python scripts/benchmark.py --out /tmp/post-r2.json --corpus all`: exit `0`; wrote JSON and Markdown artifacts.
- `.venv/bin/python plan/validate.py`: passed; `15` stages and `219` shell snippets checked.
- Final benchmark/reference comparison: `reference_matches_final True`.

## G1 — fixed

Changed `scripts/make_corpus.py`, `data/corpus/synthetic/index.json` and its 100 JSON sidecars, `backend/tests/test_corpus.py`, and `docs/corpus.md`. Authentic negative families now derive only from `landscape_original.jpg`. Every row also records the parent `source_label`; known forgeries and AI images remain available to manipulation families without being asserted authentic.

The required evidence command before the repair reported 40 authentic rows from four sources: 10 original, 10 copy-paste, and 10 from each GPT-4o receipt. The final provenance command reported:

```text
source_label_counts Counter({'authentic': 57, 'ai_generated': 30, 'known_forgery': 13})
('landscape_original.jpg', 'authentic', 'authentic'): 40
('landscape_original.jpg', 'manipulated', 'authentic'): 17
('gpt-4o-generated-receipt-01.png', 'manipulated', 'ai_generated'): 17
('gpt-4o-generated-receipt-02.png', 'manipulated', 'ai_generated'): 13
('landscape_copy_paste.jpg', 'manipulated', 'known_forgery'): 13
provenance_assertions=pass
```

The 30 bad authentic rows are gone; they were not blindly relabelled as manipulated.

## G2 — fixed

Changed `scripts/calibrate.py`, regenerated `backend/app/analysis/calibration.json`, and updated `docs/calibration.md`. `backend/app/analysis/spectral.py` was not changed.

Before: the audit benchmark reported spectral synthetic AUC `0.575` and calibration weight `0.0`. After the corpus correction and re-fit, `/tmp/post-r2.json` reports spectral synthetic AUC `0.77875` and `calibration.json` reports spectral weight `0.14700575681469943`. Real spectral AUC is `None` because the available real rows contain only one class; that is represented as unavailable rather than used as validation evidence.

The unchanged `spectral.py` plus the corrected corpus and changed result establish that the regression was a corpus consequence, not a spectral behavior change. CFA, spectral, and PRNU remain available to the runtime ensemble; only their invalid synthetic fused-verdict evidence is excluded from the synthetic benchmark.

## G3 — fixed

Changed `scripts/calibrate.py`, regenerated `backend/app/analysis/calibration.json`, and updated `docs/calibration.md`. The logistic fit now projects weights to the catalog's non-negative direction instead of allowing fusion to invert detector meaning.

Before, the evidence command reported `entropy=-0.3725150686699604` and `exif=-0.25706333390346603`. After, the actual weight dump reported:

```text
negative_weights {}
entropy 0.0
exif 0.0
```

Anti-correlated evidence is therefore dropped at zero; it is not silently used as an authenticity signal.

## G4 — partially-fixed

Changed `scripts/benchmark.py`, `plan/reference/api-contract.yaml`, `plan/stages/S05-corpus-and-benchmark.yaml`, `backend/tests/test_ghosts.py`, and `backend/tests/test_copy_move.py`.

Before and after deterministic benchmark duration buckets are `[0.0, 500.0]`; every applicable detector still reports the shared `500.0 ms` bucket. That remains intentionally coarse so S05 output is byte-identical and is now explicitly excluded from performance gates.

The performance assertions now measure wall-clock detector runs directly. The actual command reported:

```text
jpeg_ghosts 0.467s applicable
copy_move 0.076s applicable
```

Those are below the S06 `8s` and S07 `15s` caps. The finding remains `partially-fixed` because the deterministic field itself is not a useful performance ranking signal.

## G5 — fixed

Changed `.python-version` to the already-targeted Python `3.14.7`; Dockerfile and CI were already on 3.14. The 3.14 environment was rebuilt and the pinned stack installed and imported successfully; no D1 failure amendment was needed.

Before: `.venv/bin/python --version` was `Python 3.13.13` while `.python-version` was `3.14.7`. After, the actual commands reported `.venv` `Python 3.14.7`, `.python-version=3.14.7`, all required imports passed, and `pip check` reported `No broken requirements found.`

## G6 — partially-fixed

Changed `data/corpus/MANIFEST.yaml`, `scripts/fetch_corpus.py`, and `docs/corpus.md`. Nine real AI Commons entries were downloaded, verified for bytes, SHA-256, decoded format/size, label, and license copied from `extmetadata.LicenseShortName`. The fetch verifier also rejects checksum, byte-count, license, tracking-URL, and axis-criterion failures.

Before: the manifest evidence was `real_camera: 1`, `real_ai: 0`, `real_c2pa_signed: 0` against requirements `12`, `12`, and `2`. After, the actual manifest count and verifier reported:

```text
Counter({'real_ai': 9})
verified ai_dalle_001 ... verified ai_dalle_009
9 manifest entries verified
```

The remaining shortfall is `12` strict camera entries, `3` AI entries, and `2` signed-C2PA entries. Ten downloaded camera candidates were rejected after the strict EXIF command found Make/Model but no `PixelXDimension`; two downloaded C2PA candidates had invalid rather than valid trust/expiry state and were not recorded as the required pair. Subsequent direct Wikimedia upload fetches returned HTTP `429` with `Retry-After: 600`.

Commands actually tried included the S05 API query and direct upload fetch recorded in `docs/corpus.md`, the strict camera inspection:

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

and `.venv/bin/python scripts/fetch_corpus.py --check`. No aspirational manifest entries were added. S05/S08 remain failed in `plan/STATUS.yaml` until the missing qualifying bytes and axis evidence exist.

## G7 — fixed

Changed `scripts/benchmark.py` to omit the fabricated timestamp, made calibration timestamps use the actual UTC clock, updated `plan/reference/api-contract.yaml` and the stale S05 key-path check, and regenerated the benchmark artifacts.

Before, `rg` found the literal `"generated_at": "2026-08-28T12:00:00Z"` in benchmark artifacts. After, the actual check returned `placeholder_generated_at=absent`; current deterministic benchmark JSON has no `generated_at` field, while `calibration.json` records a real generated time.

## G8 — fixed

Regenerated `benchmarks/REFERENCE.json` and its Markdown companion from the final benchmark, and synchronized `post-S10`/`post-S12` artifacts with the current schema. The reference now includes the learned detector and the deterministic timing contract.

Before: the reference had 11 detectors, no learned entry, zero duration fields, and stale pre-repair metrics. After: the actual comparison reported `reference_matches_final True`, `n_images 109`, `detectors 12`, `reference_has_learned True`, and duration buckets `[0.0, 500.0]`.

## Status

`plan/STATUS.yaml` now marks S00 passed after the successful Python 3.14 rebuild. S05 and S08 remain failed for the verified corpus shortfall. S10 is failed because the non-negative fit is present but grouped held-out AUC is `null` without an authentic real-camera class. No commit was created.
