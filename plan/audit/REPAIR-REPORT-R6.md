# Round 6 repair report - 2026-08-29

No commit was created. The existing synthetic corpus and the prior
manifest-backed real corpus remain in place.

## K1 - fixed: IMD2020 fetch and triple verification

Changed:

- `.gitignore`
- `scripts/fetch_imd2020.py`

`data/corpus/imd2020/` is ignored before the download. The archive was fetched
from the verified URL, extracted locally, and checked without adding image
bytes or manifest rows. The fetcher now checks the archive hash and verifies
each manipulated image has exactly one mask and one real counterpart in its
source group.

Before/after:

| measure | before | after |
|---|---:|---:|
| local IMD2020 archive | absent | 592836398 bytes |
| complete manipulated/mask/real triples | 0 | 2010 |
| source groups with real counterparts | 0 | 414 |

Commands and results:

```text
git check-ignore -v data/corpus/imd2020/ data/corpus/imd2020/IMD2020.zip data/corpus/imd2020/IMD2020.zip.part
.gitignore:9:data/corpus/imd2020/  data/corpus/imd2020/
.gitignore:9:data/corpus/imd2020/  data/corpus/imd2020/IMD2020.zip
.gitignore:9:data/corpus/imd2020/  data/corpus/imd2020/IMD2020.zip.part

shasum -a 256 data/corpus/imd2020/IMD2020.zip
a1497d7cc21a20ee412c0758f1450ee87e35bf3da9aac114f044e8dedaec382f  data/corpus/imd2020/IMD2020.zip

.venv/bin/python scripts/fetch_imd2020.py --check
{"archive_bytes": 592836398, "archive_sha256": "a1497d7cc21a20ee412c0758f1450ee87e35bf3da9aac114f044e8dedaec382f", "complete_triples": 2010, "image_count": 4434, "manipulated_images": 2010, "masks": 2010, "real_counterparts": 414, "source_group_count": 414, "source_groups": 414, "suffixes": {".jpg": 2227, ".png": 2207}}
```

`data/corpus/MANIFEST.yaml` is unchanged because K4 did not establish an
admissible fixed-seed sample. Recording unselected rows would violate K4's
sample-size and type-balance requirement.

## K2 - skipped: secondary corpus

No secondary corpus was used. IMD2020 was sufficient for the attempted data
path, and no additional source with unclear terms was introduced.

| measure | before | after |
|---|---:|---:|
| secondary external corpora used | 0 | 0 |

## K3 - fixed: licensing and byte handling

Changed:

- `.gitignore`
- `docs/corpus.md`
- `README.md`

The citation for Novozamsky, Mahdian, and Saic, “IMD2020: A Large-Scale
Annotated Dataset Tailored for Detecting Manipulated Images,” IEEE WACV
Workshops 2020, is recorded in both requested documents. The archive URL,
size, SHA-256, local-only path, and no-explicit-license limitation are also
recorded. The manifest has no IMD rows because no sample was admissible.

Before/after citation mentions, measured against `HEAD`:

```text
README.md: 0 -> 3
docs/corpus.md: 0 -> 6
```

The required working-tree byte check was empty:

```text
git status --porcelain | grep -E '\.(jpg|jpeg|png|zip)$'
# no output
```

## K4 - blocked: fixed-seed stratified sample

Changed:

- `scripts/sample_imd2020.py`

The sampler preserves each manipulated image, its mask, and its real
counterpart, and has fixed seed `20260828`, default count `400`, balanced
allocation, deterministic ordering, and a self-test. It refuses to infer a
manipulation type from an opaque source directory.

The exact missing fact is a source-backed mapping from the IMD2020 real-life
file IDs to manipulation types. The extracted archive has 414 opaque source
directories, image files, and mask files, but no metadata files or type-bearing
paths. The source directory names identify paired originals, not manipulation
types. Masks establish manipulated regions, not the operation that produced
them.

Commands tried:

```text
find data/corpus/imd2020/extracted -type f ! -iname '*.jpg' ! -iname '*.jpeg' ! -iname '*.png' ! -iname '*.bmp' ! -iname '*.webp' ! -iname '*.tif' ! -iname '*.tiff'
# no output

unzip -Z1 data/corpus/imd2020/IMD2020.zip | sed -n '1,20p'
# image/mask files under opaque directories such as 1a07yi/ and z2/; no metadata member

.venv/bin/python scripts/fetch_imd2020.py --check
# 2010 complete triples, 414 source groups, no type metadata

.venv/bin/python scripts/sample_imd2020.py --root data/corpus/imd2020/extracted --count 400 --seed 20260828 --out /tmp/imd2020-selection.json
ValueError: cannot determine manipulation type for pair 1a07yi:c8swtoq_0; FETCH must confirm a type-bearing directory or provide a type map
```

The sampler self-test passes, but no selection JSON or manifest rows were
written. K4 therefore remains blocked rather than using a guessed type label.

## K5 - serialization fixed; enlarged refit blocked

Changed:

- `scripts/calibrate.py`
- `backend/app/analysis/calibration.json`
- `docs/calibration.md`

The computed `weight_skill_spearman` is now serialized at the top level of
`calibration.json`. The existing Hanley-McNeil guard and positive Spearman
assertion were not changed. A current-corpus serialization rerun produced:

```text
.venv/bin/python scripts/calibrate.py --corpus all --out backend/app/analysis/calibration.json --seed 20260828
weight/heldout-skill Spearman=0.627376434428478
```

Before/after for that serialization-only rerun:

| measure | before | after |
|---|---:|---:|
| fitted images | 126 | 126 |
| defined Hanley-McNeil SE values | 8 | 8 |
| `weight_skill_spearman` artifact value | `None` from artifact lookup | `0.627376434428478` |
| fused held-out AUC | 0.5888888888888889 | 0.5888888888888889 |
| best single held-out AUC | 0.7333333333333333 | 0.7333333333333333 |

No enlarged-corpus after value exists. K4's missing type mapping prevents a
compliant 300-500-pair sample, so recalibration on IMD2020 is blocked. No
detector, threshold, or fusion weight was tuned to change the AUC.

### Per-detector within-source AUC standard error

These are the Hanley-McNeil SE values used by the weight guard. “After” is
explicitly unavailable for the enlarged corpus because K4 is blocked.

| detector | before SE | after SE |
|---|---:|---:|
| c2pa | null | blocked |
| cfa | null | blocked |
| copy_move | 0.086710 | blocked |
| double_jpeg | 0.051139 | blocked |
| ela | null | blocked |
| entropy | 0.055222 | blocked |
| exif | 0.055197 | blocked |
| jpeg_ghosts | 0.054715 | blocked |
| learned | null | blocked |
| prnu | 0.055222 | blocked |
| qtable | 0.055237 | blocked |
| spectral | 0.054578 | blocked |

The requested benchmark was run after the serialization repair:

```text
.venv/bin/python scripts/benchmark.py --out /tmp/post-r6.json --corpus all
wrote /tmp/post-r6.json and /tmp/post-r6.md
corpus n_images=126, real_corpus_present=True
fused heldout_auc=0.5888888888888889
```

The required gates passed:

```text
.venv/bin/python -m pytest backend/tests -q
59 passed, 1 warning in 288.94s

.venv/bin/python plan/validate.py
All structural and shell-syntax checks passed.
```

The synthetic corpus remains present in the all-corpus output and the backend
suite. No R6 commit was created.

## Round 6b - fixed: source-directory sample and enlarged calibration

The corrected grouping key is the IMD2020 source directory. No manipulation
taxonomy is inferred from the opaque IDs. `scripts/sample_imd2020.py` now
selects breadth-first by source directory, records `max_per_source: 2`, keeps
the manipulated image/mask and its `_orig.jpg` counterpart, and writes the
selected IDs, hashes, sizes, source groups, and split labels into
`data/corpus/MANIFEST.yaml`.

The initial target of 400 manipulated pairs would have produced 800 rows. A
representative all-detector timing measured `2.200` wall-seconds per image
(10 IMD rows took `21.995` seconds; process-inclusive time was `25.22`
seconds). The full 926-row attempt was stopped after `6:17.04` without
writing an artifact, which makes the permitted fallback applicable. The
actual sample is therefore 200 manipulated pairs plus 200 originals: 400
rows across 200 IMD2020 source groups, one pair per source, with the fixed
seed `20260828`, cap 2, 280 train rows, and 120 held-out rows. The final
calibration took `9:23.26`; the final all-corpus benchmark took `10:26.70`.

IMD2020 remains local-only under the ignored `data/corpus/imd2020/` path.
The dataset citation is Novozamsky, Mahdian, and Saic, “IMD2020: A
Large-Scale Annotated Dataset Tailored for Detecting Manipulated Images,”
IEEE WACV Workshops 2020. No image or mask bytes were added to Git.

The enlarged benchmark also exposed a loader-label hazard: mapping every
manifest label other than `ai_generated` to authentic would turn the IMD2020
`manipulated` rows into negatives. The final loader preserves the manifest
label, and metric code treats both `manipulated` and `ai_generated` as
positive labels.

### Before/after measurement

Before is the pre-6b calibration artifact: 126 rows across 27 source groups.
After is the post-6b fit: 526 rows across 227 source groups, including the
400 IMD rows. AUC and SE below are the source-local detector AUC and its
Hanley-McNeil standard error, using the same guard already present in S10.
An em dash means no applicable paired observations; it is not a zero score.

| detector | before AUC +/- SE | after AUC +/- SE |
|---|---:|---:|
| c2pa | — | — |
| cfa | — | — |
| copy_move | 0.602151 +/- 0.086710 | 0.584677 +/- 0.055533 |
| double_jpeg | 0.662835 +/- 0.051139 | 0.659864 +/- 0.024896 |
| ela | —* | 0.437642 +/- 0.026404 |
| entropy | 0.501916 +/- 0.055222 | 0.472885 +/- 0.025532 |
| exif | 0.507663 +/- 0.055197 | 0.507663 +/- 0.055197 |
| jpeg_ghosts | 0.551724 +/- 0.054715 | 0.538549 +/- 0.026526 |
| learned | — | — |
| prnu | 0.501916 +/- 0.055222 | 0.527115 +/- 0.025515 |
| qtable | 0.496169 +/- 0.055237 | 0.438776 +/- 0.026412 |
| spectral | 0.559387 +/- 0.054578 | 0.535792 +/- 0.025479 |

`*` The pre-6b artifact recorded ELA as unavailable because the shared
`ImageContext.format` accessor could return an empty value on first lazy
access. Round 6b fixed that root cause; the after value is therefore the
first valid ELA measurement and the guard drops it below chance. This did not
change the detector or tune the result.

| corpus measure | before | after |
|---|---:|---:|
| rows | 126 | 526 |
| source groups | 27 | 227 |
| held-out rows | 43 | 168 |
| fused held-out AUC | 0.588889 | 0.688406 |
| best single held-out AUC | 0.733333 | 0.738462 |
| weight/skill Spearman | 0.627376 | 0.894614 |

Fusion remains below its best single detector after enlargement, so S10
remains failed on its existing relative gate. The Hanley-McNeil guard and
positive Spearman assertion remain in place; no absolute AUC floor was added
and no detector weight was tuned to raise fusion.

### Round 6b verification

```text
.venv/bin/python scripts/sample_imd2020.py --root data/corpus/imd2020/extracted --count 200 --max-per-source 2 --seed 20260828 --out /tmp/imd2020-selection-r6b.json --manifest data/corpus/MANIFEST.yaml
manifest rows added: 400 -> data/corpus/MANIFEST.yaml
selected 200 IMD2020 pairs (400 images) -> /tmp/imd2020-selection-r6b.json
selected 200 source groups with max_per_source=2

.venv/bin/python scripts/calibrate.py --corpus all --out backend/app/analysis/calibration.json --seed 20260828
weight/heldout-skill Spearman=0.8946135105917714

.venv/bin/python scripts/benchmark.py --out /tmp/post-r6.json --corpus all
wrote /tmp/post-r6.json and /tmp/post-r6.md

.venv/bin/python scripts/fetch_corpus.py --check
426 manifest entries verified

.venv/bin/python -m pytest backend/tests -q
59 passed, 1 warning in 282.29s (0:04:42)

.venv/bin/python plan/validate.py
All structural and shell-syntax checks passed.

git check-ignore -v data/corpus/imd2020/ data/corpus/imd2020/IMD2020.zip data/corpus/imd2020/extracted/1a07yi/c8swtoq_0.jpg
.gitignore:9:data/corpus/imd2020/  data/corpus/imd2020/
.gitignore:9:data/corpus/imd2020/  data/corpus/imd2020/IMD2020.zip
.gitignore:9:data/corpus/imd2020/  data/corpus/imd2020/extracted/1a07yi/c8swtoq_0.jpg

git status --porcelain | grep -E '\\.(jpg|jpeg|png|zip)$'
# no output
```

The synthetic corpus and CI path remain unchanged; IMD2020 is only loaded by
the optional `all`/`real` benchmark path. No commit was created.
