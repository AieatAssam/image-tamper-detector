# Round 4 repair report - 2026-08-29

Round 4 fixes the copy-move false-clean result, makes calibration reject weak
evidence with a confidence interval, and fills the strict-camera and signed-C2PA
axes. The real-AI axis remains short by three entries. S05 and S08 therefore
remain failed, and S10 remains failed because its relative fusion gate is still
red. No stage failure was weakened or relabelled.

## I1 - fixed as honest uncertainty

Changed `backend/app/analysis/copy_move.py` and
`backend/tests/test_copy_move.py`.

The instrumented `landscape_copy_paste.jpg` run found:

```text
downscaled image: 1600x900; expected paste offset: -640px
keypoints: 542
after Lowe ratio: 98
after min_offset: 64
offset bins: 59; largest cluster: 2
eligible clusters: 0; RANSAC inliers: none
```

The true offset survived the coordinate-space filter. Both pasted regions had
zero SIFT keypoints, so the textureless-region hypothesis is the cause. No
parameter was loosened. A keypoint-sufficient image with no verified cluster
now returns `NOT_APPLICABLE`, `score=None`, and a low-confidence reason instead
of a clean result.

Before, the findings reproduce returned `likely_authentic`, score `0.283`,
with `copy_move` score `0.000` and `state=applicable`. The exact API reproduce
after the fix returned HTTP 200, verdict `inconclusive`, score
`0.377033409391973`, and:

```text
copy_move state=not_applicable score=None flagged=None
reason=low confidence: no verified affine cluster; copy-move evidence could not be assessed
```

The corpus guard was checked with the focused benchmark commands. Before,
`/tmp/pre-r4-copymove.json` had 100 applicable rows and synthetic AUC
`0.61625`; 11 of 14 `copy_move` family rows were flagged. After,
`/tmp/post-r4-copymove.json` had 19 applicable and 81 not-applicable rows,
with 11 of 14 copy-move rows still flagged and the other three explicitly
not-applicable. The changed aggregate AUC is not comparable because the
previous false-clean state was converted to uncertainty; the targeted family
flag count did not regress.

## I2 - fixed, with an honest S10 remainder

Changed `scripts/calibrate.py`, `backend/app/analysis/calibration.json`, and
`docs/calibration.md`.

The before weights were:

```text
prnu 0.189881682  spectral 0.163959234  double_jpeg 0.140273177
copy_move 0.099078632  jpeg_ghosts 0.085116643  exif 0.035988720
```

The cause was both defects identified in the findings: the point guard on
`within_source_auc` let near-chance PRNU and below-chance evidence survive,
and raw logit feature scales made L2 shrinkage unequal. The fit now z-scores
the feature columns for L2 and translates coefficients back to runtime logit
scale. It fits only source groups containing both classes, so singleton
camera/AI provenance rows cannot reintroduce a source-identity shortcut.

The after weights from the generated calibration are:

```text
double_jpeg 0.101688312  within 0.662835  heldout 0.500000
spectral    0.062601481  within 0.559387  heldout 0.600000
all others  0.000000000
```

The calibration command printed `weight/heldout-skill Spearman=0.2445269506`.
No detector with held-out AUC below `0.5` has positive weight. The fitted
held-out fusion AUC is `0.5375`; the best held-out detector is spectral at
`0.6000`, so the existing relative S10 gate still fails and remains recorded
as failed. The calibration is generated, not hand-weighted.

## I2b - fixed

The point guard is replaced with the Hanley-McNeil one-standard-error rule:
keep only when `within_source_auc > 0.5 + SE`. Each detector now records
`weight_guard.auc`, `weight_guard.se`, class counts, rule, and `drop` in
`calibration.json`.

Measured examples from the generated file:

```text
double_jpeg AUC 0.662835 SE 0.051139 keep
spectral    AUC 0.559387 SE 0.054578 keep
prnu        AUC 0.501916 SE 0.055222 drop
qtable      AUC 0.496169 SE 0.055237 drop
```

The positive Spearman assertion is a calibration gate, not a hand-set weight.

## I3 - fixed for both requested axes; real-AI shortfall remains

Changed `data/corpus/MANIFEST.yaml`, `scripts/fetch_corpus.py`, and
`docs/corpus.md`.

Before: 11 `real_camera` entries, all labelled `relaxed`; 0 strict cameras;
0 signed C2PA entries; 9 real-AI entries. After:

```text
23 manifest entries verified
12 real_camera, all strict
9 real_ai
2 real_c2pa_signed
```

The prior camera shortfall was partly a verifier bug: the camera bytes store
`PixelXDimension` in EXIF sub-IFD `0x8769`. The fetch verifier now reads that
sub-IFD. `cam_013` is a fetched Nikon D70 image with decoded width 1200 and
nested `PixelXDimension=1200`, SHA-256
`c4f961e904540bdd5beac966b2512ffb8a78dfb2bbb2e7ea7a1de06d8f1e60ef`.

The two signed fixtures are pinned to c2pa-rs commit
`be7f5ea22b385ee1af6c327906ba002747687628`, under the recorded
`MIT OR Apache-2.0` license. `c2pa_001` parses as `Valid`; `c2pa_002` parses as
`Invalid` with `assertion.dataHash.mismatch`. The fetcher verifies the
manifest state with pinned `c2pa-python 0.37.8`.

Commands run included:

```text
.venv/bin/python scripts/fetch_corpus.py
.venv/bin/python scripts/fetch_corpus.py --check
.venv/bin/python -m py_compile scripts/fetch_corpus.py
git diff --check -- data/corpus/MANIFEST.yaml scripts/fetch_corpus.py docs/corpus.md
```

The earlier Wikimedia direct upload attempt returned HTTP 429 with
`Retry-After: 600`; a later category metadata query yielded the strict camera
candidate and its direct fetch produced 197670 bytes. No unverified entry was
recorded. The exact remaining fact is three additional real-AI image files
with named generative provenance, permissive license and attribution, fetched
bytes, and matching SHA-256/byte metadata. S05 remains failed on that shortfall.

## I4 - verified

The required final commands produced:

```text
.venv/bin/python -m pytest backend/tests -q
58 passed, 1 warning in 278.16s

.venv/bin/python scripts/benchmark.py --out /tmp/post-r4.json --corpus all
wrote /tmp/post-r4.json and /tmp/post-r4.md

.venv/bin/python plan/validate.py
All structural and shell-syntax checks passed.
```

The exact sample reproduce returned HTTP 200, `verdict=inconclusive`,
`score=0.377033409391973`, and `copy_move state=not_applicable score=None`.

The source-balance check on `data/corpus/synthetic/index.json` returned 12
sources, class totals 40 authentic / 60 manipulated, maximum per-source class
shares 0.10 authentic and 0.116667 manipulated. Every source supplied both
classes, well below the 0.40 cap. Pooled and within-source AUCs remained
directionally aligned across the available detectors; their rank correlation
was `0.3488426968`. Examples were double-JPEG `0.550833/0.662835`, copy-move
`0.735294/0.571429`, PRNU `0.531667/0.501916`, and spectral
`0.497917/0.559387` (pooled/within-source).

Supporting integration fixes made the strict nested EXIF evidence visible to
CFA and made Reader's explicit `validation_state=Valid` take precedence over
the valid fixture's informational `failure` details. Focused CFA, C2PA, and
copy-move tests passed as part of the 58-test suite.

No git commit was created.
