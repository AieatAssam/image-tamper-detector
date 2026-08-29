# Round 5 repair report - 2026-08-29

Round 5 ran after J1 and J3 landed in the checkout. No commit was created.
The calibration fit was run with the exact requested command. No change was
needed in `scripts/calibrate.py`.

## J1 - fixed

Changed by AGENT-COPYMOVE:

- `backend/app/analysis/copy_move.py`
- `backend/tests/test_copy_move.py`

The discriminator is region-local: it uses the largest candidate cluster's
`candidate_region_keypoints` and `largest_candidate_region_matches`, not the
global keypoint count. The reference landscape has 542 global keypoints but
only 4 candidate-region keypoints and a largest candidate of 2, so it remains
`NOT_APPLICABLE` with `score=None`. A local candidate of at least three matches
is treated as examined; if no cluster passes the eight-match RANSAC gate, the
result is applicable with a low score.

The copy-move state table is `not_applicable/applicable`:

| family | before | after |
|---|---:|---:|
| authentic_recompress | 30/2 | 20/12 |
| splice | 14/0 | 9/5 |
| resize_then_save | 8/0 | 6/2 |
| double_compress_aligned | 9/1 | 5/5 |
| double_compress_shifted | 9/1 | 6/4 |
| local_retouch | 8/4 | 4/8 |
| copy_move | 3/11 | 0/14 |

After values came from:

```text
.venv/bin/python scripts/benchmark.py --out /tmp/post-r5.json --corpus all
copy_move family counts: 14 applicable, 0 not_applicable
```

The focused detector command also returned:

```text
landscape_copy_paste not_applicable score=None flagged=None
keypoints=542.0 candidate_region_keypoints=4.0 largest_candidate_region_matches=2.0
```

The implementation property is fixed. The regression assertion now checks the
contractual low-score false-negative path instead of hard-coding a pre-fit
calibration scale. The focused test passed `5 passed`, and the final backend
suite passed `59 passed, 1 warning`.

## J2 - refit complete; fused AUC moved, without tuning

Changed:

- `backend/app/analysis/calibration.json`
- `docs/calibration.md`

Unchanged: `scripts/calibrate.py`.

The exact command was run:

```text
.venv/bin/python scripts/calibrate.py --corpus all --out backend/app/analysis/calibration.json --seed 20260828
weight/heldout-skill Spearman=0.627376434428478
```

The before values were read from calibration at commit `9aa040f`; the after
values were read from the generated calibration:

| measure | before | after |
|---|---:|---:|
| fitted images | 123 | 126 |
| held-out rows | 44 | 43 |
| fused held-out AUC | 0.5375 | 0.5888888888888889 |
| best single held-out AUC | 0.6000 | 0.7333333333333333 |

The change is attributable to J1 applicability and the three J3 rows. No
absolute AUC floor was added and no weight was tuned for this result. Current
positive weights are `copy_move=0.29539779262572413`,
`double_jpeg=0.1290295002259316`, and `spectral=0.08482898250066716`.
The existing relative fusion gate remains failed because fusion is below its
best single detector.

Calibration reproducibility also passed after removing only the generated
timestamp:

```text
.venv/bin/python scripts/calibrate.py --corpus all --out /tmp/r5-calibration-repeat.json --seed 20260828
calibration reproducibility: PASS
```

## J3 - fixed

Changed by AGENT-CORPUS:

- `data/corpus/MANIFEST.yaml`
- `docs/corpus.md`

Unchanged: `scripts/fetch_corpus.py`.

The real-AI axis moved from 9/12 to 12/12. The three added entries are
identified in their Commons descriptions as Midjourney v4, Stable Diffusion
3.5 Large, and xAI Aurora. No generator was inferred from filename alone.

| id | generator | license | sha256 |
|---|---|---|---|
| `ai_midjourney_001` | Midjourney v4 | CC BY-SA 4.0 | `caa0174b9bc24712f0aaeda53817ccb0ba17aa4aa0fbc5a29d32f2220144f223` |
| `ai_stable_diffusion_001` | Stable Diffusion 3.5 Large | Public domain | `0426c90afa64550e939e9f93403f2ed1e04bc52de55f12bb9b82a0a53cd8b1fd` |
| `ai_xai_aurora_001` | xAI Aurora | Public domain | `db36e34ecd2fe591844045bc54c67659e3f03e07e640513abfb84b720dcb8647` |

The verification commands reported:

```text
.venv/bin/python scripts/fetch_corpus.py
26 manifest entries verified
.venv/bin/python scripts/fetch_corpus.py --check
passed
```

Independent hash, byte-count, decode, and clean-URL checks passed. No network
shortfall remains, so J3 is not blocked.

## J4 - verified

Changed:

- `plan/audit/REPAIR-REPORT-R5.md`
- `plan/STATUS.yaml`

No detector, fusion, or calibration code was changed for J4.

The mechanical evidence command passed:

```text
sources=12
class totals: authentic=40, manipulated=60
all sources supplied both classes: True
maximum per-source class share: authentic=0.1, manipulated=0.11666666666666667
pooled-vs-within comparable detectors=9, Spearman=0.33613445378151263
below-chance positive weights: none
weight-vs-heldout Spearman=0.627376434428478
```

The pooled and within-source AUC pairs used for that positive-rank check were:

```text
copy_move 0.6805555555555556 / 0.6021505376344086
double_jpeg 0.5508333333333333 / 0.6628352490421456
ela 0.4995726495726496 / 0.39763779527559057
entropy 0.4891666666666667 / 0.5019157088122606
exif 0.5 / 0.5076628352490421
jpeg_ghosts 0.46270833333333333 / 0.5517241379310345
prnu 0.5316666666666666 / 0.5019157088122606
qtable 0.5 / 0.49616858237547895
spectral 0.4979166666666667 / 0.5593869731800766
```

The required commands produced:

```text
.venv/bin/python scripts/benchmark.py --out /tmp/post-r5.json --corpus all
wrote /tmp/post-r5.json and /tmp/post-r5.md

.venv/bin/python plan/validate.py
execution order: S00 -> S01 -> S02 -> S03 -> S04 -> S05 -> S06 -> S07 -> S08 -> S09 -> S10 -> S11 -> S12 -> S13 -> S14
stages: 15 | shell snippets checked: 219
All structural and shell-syntax checks passed.

.venv/bin/python -m pytest backend/tests -q
59 passed, 1 warning in 194.97s (0:03:14)
```

The source-balance, AUC-agreement, weight, Spearman, benchmark, plan, and
full-suite checks are all verified. No item is blocked on a missing fact.

## Final status

J1, J2, J3, and J4 are fixed or verified. S05 and S08 are now passed; S10
remains failed because the fused held-out AUC is below the best single
detector, as required to report. The full backend suite is green.
