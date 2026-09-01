# Round 17B repair report: content shortcut check

Date: 2026-09-01

## Status

COMPLETE — the checker, self-validation, and current-corpus measurements ran
with no image-data changes.

## Method

`scripts/check_content_shortcut.py` converts each image to RGB, resizes it to a
32x32 thumbnail, and applies a Gaussian blur with radius 1.0 at thumbnail
resolution. The classifier is a standardized nearest-centroid model: its two
class centroids and per-pixel training standard deviation are fit on the
training groups only. A score is the negative-class distance minus the
positive-class distance.

The split reuses the metadata gate's deterministic seed (`20260828`), grouped
by `source_image`, with its 70/30 stratified group convention. The report gives
raw AUC and Hanley–McNeil SE. `shortcut_auc = max(AUC, 1-AUC)` makes the gate
orientation-independent; the 0.55 ceiling is the metadata gate's standing
threshold.

`--axes axis_a,axis_b` selects one or more axes and automatically includes
`real_camera` as the negative class. The default current-corpus selection is
`real_ai`, `sd35_flux`, and `synthbuster` against `real_camera`.

## Tool self-validation

The focused tests construct 40 temporary images in two classes, with one
source group per image:

| case | held-out AUC +/- Hanley–McNeil SE | result |
|---|---:|---|
| identical gray thumbnails, 20/20 labels | 0.500 +/- 0.173 (n=6/6) | stays at chance; gate passes |
| solid red versus solid blue thumbnails, 20/20 labels | 1.000 +/- 0.000 (n=6/6) | fires; gate fails |

The synthetic files are temporary test inputs, not corpus data.

## Current-corpus measurements

The current native corpus selection contains 402 AI rows from the three current
AI axes and 12 `real_camera` rows. The aggregate and each requested axis use
the same seed and grouped split; the 12 camera groups produce four held-out
camera rows. Values below are held-out results; `nAI` and `nR` are held-out
positive and negative counts. Pooled results are included to show the full
sample but are not the held-out gate.

| pairing | held-out nAI / nR | held-out AUC +/- SE | pooled AUC +/- SE | gate |
|---|---:|---:|---:|---|
| `real_ai` + `sd35_flux` + `synthbuster` vs `real_camera` | 142 / 4 | 0.586 +/- 0.136 | 0.771 +/- 0.054 | FAIL |
| `real_ai` vs `real_camera` | 4 / 4 | 0.750 +/- 0.184 | 0.861 +/- 0.079 | FAIL |
| `sd35_flux` vs `real_camera` | 36 / 4 | 0.625 +/- 0.138 | 0.792 +/- 0.055 | FAIL |
| `synthbuster` vs `real_camera` | 81 / 4 | 0.565 +/- 0.141 | 0.777 +/- 0.054 | FAIL |

The aggregate total is 414 rows (402 AI, 12 authentic); its held-out split is
268/146 rows for train/test. The individual pair totals are 24, 132, and 282
rows respectively. The check also accepted the explicit pair command
`--axes sd35_flux,synthbuster`: it selected 402 rows including the camera
negative class and returned held-out AUC 0.622 +/- 0.131.

Commands and JSON artifacts:

```
.venv/bin/python scripts/check_content_shortcut.py \
  --out /tmp/r17b-content-all.json
.venv/bin/python scripts/check_content_shortcut.py --axes sd35_flux \
  --out /tmp/r17b-content-sd35_flux.json
.venv/bin/python scripts/check_content_shortcut.py --axes synthbuster \
  --out /tmp/r17b-content-synthbuster.json
.venv/bin/python scripts/check_content_shortcut.py --axes real_ai \
  --out /tmp/r17b-content-real_ai.json
.venv/bin/python scripts/check_content_shortcut.py \
  --axes sd35_flux,synthbuster \
  --out /tmp/r17b-content-sd35_flux-synthbuster.json
```

## Interpretation scope

A high content AUC means the labels are predictable after the forensic detail
has been removed; detector scores on that pairing are therefore confounded by
content and cannot be read as forensic evidence alone. A near-0.5 result only
rules out this tested thumbnail shortcut. Content differences can be intrinsic
to the sampled datasets, so a perfectly assembled corpus might still not reach
0.5; this is evidence about corpus separability, not a universal pass/fail
claim about image generators.

This check is specifically the missing control for Round 16A's CLIP AUC of
1.000 on the parity corpus. The current aggregate held-out thumbnail AUC is
0.586, above the 0.55 ceiling, and the pooled value is 0.771. The individual
axes are also above the ceiling, although each estimate has only four held-out
camera examples. This supplies a direct content-based confound consistent
with the CLIP result: the corpus remains partially separable after forensic
detail is removed. It does not prove that CLIP uses only content, and the
nearer-to-chance held-out result than pooled result shows why the grouped
held-out number is the relevant caution. CLIP's 1.000 therefore remains
unaccepted as generation evidence.

No commit was made.
