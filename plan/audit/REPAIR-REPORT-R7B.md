# Round 7b repair report - 2026-08-30

No commit was created. The detector algorithm work from Round 7 was not reopened.

## M1 - expose detector calibration AUC and standard error

Status: complete.

The API now adds `metrics.auc` from `within_source_auc` and
`metrics.auc_standard_error` from `weight_guard.se`. Both values are training-time
properties of the detector. They are not uncertainty estimates for the uploaded
image. Unfitted detectors serialize both fields as JSON `null`.

Changed:

- `backend/app/analysis/base.py`
- `backend/app/analysis/fusion.py`
- `backend/app/api/endpoints.py`
- `frontend/src/types/api.ts`
- `frontend/src/components/AnalysisResults.tsx`
- `backend/tests/test_contract.py`
- `frontend/src/components/AnalysisResults.test.tsx`

Before, the Round 7 search found no API field or production metric populated for
detector SE; only the optional frontend probes existed. After, this command hit the
real API with a camera sample:

```text
status 200
qtable applicable 0.4957081545064378 0.058134984141335326
c2pa not_applicable None None
learned not_applicable None None
```

The table, dot-plot legend, accessible labels, evidence cards, and metric labels now
say training-time SE/AUC. Missing values render `Not returned`.

## M2 - palette corrections

Status: complete. `frontend/src/styles.css` changed only the six categorical status
swatches. The sequential evidence ramp was unchanged.

The local dataviz validator is not available in the repository or installed skill
runtime, so it was not claimed as run. The following measured OKLCH values are from
the local conversion command run against the final CSS:

| swatch | before | after | target L | target C |
|---|---|---|---:|---:|
| light manipulated | `#a7472d` | `#a7472d` | 0.519793 | 0.133073 |
| light authentic | `#267157` | `#007455` | 0.495834 | 0.102250 |
| light inconclusive | `#7657a7` | `#7657a7` | 0.524762 | 0.125095 |
| dark manipulated | `#f08a65` | `#ef8964` | 0.732301 | 0.134896 |
| dark authentic | `#70c9a2` | `#6dc69f` | 0.759928 | 0.103884 |
| dark inconclusive | `#c5a8ee` | `#bea1e7` | 0.759579 | 0.102844 |

The light authentic swatch now clears the requested `C >= 0.1` floor. All dark
swatches target `L <= 0.77`, with hue retained closely enough to preserve the
verdict semantics.

## M3 - qtable anti-correlation diagnosis

Status: complete. qtable is now `not_applicable` unless the image is JPEG, has
quantization tables, and has both EXIF Make and Model. It is therefore not
measurable on IMD2020, where encoder identity cannot be compared with claimed
provenance. No sign flip was made; `higher_is_worse: false` remains unchanged.

Changed:

- `backend/app/analysis/qtable.py`
- `backend/tests/test_qtable.py`
- `backend/app/analysis/calibration.json` (regenerated after the applicability gate)

The pre-change diagnostic reported:

```text
IMD2020 qtable: 380 applicable JPEG rows
  manipulated: 180 applicable
  authentic: 200 applicable
  manipulated PNG rows: 20 not applicable
raw paired counts authentic<manipulated/equal/authentic>manipulated: 78/76/26
```

That raw statistic is consistent with the suspected inversion: ordinary software
JPEG tables are common in the authentic internet-sourced images, while the DQT
identity is not a provenance claim on these rows. A standalone qtable score would
therefore make the chart and fusion speak beyond the available evidence.

After the gate, the same diagnostic command reported:

```text
IMD2020 rows 400
after gate states {'not_applicable': 400}
raw distance mean by label {'manipulated': 105.267, 'authentic': 31.45}
raw paired authentic<manipulated/equal/authentic>manipulated 78 76 26
```

The regenerated calibration records the remaining provenance-bearing data without
rescuing the detector:

| measure | before | after |
|---|---:|---:|
| within-source AUC | 0.438776 | 0.495708 |
| Hanley-McNeil SE | 0.026412 | 0.058135 |
| guard population | 231 negative / 240 positive | 47 negative / 53 positive |
| held-out AUC | 0.388462 | 0.391304 |
| weight | 0.0 | 0.0 |

The post-change all-corpus benchmark also reported qtable as synthetic `89/100`
applicable and other real rows `12/426` applicable; the IMD2020 subset is the
`0/400` portion of the latter. Fused held-out AUC remained `0.677536231884058`.

## Verification

```text
.venv/bin/python -m pytest backend/tests -q
67 passed, 1 warning in 460.15s

.venv/bin/python scripts/benchmark.py --out /tmp/post-r7b.json --corpus all
wrote /tmp/post-r7b.json and /tmp/post-r7b.md

.venv/bin/python plan/validate.py
All structural and shell-syntax checks passed.

cd frontend && npx tsc --noEmit && npx biome check . && npm run test -- --run && npm run build
Test Files 2 passed; Tests 4 passed
build succeeded

git diff --check
passed
```

Biome emitted only its existing configuration deprecation info, and Vite emitted
its existing chunk-size warning. No image or archive files were added.
