# Round 7 repair report - 2026-08-30

No commit was created. The user-provided `FINDINGS-R7-2026-08-29.md` remains
untracked. No image or archive files were added.

## Measurement method

The baseline was measured before the R7 code changes with a one-off command
that loaded `scripts.benchmark._real()`, selected `axis == "imd2020"`, ran
`registry.run_all` for every row, and calculated within-source paired AUC and
Hanley-McNeil standard error. The same command was rerun after the detector
changes and calibration. There are 400 IMD2020 rows, 200 authentic and 200
manipulated, in 200 source groups.

| detector | before applicable | before AUC +/- SE | after applicable | after AUC +/- SE |
|---|---:|---:|---:|---:|
| c2pa | 0 | N/A | 0 | N/A |
| cfa | 0 | N/A | 0 | N/A |
| copy_move | 119 | 0.532258 +/- 0.073726 | 119 | 0.544745 +/- 0.073553 |
| double_jpeg | 380 | 0.655556 +/- 0.028645 | 380 | 0.584568 +/- 0.029937 |
| ela | 380 | 0.511111 +/- 0.030462 | 380 | 0.492068 +/- 0.030467 |
| entropy | 400 | 0.435000 +/- 0.028605 | 400 | 0.474400 +/- 0.028857 |
| exif | 2 | N/A | 2 | N/A |
| jpeg_ghosts | 380 | 0.519444 +/- 0.030443 | 380 | 0.510833 +/- 0.030463 |
| learned | 0 | N/A | 0 | N/A |
| prnu | 400 | 0.560000 +/- 0.028649 | 400 | 0.487888 +/- 0.028893 |
| qtable | 380 | 0.355556 +/- 0.028900 | 380 | 0.344367 +/- 0.028643 |
| spectral | 400 | 0.505000 +/- 0.028902 | 400 | 0.510350 +/- 0.028896 |
| zero | not registered | N/A | 400 | 0.506525 +/- 0.028901 |

The post-change all-corpus command was:

```text
.venv/bin/python scripts/benchmark.py --out /tmp/post-r7.json --corpus all
wrote /tmp/post-r7.json and /tmp/post-r7.md
corpus n_images=526, n_source_groups=227
fused heldout_auc=0.677536231884058
```

The calibration command was run after the algorithm changes:

```text
.venv/bin/python scripts/calibrate.py --corpus all --out backend/app/analysis/calibration.json --seed 20260828
weight/heldout-skill Spearman=0.6487300740092935
```

The direction audit covered every below-chance IMD2020 result. `qtable` and
`entropy` remain `higher_is_worse: false` as specified by the catalog. `ela`,
`prnu`, and the other higher-is-worse detectors were not inverted in fusion.
The weaker results are reported as results, not repaired by weight changes.

## L1 - fixed: Noisesniffer noise inconsistency

Status: fixed algorithmically; IMD2020 skill is lower after the principled
replacement and is retained as an honest result.

Changed:

- `backend/app/analysis/prnu.py`
- `backend/app/analysis/adapters.py`
- `backend/tests/test_prnu.py`
- `docs/noisesniffer.md`
- `backend/app/analysis/calibration.json`

Noisesniffer was adapted from IPOL 2024/462 under Apache-2.0, with attribution
in the module docstring and documentation. The implementation uses the
background block selection, low-frequency DCT energy, region growing, and
binomial NFA significance. Its production adapter consumes the NFA statistic
through `base.to_probability()`.

The old explicit `variance_threshold` behavior survives only in a compatibility
wrapper for existing direct callers. It is not used by the production adapter
and does not feed fusion.

IMD2020 changed from `0.560000 +/- 0.028649` to `0.487888 +/- 0.028893`,
with 400 applicable rows. The detector getting worse after a principled
reimplementation is not hidden or compensated by fusion.

Focused Noisesniffer tests passed. The full backend command passed 65 tests.

## L2 - partial: CFA intermediate-values analysis

Status: implemented from the paper; real-corpus validation remains blocked by
the strict applicability gate.

Changed:

- `backend/app/analysis/cfa.py`
- `backend/tests/test_cfa.py`

This is an independent reimplementation of the paper-only IPOL 2021/355
intermediate-values method. No AGPL source was copied or translated. The
implementation computes horizontal and vertical intermediate-value masks,
estimates the dominant Bayer diagonal and red/blue arrangement, and returns a
local disagreement map. Strict CFA analysis uses the full-resolution image
only after matching JPEG format, camera Make/Model, PixelXDimension, and
PixelYDimension evidence.

IMD2020 was `0/400` applicable before and remains `0/400` after, so both AUC
and SE are N/A. The exact missing fact is strict camera-original evidence in
the IMD2020 sample with both matching EXIF dimensions. The command that
established this was the IMD2020 measurement command above; it reported
`cfa applicable=0, not_applicable=400` before and after. No strict validation
claim is made.

## L3 - partial: quantization-table direction and estimator

Status: direction verified; coefficient-only estimator not implemented.

Changed:

- `backend/tests/test_qtable.py`

The existing DQT statistic already had the catalog-correct direction:
lower `libjpeg_distance` is more suspicious, so `higher_is_worse` remains
false. A focused regression test covers the direction. The paper-only IPOL
2022/399 estimator was not implemented because this round's IMD2020 rows did
not establish a stripped-DQT or coefficient-only validation slice; the
current applicable rows continue to expose the direct-table path.

IMD2020 changed from `0.355556 +/- 0.028900` to `0.344367 +/- 0.028643`,
with 380 applicable rows. No AGPL source was used. The result remains below
chance in the catalog's already-correct direction and remains zero-weight
after calibration.

## L4 - fixed: ZERO JPEG grid origin

Status: implemented and registered; validation is near chance on IMD2020.

Changed:

- `backend/app/analysis/zero.py`
- `backend/app/analysis/registry.py`
- `backend/tests/test_zero.py`
- `scripts/benchmark.py`
- `scripts/calibrate.py`
- `backend/app/analysis/calibration.json`

ZERO is an independent reimplementation from the paper-only IPOL 2021/390
article. No AGPL source was copied, translated, or retained in the repository.
It counts AC DCT zeros for the 64 candidate 8x8 grid phases in bounded image
cells, applies binomial a-contrario tests to global and foreign local regions,
and returns both a global score and an 8x8-grid localization mask. The bounded
cell sampling is deliberate: exhaustive overlapping DCT windows made the
all-corpus calibration impractical while not changing the candidate-phase
test.

ZERO had no baseline score because it was not registered. After registration,
IMD2020 had 400 applicable rows and AUC `0.506525 +/- 0.028901`. The focused
clean-JPEG versus foreign-grid test passed, including the localization map.
Calibration assigned it zero weight because its within-source result does not
clear the existing Hanley-McNeil guard.

## L5 - measured, not removed: ELA and entropy

Status: measured and retained as required; removal is proposed for a later
decision, not performed in R7.

No detector source was changed. IMD2020 ELA moved from
`0.511111 +/- 0.030462` to `0.492068 +/- 0.030467`. Entropy moved from
`0.435000 +/- 0.028605` to `0.474400 +/- 0.028857`. The catalog directions
were checked and neither result was rescued by a sign flip. Both remain
available for comparison and are zero-weight under the existing calibration
guard.

## L6 - spectral verified; EXIF EX1 coverage added

Status: spectral verified; EXIF thumbnail validation remains data-blocked.

Changed:

- `backend/tests/test_spectral.py`
- `backend/tests/test_exif.py`

The spectral test verifies azimuthal-average subtraction and JPEG 8x8 grid
frequency exclusion. No production spectral code change was needed. Its
IMD2020 result changed from `0.505000 +/- 0.028902` to
`0.510350 +/- 0.028896`, with 400 applicable rows.

The EXIF EX1 test now creates an edited full image with a stale embedded
thumbnail and verifies the mismatch signal. IMD2020 itself has only 2
applicable rows, both without a paired authentic/manipulated source comparison,
so AUC and SE are N/A before and after. The exact missing fact is an
IMD2020 row with an embedded thumbnail and a paired source group suitable for
the EX1 comparison. The measurement command above established
`exif applicable=2, not_applicable=398` in both runs. No EXIF estimator was
claimed from that data.

## L7 - partial: visualization

Status: frontend implementation complete; requested external visual QA is
blocked by missing local tooling and by the API not returning detector SE.

Changed:

- `frontend/src/components/AnalysisResults.tsx`
- `frontend/src/components/AnalysisResults.test.tsx`
- `frontend/src/styles.css`

The frontend now provides a score dot plot ordered by score, optional error
whiskers, threshold reference line, a companion table, verdict icon and label,
distinct NOT_APPLICABLE state, a single-hue light/dark evidence ramp, an
edge-highlighted mask, swipe divider, synced viewport controls, and detector
map switching that preserves zoom. It keeps the "evidence, not proof" framing.
The UI reads optional `hanley_mcneil_se`, `auc_standard_error`, or
`standard_error` metrics and displays `Not returned` when no uncertainty is
present; it does not fabricate an SE.

The exact blocked facts and checks were:

```text
find . -name validate_palette.js -o -iname '*dataviz*'
# no matching local dataviz skill or scripts/validate_palette.js

rg -n 'standard_error|hanley_mcneil_se|auc_standard_error' backend/app/api frontend/src
# no DetectorResponse field or production metric populated by the API;
# calibration.json stores a training weight_guard.se, but the endpoint does not expose it

in-app browser setup via browser-client.mjs getDefault()
# No browser is available
```

The repository frontend checks passed:

```text
cd frontend && npx tsc --noEmit
cd frontend && npx biome check .
cd frontend && npm run test -- --run
Test Files 2 passed; Tests 4 passed
cd frontend && npm run build
build succeeded
```

The palette validator and actual rendered visual inspection could not be
claimed. Adding a backend SE field and rerunning the palette/render QA belongs
to the next UI/API pass.

## Final verification

```text
.venv/bin/python -m pytest backend/tests -q
65 passed, 1 warning

.venv/bin/python scripts/benchmark.py --out /tmp/post-r7.json --corpus all
wrote /tmp/post-r7.json and /tmp/post-r7.md

.venv/bin/python plan/validate.py
All structural and shell-syntax checks passed.

git status --porcelain | grep -E '\\.(jpg|jpeg|png|zip)$'
# no output
```

