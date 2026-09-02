# Round 19 repair report - JPEG family

Date: 2026-09-01
Scope: `double_jpeg.py`, `ghosts.py`, `zero.py`, `qtable.py`, their owned
tests, and their catalog entries. Calibration artifacts and non-JPEG files
were not changed.

## Result summary

| detector | audit grade | grade after repair | resolution |
|---|---|---|---|
| `double_jpeg` | MAJOR-DRIFT | MAJOR-DRIFT against the papers; claim narrowed honestly | (a) direction and fit; (b) coefficient-domain and likelihood claim |
| `jpeg_ghosts` | MAJOR-DRIFT | MAJOR-DRIFT against Farid's full method; claim narrowed honestly | (a) sweep, variance guard, alignments; (b) final K-S decision and resize claim |
| `zero` | MAJOR-DRIFT | MINOR-DRIFT for the implemented core; local morphology and resize remain variants | (a) vote, NFA, validity, QF=99; (b) local/runtime details |
| `qtable` | MAJOR-DRIFT | MAJOR-DRIFT against Farid's database method; claim narrowed honestly | (b) source database, distance interpretation, and EXIF requirement |

Grades remain conservative: correcting a catalog claim does not turn a
repository variant into the cited paper's method.

## 1. `double_jpeg`

### Changes and resolutions

- Audit P0, inverted aggregate direction: **(a)**. The raw aggregate at
  `backend/app/analysis/double_jpeg.py:119-120` is now the positive mean of
  the Benford and periodicity indicators. The catalog direction remains
  higher for recompression at `plan/reference/detector-catalog.yaml:233`.
- Audit P1, non-paper `s=0..2`, `q=0.5..2` grid: **(a)**. The fit at
  `double_jpeg.py:46-62` uses the installed SciPy least-squares optimizer,
  permits `s` down to `-0.99`, and keeps a positive denominator for digits 1..9.
- Audit P0, decoded DCT instead of JPEG-stream quantized coefficients: **(b)**.
  Pillow does not expose entropy-coded JPEG DCT coefficients, and no third-party
  JPEG-DCT package or paper/reference source can be added in this round. The
  catalog now describes a decoded-DCT/FFT repository variant at
  `detector-catalog.yaml:206-225`.
- Audit P0, Bianchi-Piva EM/aligned/non-aligned likelihood: **(b)**. The FFT
  ratio and energy map remain diagnostics, but the catalog now says explicitly
  at `detector-catalog.yaml:235-241` that the cited likelihood method is not
  implemented.
- Audit P1, fixed crop origin/no non-aligned branch: **(b)**. The limitation is
  now explicit in the catalog instead of being implied as paper coverage.

### Before and after behavior

The deterministic probe used 12 source groups. Each group had a q=85 JPEG
negative and its q=75 second recompression positive. This is a controlled
synthetic probe, not a corpus replacement.

Before repair, the exact output was:

```text
rows 24 pairs 12
authentic_mean -1.5104094517050246
double_mean -1.685505068280533
within_source_auc 0.0
first_pair [(0, -1.472876344958696, False), (0, -1.6742362576578067, True)]
```

After repair, the same seeds produced:

```text
rows 24 pairs 12
authentic_mean 1.5100991809592166
double_mean 1.6852206632912372
within_source_auc 1.0
first_pair [(0, 1.4725615896454964, False), (0, 1.6734660873829987, True)]
```

The within-source AUC moved from 0.000 to 1.000, with the intended physical
direction. The regression assertions are at
`backend/tests/test_double_jpeg.py:24-29`; the negative-`s` fit test is at
`:32-39` and fails against the old bounded-grid implementation.

### Calibration impact

The `aggregate` key is preserved, but its sign is reversed and the Benford fit
can choose different parameters. `backend/app/analysis/calibration.json` was
not edited. A human must refit threshold and weight after all round-19 changes.

### Still open

Quantized JPEG-stream coefficient extraction and the Bianchi-Piva EM likelihood
map remain unimplemented by resolution (b). The catalog no longer claims them.

## 2. `jpeg_ghosts`

### Changes and resolutions

- Audit P1, sweep constants: **(a)**. `backend/app/analysis/ghosts.py:13` now
  uses q=30..90 inclusive, step 1, matching Farid's main experiment.
- Audit P1, low-variance exclusion: **(a)**. `:15` defines the 2.5 gray-value
  floor and `:109-120` excludes those blocks before mode selection.
- Audit P0, alignment: **(a), bounded implementation**. `:111-124` evaluates
  all 64 8x8 window offsets after the quality sweep and retains the strongest
  mode/coherence candidate.
- Audit P0, final region-selected K-S decision: **(b)**. `:55-70` adds a
  `ks_max` diagnostic, but the production raw key remains `distinct_modes`.
  The paper's region-selected K-S decision is not silently claimed; see
  `plan/reference/detector-catalog.yaml:248-262`.
- Audit P1, 1024px resize: **(b)**. The resize at `:18-24` remains an
  application latency policy and is documented as non-paper preprocessing.

### Evidence and tests

Command:

```text
.venv/bin/python -m pytest backend/tests/test_double_jpeg.py backend/tests/test_ghosts.py backend/tests/test_zero.py backend/tests/test_qtable.py -q
```

Output after the final owned-test changes:

```text
..............                                                           [100%]
14 passed in 8.62s
```

The contract test at `backend/tests/test_ghosts.py:11-13` fails against the old
sweep and missing variance constant. A read-only HEAD-versus-working-tree
probe on the same q=85, 256x384 JPEG reported:

```text
ghost_old {'distinct_modes': 2.0, 'spatial_coherence': 1.0, 'q0_min': 86.0, 'q0_max': 100.0}
ghost_new {'distinct_modes': 1.0, 'spatial_coherence': 1.0, 'ks_max': 0.8717201352119446, 'alignment_y': 7.0, 'alignment_x': 2.0, 'q0_min': 85.0, 'q0_max': 90.0}
```

The mode count moved from 2 to 1 on this control. This is an observed behavior
change, not evidence of improved corpus accuracy. No corpus AUC was rerun
because benchmark and calibration work is prohibited in this round.

### Calibration impact

`distinct_modes` remains the configured raw key, but the sweep, valid-block
filter, and alignment selection change its distribution. `ks_max` and alignment
metrics are diagnostics only. The old ghost calibration must be refit; no
calibration file was changed.

### Still open

The paper's region-selected K-S statistic is not the production decision, and
the 1024px resize changes the physical size of the 16px window. Both boundaries
are now explicit repository-variant claims.

## 3. `zero`

### Changes and resolutions

- Audit P0, coarse cell vote: **(a)**. `backend/app/analysis/zero.py:39-80`
  computes every aligned 8x8 DCT once, counts AC coefficients with
  `abs(DCT)<0.5`, broadcasts each count to covered pixels, resolves winners per
  pixel, and invalidates ties and the seven-pixel border. The former 32px cell
  and four-offset approximation is gone.
- Audit P0, AC/DC and directional validity: **(a)**. `:53-59` excludes only
  DC and rejects blocks constant along a horizontal or vertical direction. This
  also fixes the old omission of an entire DCT row.
- Audit P0, NFA sample model: **(a)**. `:94-102` uses the paper's conservative
  `/64` support/vote counts and `64^2*(XY)^2` multiplicity. `_global_nfas` at
  `:105-113` requires log10 NFA below zero for a main grid and chooses the most
  meaningful one.
- Audit P0, missing-grid and no-global-grid paths: **(a)**. `:116-151` accepts
  local candidates without requiring a dominant grid. `:199-218` performs the
  QF=99 pass, suppresses original main-grid pixels, and tests grid 0 for missing
  regions.
- Audit P0, literal greedy region growing: **(b)**. The local search at
  `:126-150` remains an OpenCV morphology approximation. Its 19x19 footprint
  corresponds to W=9, but the catalog at `detector-catalog.yaml:264-285`
  says plainly that it is not a reference-code translation.
- Audit P1, 1600px resize: **(b)**. The shared
  `ctx.downscaled_rgb_uint8` path at `base.py:78-88` remains for latency and
  is documented as an application variant.
- The missing ZERO catalog entry was added at
  `plan/reference/detector-catalog.yaml:264-285`.

### Evidence and tests

The pixel-shape/border test at `backend/tests/test_zero.py:55-63` fails against
the old cell implementation. The NFA multiplicity test at `:66-68` fails
against the old single-factor model.

The same synthetic host/donor splice through HEAD and the repaired code gave:

```text
clean old 0.2689414213699951 {'dominant_phase': 0.0, 'foreign_grid_strength': 0.0, 'foreign_region_count': 0.0}
clean new 1.2435957674293234e-06 {'dominant_phase': 0.0, 'foreign_grid_strength': 0.0, 'foreign_region_count': 0.0, 'missing_region_count': 0.0, 'valid_vote_fraction': 0.8603617350260416}
forged old 0.9999999999975637 {'dominant_phase': 0.0, 'foreign_grid_strength': 26.740517091646943, 'foreign_region_count': 1.0}
forged new 1.0 {'dominant_phase': 0.0, 'foreign_grid_strength': 999986.4024976727, 'foreign_region_count': 1.0, 'missing_region_count': 0.0, 'valid_vote_fraction': 0.8402811686197916}
```

The clean score decreased and the forged score saturated. This is the expected
scale change from the corrected pixel/NFA population, not a threshold-tuning
success.

The real-image runtime probe was:

```text
time .venv/bin/python - <<'PY'
from backend.app.analysis.base import ImageContext
from backend.app.analysis.zero import ZeroDetector
result = ZeroDetector().run(ImageContext.from_path('data/samples/original/landscape_original.jpg'))
print(result.state.value, result.metrics)
PY
```

It was interrupted after `126.13s user` in local region processing:

```text
KeyboardInterrupt
.venv/bin/python - <<<''  126.13s user 0.31s system 100% cpu 2:06.38 total
```

This large-image runtime issue remains open. It was not hidden by lowering the
cap or weakening the vote.

### Calibration impact

`foreign_grid_strength` remains the raw calibration key, but exact votes,
`/64` NFA, main-grid selection, and the QF=99 path materially change its scale
and can saturate it. New missing-grid metrics were added. Existing calibration
was not edited and must be regenerated by the human.

### Still open

The literal greedy region-growing implementation and a practical large-image
runtime strategy remain open. The statistical direction was not changed:
lower log10 NFA produces larger evidence with `higher_is_worse=True`.

## 4. `qtable`

### Changes and resolutions

- Audit P0, missing camera/software table database: **(b)**. The entry at
  `plan/reference/detector-catalog.yaml:169-199` now calls the implementation a
  camera-provenance consistency heuristic and states that exact standard tables
  are not proof of generic software re-saving.
- Audit P1, quality 1..100 search and L1 distance: **(b)**. The deterministic
  search remains, but the catalog labels the range and distance as engineering
  choices, not Farid parameters.
- Audit P1, EXIF Make/Model gate: **(b)**. The gate remains an explicit safety
  policy and is no longer described as a paper requirement.
- Audit P1, direction: **retain, with physical qualification**. At
  `backend/app/analysis/qtable.py:99-100`, lower standard-table distance maps to
  greater suspicion only under the camera-EXIF consistency hypothesis. Farid's
  general source-identification method has no monotone suspicious-distance rule,
  so no sign flip was made.
- Audit P2 table ordering: **(a) verify/document**. `qtable.py:15-16` now says
  the constants are natural Annex-K order and Pillow converts the JPEG DQT
  payload. Existing real Pillow fixtures keep exact-distance assertions.

### Evidence and tests

The qtable tests cover qualities 60, 75, 85, and 95, unique fingerprints,
custom-table distance, EXIF gating, and PNG abstention. The description check
is at `backend/tests/test_qtable.py:20-22`.

Post-repair direct fixture output:

```text
qtable {'libjpeg_distance': 0.0, 'estimated_quality': 85.0, 'table_count': 2.0} state applicable
```

No qtable metric moved. Only the runtime description, limitation, catalog
interpretation, and table-order comment changed.

### Calibration impact

`libjpeg_distance` and `higher_is_worse=False` are unchanged. No calibration
distribution changed in this patch, and no calibration file was edited. A
provenance-labeled camera/software table corpus is still required for Farid-style
source identification.

### Still open

The table database and source comparison are intentionally absent. This is a
complete claim correction, not a partial database implementation.

## 5. Verification

Baseline command before repair:

```text
.venv/bin/python -m pytest backend/tests -q
102 passed, 1 warning in 100.23s (0:01:40)
```

Required YAML command after catalog edits:

```text
.venv/bin/python -c "import yaml,sys; yaml.safe_load(open('plan/reference/detector-catalog.yaml'))"
```

Output was empty and the exit status was zero.

The required full suite after repair reached 115 passing tests but failed nine
tests in concurrently edited non-JPEG families:

```text
115 passed, 9 failed, 1 warning in 99.33s (0:01:39)
FAILED backend/tests/test_cfa.py::test_cfa_requires_strict_real_camera_evidence
FAILED backend/tests/test_cfa.py::test_cfa_reads_nested_exif_pixel_dimensions
FAILED backend/tests/test_copy_move.py::test_constructed_copy_move_and_negative
FAILED backend/tests/test_copy_move.py::test_real_textureless_copy_move_is_low_confidence
FAILED backend/tests/test_copy_move.py::test_copy_move_duration_cap_is_wall_clock
FAILED backend/tests/test_ela.py::test_ela_analyzer_initialization
FAILED backend/tests/test_ela.py::test_detect_tampering_original_images
FAILED backend/tests/test_ela.py::test_image_preprocessing
FAILED backend/tests/test_npr.py::test_npr_relationships_use_aligned_grids_and_include_zero_reference
```

None of those failures imports or exercises the four JPEG-family changes.
They were not modified because the round prohibits edits outside this family.

No git state-changing command was run. No commit was made.
