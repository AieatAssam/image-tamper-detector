# Round 19 — paper-fidelity repair across all five detector families

Date: 2026-09-01/02
HEAD at start: `a8fc8ef`

Round 17 produced five read-only paper-fidelity audits and nobody had acted on
them. Round 19 acts on every P0 and P1 item in all five, one agent per family,
each restricted to its own detector files, its own tests, and its own catalog
entries. The per-family detail is in:

- `plan/audit/REPAIR-REPORT-R19-jpeg.md` — `double_jpeg`, `ghosts`, `zero`, `qtable`
- `plan/audit/REPAIR-REPORT-R19-residual.md` — `prnu`/Noisesniffer, `splicebuster`, `resampling`
- `plan/audit/REPAIR-REPORT-R19-spatial.md` — `copy_move`, `cfa`, `ela`
- `plan/audit/REPAIR-REPORT-R19-aigen.md` — `spectral`, `npr`, `aeroblade`, `entropy`
- `plan/audit/REPAIR-REPORT-R19-meta.md` — `exif`, `c2pa`, `clip_probe`, `learned`

## The rule the round was run under

Every drift finding had exactly two honest resolutions: **(a)** implement the
published method, or **(b)** correct the catalog so the entry stops claiming a
method the code does not implement. Where a paper requires trained weights, a
training corpus, or a model family this repository does not have, (b) was taken
and the missing components were named. No classifier was invented to close a
gap, no `higher_is_worse` declaration was flipped, and no threshold, gate, or
applicability precondition was moved to make a number rise.

## Grade movement

| detector | R17 grade | R19 grade | resolution |
|---|---|---|---|
| `copy_move` | MAJOR-DRIFT | MINOR-DRIFT | (a) Amerini generalized 2NN `T=0.5`, Ward clustering `Th=2.2`, normalized affine estimation |
| `cfa` | MAJOR-DRIFT | MINOR-DRIFT | (a) the module implements Bammey, not the cited Popescu-Farid/Ferrara; extra `/255` scale removed, two-pixel border, localization path |
| `splicebuster` | MAJOR-DRIFT | MINOR-DRIFT | (a) paper feature reduction and GG-EM model replace the one-class Mahalanobis substitute |
| `resampling` | MAJOR-DRIFT | MINOR-DRIFT | (a) Kirchner's p-map and cumulative periodogram replace the block-disagreement heuristic |
| `zero` | MAJOR-DRIFT | MINOR-DRIFT | (a) paper vote, NFA, validity rule, and `QF=99` handling |
| `c2pa` | MAJOR-DRIFT | MINOR-DRIFT | (a) validation and trust states are no longer collapsed, failures are typed, non-JPEG MIME dispatch fixed |
| `prnu` / Noisesniffer | MINOR-DRIFT | MINOR-DRIFT | (a) paper region-growth constant; (b) the catalog described a wavelet/variance PRNU method the code has never run |
| `double_jpeg` | MAJOR-DRIFT | MAJOR-DRIFT | (a) aggregate direction and Benford fit; (b) Pillow exposes no quantized JPEG coefficients, so the Bianchi-Piva likelihood is not implemented and is no longer claimed |
| `ghosts` | MAJOR-DRIFT | MAJOR-DRIFT | (a) sweep, variance guard, alignments; (b) the final K-S decision is not implemented and is no longer claimed |
| `qtable` | MAJOR-DRIFT | MAJOR-DRIFT | (b) no source quantization-table database exists here |
| `spectral` | MAJOR-DRIFT | MAJOR-DRIFT | (b) both cited papers require a trained classifier |
| `npr` | MAJOR-DRIFT | MAJOR-DRIFT | (b) the NPR paper requires a trained CNN |
| `aeroblade` | MAJOR-DRIFT | MAJOR-DRIFT | (b) the published multi-autoencoder minimum needs a model family this repository does not have |
| `clip_probe` | MAJOR-DRIFT | MAJOR-DRIFT | (b) no paper-matched augmentation, checkpoint, or single-source training protocol |
| `entropy` | UNVERIFIED | UNVERIFIED | (b) the citation is a blog post, not a paper; blog constants and arithmetic corrected |
| `exif` | UNVERIFIED | UNVERIFIED | (a) editor-marker and non-JPEG thumbnail handling; (b) no single primary paper exists |
| `learned` | UNVERIFIED | UNVERIFIED | (b) artifact-verified against the model config; no training paper is cited |
| `ela` | MAJOR-DRIFT | MAJOR-DRIFT | (a) raw residual analyzer repaired; (b) the served adapter still scores the repository heuristic, now labelled as one |

Correcting a claim does not turn a repository variant into the cited paper's
method, so a (b) resolution leaves the grade where it was. That is the point of
recording both columns.

## Consolidation work outside the family scopes

Three defects sat in files no family owned:

1. The `prnu` adapter reported a metric named `uniformity_score` and the reason
   string "noise residual variance" while the code computed Noisesniffer
   `-log10(NFA)`. The metric is now `noisesniffer_significance`
   (`backend/app/analysis/adapters.py`, `scripts/calibrate.py`), and
   `docs/detection-principles.md` describes the a-contrario test instead of a
   local-variance statistic.
2. `_has_validation_problem` in `c2pa.py` treated the presence of any validation
   payload as a failure. A validation payload also lists successes, so an
   unrecognised validation state with a clean payload would have been scored
   `0.95` "validation failed". It now requires an explicit failure token.
3. `_pca` in `splicebuster.py` computed the centred matrix before its
   `len(features) < 2` guard, emitting `RuntimeWarning: Mean of empty slice` on
   images with no eligible blocks. The guard now runs first; the returned values
   are unchanged.

## Tests

`.venv/bin/python -m pytest backend/tests -q` moved from **102 passed** to
**134 passed**. Every behaviour change carries a test that fails against the old
behaviour.

One test needed decoupling from calibration after the refit.
`test_copy_move_uses_normalized_full_affine_ransac` asserted `result.flagged is
True`, which is a calibrated operating point (the refit sets `copy_move`'s
threshold to `9.0` verified clusters) rather than the mechanism the test is
named for. It now asserts `verified_clusters` and the RANSAC arguments and
leaves flagging to calibration. Nothing was loosened to obtain a pass: the
mechanism assertions are unchanged.

`test_ghosts_duration_cap_is_wall_clock` asserts an 8-second wall-clock budget
against a roughly 4-second detector. It fails under heavy CPU contention and
passes on an idle machine; it is load-sensitive by construction.

## Calibration

A consolidation refit was run once, by hand, after all five agents landed:

```text
.venv/bin/python scripts/calibrate.py --corpus all --variant both \
  --out backend/app/analysis/calibration.json --seed 20260828
```

It scored 1530 rows across both variants and took about 4h40m; the repaired
`copy_move` matcher and the GG-EM `splicebuster` dominate the new per-image
cost.

| quantity | before (native legacy fit) | after (R19, both variants) |
|---|---:|---:|
| fused held-out AUC | 0.5784615384615385 | 0.6079881656804734 |
| held-out n | 557 | 1022 |
| fit population | 916 rows, native | 1530 rows, native + parity |
| weight/held-out-skill Spearman | 0.5458137240480705 | 0.49466263219604506 |

**The two numbers are not comparable.** The population, the variant selection,
and several detector statistics changed in the same step. The refit is recorded
because the old artifact no longer described the code, not as evidence that
anything improved.

### Detector movement

| detector | weight before | weight after | within-source AUC before | after |
|---|---:|---:|---:|---:|
| `exif` | 0.000 | 0.315 | 0.500 | 0.608 |
| `jpeg_ghosts` | 0.082 | 0.148 | 0.539 | 0.597 |
| `copy_move` | 0.229 | 0.083 | 0.585 | 0.593 |
| `zero` | 0.000 | 0.022 | 0.507 | 0.542 |
| `double_jpeg` | 0.133 | 0.000 | 0.660 | 0.297 |
| `prnu` | 0.010 | 0.000 | 0.543 | 0.570 |
| `ela` | 0.000 | 0.000 | 0.438 | 0.516 |
| `qtable` | 0.000 | 0.000 | 0.470 | 0.471 |

### `double_jpeg` — the round's largest movement

`double_jpeg` was the ensemble's best member and is now zero-weighted. The old
code negated its aggregate, with a comment stating that the indicators were
anti-correlated on the corpus and that the negation preserved the catalog's
higher-is-worse direction. That is fitting a sign to a corpus. The repair
removed the negation, restoring the physical premise that a recompressed block
histogram deviates further from generalized Benford and carries more
periodicity.

On a controlled probe — 12 source groups, each a `q=85` negative and its `q=75`
recompression positive — the repaired detector gives within-source AUC `1.0`
and the old code gives `0.0`. On the calibration corpus, whose positive label is
*tampering* rather than *double compression*, the repaired detector reads
`0.2966101694915254` (n=200/208, SE `0.0255855`), and the non-negative weight
guard drops it to zero with the reason recorded in the artifact.

Both facts are kept. The detector is now physically right and corpus-wise
anti-correlated, which is the same treatment AEROBLADE (`0.330`) and `learned`
(`0.086`) received in R18A. No sign was flipped to recover the number.

### Detectors with no fitted result

`aeroblade`, `splicebuster`, `spectral`, `npr`, `entropy`, `resampling`,
`clip_probe`, `cfa` and `c2pa` carry weight `0.0`. Most record
`within_source_auc unavailable for Hanley-McNeil guard` with zero applicable
positive rows under their variant and axis scope; `spectral` does have an AI-axis
result (`0.6148 +/- 0.0233`, 390 positive against 212 parity negatives) but no
within-source population, so the guard drops it. These are scope limits of the
corpus, not new failures introduced by the repairs.

### S10's fusion gate passes for the first time

S10 requires the fused held-out AUC to beat the best single detector. It has
been `failed` since R7. On the R19 refit the fused held-out AUC is
`0.6079881656804734` and the best single held-out detector is `jpeg_ghosts` at
`0.5901639344262295`, so the gate passes on the point estimate and
`plan/STATUS.yaml` is updated.

The margin is `0.0178` on 1022 held-out rows and no standard error is recorded
for the fused estimate, so this is a gate result, not a demonstration of fusion
skill. Part of the change is that `double_jpeg`, previously the strongest member
and the reason fusion looked weak by comparison, is now zero-weighted. The gate
itself was not altered.

### `exif` now carries the largest weight

`exif` holds weight `0.315` on within-source AUC `0.6081 +/- 0.0802` with 20
positive and 37 negative rows. It clears the guard, but on 57 rows the estimate
is fragile and it should not be read as a strong detector.

## Still open

- `ela` as served is still the repository heuristic rather than Krawetz's
  residual; whether to expose the raw residual is a product decision.
- `cfa` has no eligible full-resolution camera row to validate against.
- `copy_move` cannot see a textureless pasted region at all; that needs a
  separately named dense/block method, not looser SIFT thresholds.
- `splicebuster` implements the GG branch only, on a bounded 4096-row EM sample.
- `resampling` does not implement Popescu-Farid's exhaustive similarity search,
  and the corpus still has no labelled positive resampling family.
- `spectral`, `npr`, `aeroblade`, `clip_probe` and `learned` remain without the
  trained components their sources specify.
- The corpus remains the binding constraint identified in
  `plan/audit/OPEN-ITEMS.md` §C1. Paper fidelity does not create signal that the
  statistics do not contain.
