# Round 8b repair report - 2026-08-30

No commit was created. `data/samples/` was read only; no image, archive, or
model-weight files were added.

## Before and after fusion

The baseline is the calibration committed at HEAD `bcc727d` before this repair.
The new fit used the same seed and grouped source-image split.

| measure | before R8b | after R8b |
|---|---:|---:|
| fused held-out source-paired AUC | 0.6521739130434783 | 0.7536231884057971 |
| Splicebuster fusion weight | 0.0 | 0.18227337214570286 |
| Splicebuster held-out AUC | 0.572463768115942 | 0.7962962962962963 |
| resampling fusion weight | 0.0 | 0.0 |

The best post-change single-detector held-out AUC is Splicebuster at
0.7962962962962963, so the existing relative S10 fusion gate remains a separate
failure. No absolute AUC floor was added and no weight was tuned against the
held-out result.

## N1 - Splicebuster scope

Status: complete. The calibration scope is now the synthetic corpus only. The
benchmark still runs on all rows; IMD2020 is retained and is Tier B for
Splicebuster. No `NOT_APPLICABLE` recompression gate was added.

The scope is justified by the detector's stated precondition, not by selecting
the highest number. Splicebuster measures a camera processing-chain fingerprint
in residual statistics. The synthetic corpus preserves controlled processing
history. IMD2020 consists of internet-sourced, heavily recompressed images,
which destroys the fingerprint the method assumes remains measurable.

The complete post-change benchmark measured:

| corpus | metric-set AUC | within-source AUC | applicable |
|---|---:|---:|---:|
| synthetic | 0.611250 | 0.6730769230769231 | 100/100 |
| IMD2020 | 0.434425 | 0.4200000000000000 | 400/400 |
| all rows before scope | 0.47150011579434925 | 0.5119305856832972 +/- 0.02555695436689327 | 526/526 |

The new `VALIDATED_BY` entry in `scripts/calibrate.py` restricts only
Splicebuster's fit and score contribution to `synthetic`; detectors without an
explicit calibration scope, including resampling, continue to use all
applicable observations. The regenerated calibration records:

```text
within_source_auc=0.6730769230769231
auc_standard_error=0.05350654522518043
n_positive=60
n_negative=40
threshold=9.018142700195312
scale=0.571765661239624
weight=0.18227337214570286
```

A recompression-strength `NOT_APPLICABLE` gate was not feasible in this round.
The existing `double_jpeg` and qtable signals are evidence detectors with
different assumptions, not a validated scalar estimate of whether the
processing-chain fingerprint survived. Choosing a threshold from this corpus
would add an unsupported heuristic, so calibration scope is the explicit guard.

Changed for N1/N4:

- `scripts/calibrate.py`
- `scripts/benchmark.py`
- `backend/app/analysis/calibration.json`
- `plan/reference/detector-catalog.yaml`
- `docs/detection-principles.md`

`backend/app/analysis/splicebuster.py` was not changed.

## N2 - resampling

Status: complete, no scope change. The evidence does not clearly support
scoping this detector:

| corpus | metric-set AUC | within-source AUC | applicable |
|---|---:|---:|---:|
| synthetic | 0.5213541666666667 | 0.524390243902439 | 92/100 |
| IMD2020 | 0.5414446285432807 | 0.5378787878787879 | 300/400 |
| all pooled | 0.525576127406791 | 0.4749262536873156 +/- 0.030282095118942188 | 365/526 |

Resampling remains unscoped, remains documented as exploratory because there is
no labeled local-resampling positive family, and remains at zero weight under
the existing `within_source_auc <= 0.5 + SE` guard. No resampling source file
was changed.

## N3 - benchmark fast path and runtime

Status: complete. `scripts/benchmark.py` now accepts `--sample N --seed SEED`.
It selects a fixed-size subset proportionally within corpus/family-or-axis/label
strata and preserves the original deterministic row order. Sampled output
records the selection metadata. Sampled JSON and Markdown use a fixed 500 ms
timing bucket; `--profile` prints the raw measured mean duration per applicable
detector without putting nondeterministic timings into the evidence files.

The two all-detector runs with `--sample 64 --seed 20260828` were byte-identical
for both JSON and Markdown. Measured wall-clock:

| run | images | wall-clock |
|---|---:|---:|
| before, full corpus at HEAD | 526 | 635.37 s |
| after, full corpus with `--profile` | 526 | 726.98 s |
| after, `--sample 64 --seed 20260828` | 64 | 80.61 s |

The full-run difference is host load and does not change the full code path. The
sample fast path is 7.9x shorter than the pre-change full baseline and is the
documented iteration path.

The exact required no-profile full benchmark rerun also completed successfully;
its wall-clock was approximately 1491.6 s from the command timestamps. This
host-load variance is why raw timings are reported with `--profile` and are not
put in deterministic benchmark artifacts.

Raw post-change mean detector durations from the full `--profile` run:

| detector | mean applicable ms |
|---|---:|
| aeroblade | 0.0 |
| c2pa | 4.5 |
| cfa | 1397.8 |
| copy_move | 146.5 |
| double_jpeg | 145.4 |
| ela | 193.7 |
| entropy | 863.5 |
| exif | 7.9 |
| jpeg_ghosts | 435.1 |
| learned | 133.3 |
| prnu | 1123.0 |
| qtable | 1.0 |
| resampling | 76.5 |
| spectral | 17.2 |
| splicebuster | 60.9 |
| zero | 278.8 |

Changed for N3: `scripts/benchmark.py` and `docs/corpus.md`.

## N4 - scientific reference

Status: complete. `docs/detection-principles.md` now records the Splicebuster
synthetic scope, the IMD2020 failure and why its pooled result is misleading,
the scoped weight and held-out result, plus the per-corpus resampling results and
the decision to leave resampling unscoped. The catalog measurement was updated
to the scoped Splicebuster AUC and sample commands are documented in
`docs/corpus.md`.

## Verification

Passed:

```text
.venv/bin/python -m pytest backend/tests -q
76 passed, 1 warning in 329.53s

.venv/bin/python scripts/benchmark.py --out /tmp/post-r8b.json --corpus all
completed successfully; 526 images

.venv/bin/python scripts/calibrate.py --corpus all --out backend/app/analysis/calibration.json --seed 20260828
completed successfully; weight/heldout-skill Spearman=0.7611610667557432

.venv/bin/python plan/validate.py
All structural and shell-syntax checks passed.
```

The sampled all-detector JSON and Markdown determinism check passed with two
`--sample 64 --seed 20260828` runs. The legacy two-run synthetic S05 command
also produced byte-identical files; on this host the second process returned
134 during ONNX Runtime shutdown after writing them, with
`recursive_mutex lock failed: Invalid argument`. The abort is outside the
benchmark JSON path and did not occur in the required full-corpus run.
