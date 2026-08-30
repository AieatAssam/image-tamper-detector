# Round 8c repair report - 2026-08-30

No commit was created. `data/samples/` was read only; no sample, archive, or
model-weight files were added.

## O1 - Splicebuster train/serve scope mismatch

Status: implemented, with the equivalence check failing honestly. Splicebuster
now gates at inference on a measurable JPEG recompression proxy. The old
calibration-only scope was removed, but the self-gate does not reproduce the
old synthetic-only fit closely enough to call the two conditions equivalent.

### Changes

- `backend/app/analysis/qtable.py` exposes the existing qtable estimated-quality
  calculation as `jpeg_quality_proxy`; no third recompression estimator was
  added.
- `backend/app/analysis/splicebuster.py` requires JPEG qtables and an estimated
  quality of at least 80, in addition to its existing bounded-size check.
  Non-JPEG, missing-table, and lower-quality inputs return
  `NOT_APPLICABLE` with a reason and null score.
- `scripts/calibrate.py` no longer has a Splicebuster `VALIDATED_BY` entry or
  calibration-scope filter; all applicable observations are used.
- `backend/app/analysis/calibration.json` was regenerated from the all-corpus
  gated observations.
- `backend/tests/test_splicebuster.py` covers the quality gate and the existing
  detector path now uses an applicable JPEG fixture.
- `plan/reference/detector-catalog.yaml` and `docs/detection-principles.md`
  describe the gate and its limitation.

### Threshold-selection evidence

The proxy is the lowest estimated libjpeg quality across the JPEG quantisation
tables. The controlled synthetic corpus was measured by proxy bucket. The
quality-70 bucket is the excluded low-quality condition; its AUC is below
chance, while the next measured bucket separates the classes.

| measured proxy | rows | positive / negative | pooled AUC | within-source AUC |
|---:|---:|---:|---:|---:|
| 70 | 12 | 4 / 8 | 0.375000 | 0.454545 (11 pairs, 3 groups) |
| 80 | 12 | 4 / 8 | 0.687500 | 0.818182 (11 pairs, 3 groups) |
| 84 | 25 | 15 / 10 | 0.806667 | 0.750000 (28 pairs, 3 groups) |
| 90 | 51 | 37 / 14 | 0.791506 | 0.898305 (59 pairs, 7 groups) |

The selected precondition is therefore `estimated_quality >= 80`. The q=95
re-encode probe is `APPLICABLE`; the q=45 probe is `NOT_APPLICABLE` with
`estimated JPEG quality 45 is below the Splicebuster minimum 80`.

With that gate, the full benchmark measured:

| corpus | applicable | not applicable | metric-set AUC | within-source AUC |
|---|---:|---:|---:|---:|
| synthetic, 100 | 88 | 12 | 0.668527 | 0.748428 |
| local manifest, 426 | 293 | 133 | 0.486766 | 0.459184 |
| pooled calibration population | - | - | - | 0.608696 |

### Calibration impact

The comparison is material, so the self-gate is not equivalent to the former
synthetic calibration scope:

| measure | before R8c (R8b) | after R8c |
|---|---:|---:|
| Splicebuster within-source AUC | 0.673077 | 0.608696 |
| Splicebuster held-out AUC | 0.796296 | 0.674699 |
| Splicebuster fusion weight | 0.182273 | 0.000000 |
| fused held-out source-paired AUC | 0.753623 | 0.652174 |

The lower fused value is the honest result of removing the optimistic
calibration-only scope. The final calibration also records
`weight_skill_spearman=0.5570093470655848`.

## O2 - cfa, spectral, and prnu audit

Status: complete; no detector has the same calibration-only scope mismatch.
The `VALIDATED_BY` map in `scripts/benchmark.py` assigns benchmark evidence
tiers only; it is not consulted by inference. The calibration scope map in
`scripts/calibrate.py` is now empty.

Files changed for O2: none of the cfa, spectral, or prnu implementation files;
the audit is recorded here because their existing applicability behavior was
verified rather than changed.

| detector | inference applicability | audit result |
|---|---|---|
| cfa | JPEG, EXIF Make/Model, and exact EXIF PixelX/PixelY dimensions | Self-gates on strict real-camera evidence. The real-corpus audit found 12 applicable and 414 `NOT_APPLICABLE`; existing tests cover missing provenance, dimension mismatch, nested dimensions, and full-resolution use. |
| spectral | decoded images at least 32x32 | Self-gates on its actual size precondition and has no calibration scope. It is a generator/up-sampling spectral cue, not a real-sensor detector, so a real-camera provenance gate would be a different capability. The real-corpus audit found 426 applicable. |
| prnu / noise residual | all decoded supported formats | The adapter is intentionally always applicable because the implementation is a blind noise-residual inconsistency cue, not camera PRNU attribution. It has no calibration scope and its documented limitations cover resizing, recompression, and denoising. It therefore has no same train/serve mismatch, but no image-checkable sensor-provenance gate exists for this blind cue. |

The missing real-sensor gates for spectral and the blind noise-residual
implementation are explicit methodological limitations, not hidden
calibration scopes. They remain open only if either detector is later presented
as a sensor-provenance detector; adding a gate now would change the documented
capability rather than repair the O1 mismatch.

## O3 - methodological documentation

Status: complete. `docs/detection-principles.md` now states the general rule:
applicability must be checkable from the image at inference time, and
calibration scope is evidence-population metadata rather than a substitute for
an inference-time precondition.

File changed for O3: `docs/detection-principles.md`.

## Verification

Results:

```text
.venv/bin/python -m pytest backend/tests -q
77 passed, 1 warning in 300.39s

.venv/bin/python scripts/benchmark.py --out /tmp/post-r8c.json --corpus all
full 526-row artifact written; process then exited 134 during the known
ONNX Runtime shutdown mutex failure after writing the JSON and Markdown
The artifact was produced before the requested calibration rerun, so its
convenience `fused.heldout_auc` field echoes the prior calibration; the final
fused value above is from the subsequent calibration artifact.

.venv/bin/python scripts/calibrate.py --corpus all --out backend/app/analysis/calibration.json --seed 20260828
completed; weight/heldout-skill Spearman=0.5570093470655848

.venv/bin/python plan/validate.py
All structural and shell-syntax checks passed.

Final registry probe on a 384x512 gray JPEG:

q=95 -> state=applicable score=5.360626884585721e-07
q=45 -> state=not_applicable score=None
```
