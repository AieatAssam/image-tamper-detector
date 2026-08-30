# Round 8 repair report - 2026-08-30

No commit was created. The user-provided `FINDINGS-R8-2026-08-30.md` remains
untracked. No image, archive, or model-weight files were added to the tree.

## Measurement method

The required calibration and benchmark commands were run over the complete
corpus:

```text
.venv/bin/python scripts/benchmark.py --out /tmp/post-r8.json --corpus all
wrote /tmp/post-r8.json and /tmp/post-r8.md
corpus n_images=526, n_source_groups=227
fused heldout_auc=0.6521739130434783

.venv/bin/python scripts/calibrate.py --corpus all --out backend/app/analysis/calibration.json --seed 20260828
weight/heldout-skill Spearman=0.6509627309079725
```

The AUC and standard errors below are the `weight_guard` values written by
that calibration run and then read from `calibration.json`. They are
within-source paired estimates over the complete `all` corpus. `n+` and `n-`
are the valid positive and negative counts. The R7 fused held-out baseline was
`0.677536231884058`; no R8 detector or fusion weight was tuned to recover it.

| item | before R8 | after R8 AUC +/- SE | n+ / n- | fusion weight |
|---|---|---:|---:|---:|
| R1 Splicebuster | not registered; no measurement | 0.5119305856832972 +/- 0.02555695436689327 | 260 / 251 | 0.0; dropped by guard |
| R2 resampling | not registered; no measurement | 0.4749262536873156 +/- 0.030282095118942188 | 192 / 173 | 0.0; dropped by guard |
| R3 AEROBLADE | not registered; no measurement | N/A; AUC and SE null | 0 / 0 | 0.0; opt-in unavailable |

The catalog stores these values under `measurements.detectors` and remains the
numeric source of truth for the scientific reference.

## R1 - implemented: Splicebuster

Status: implemented and registered. This is an independent derivation from
Cozzolino, Poggi, and Verdoliva's paper, not copied or translated source.

Changed:

- `backend/app/analysis/splicebuster.py`
- `backend/tests/test_splicebuster.py`
- `backend/app/analysis/registry.py`
- `scripts/calibrate.py`
- `scripts/benchmark.py`
- `backend/app/analysis/calibration.json`
- `plan/reference/detector-catalog.yaml`

The first implementation is the requested cheaper Mahalanobis variant:
third-order residuals, three-symbol quantisation, vectorised four-symbol
co-occurrences, integral block histograms, and a regularised single-Gaussian
distance. It uses numpy/cv2 only, bounded analysis at 1024 pixels, and no
third-party source. The full-corpus result is near chance, so calibration gives
it zero weight. A focused synthetic splice benchmark was also run:

```text
.venv/bin/python scripts/benchmark.py --out /tmp/r8-splicebuster.json --corpus synthetic --detectors splicebuster
within_source_auc=0.6730769230769231
synthetic metric-set AUC=0.61125
```

That exploratory synthetic result does not override the complete-corpus
calibration result. The EM variant was not attempted because the simpler
variant did not justify the extra runtime/complexity in this round.

## R2 - implemented, exploratory: local resampling inconsistency

Status: implemented and registered in `DEFAULT_ENABLED`, but exploratory.
This is an independent derivation from Popescu--Farid and Kirchner, not copied
or translated source.

Changed:

- `backend/app/analysis/resampling.py`
- `backend/tests/test_resampling.py`
- `backend/app/analysis/registry.py`
- `scripts/calibrate.py`
- `backend/app/analysis/calibration.json`
- `plan/reference/detector-catalog.yaml`

The implementation uses Kirchner's fixed 3x3 predictor, absolute prediction
residual, two-dimensional DFT peak-to-background ratio per block, and the
75th-percentile disagreement from the block median. It does not score the
mere presence of global resampling. A conservative 512-pixel minimum on both
bounded axes makes small images abstain instead of converting benign web
resizing into a tampering claim.

The post-gate synthetic benchmark was run and reported 92 applicable rows and
8 not-applicable `resize_then_save` rows; its synthetic metric-set AUC was
`0.5213541666666667`. The corpus has no labeled local-resampling positive
family, so this is not a validated local-resampling performance claim. The
all-corpus calibration result is below chance and is assigned zero weight.

## R3 - implemented, opt-in: AEROBLADE-style reconstruction

Status: not blocked, but not part of the default detector set. The detection
algorithm is reimplemented from Ricker, Lukovnikov, and Fischer's paper. The
runtime uses the MIT-licensed distilled TAESD ONNX conversion pinned in
`aeroblade.py`; no source was copied into this repository and no torch was
added.

Changed:

- `backend/app/analysis/aeroblade.py`
- `backend/tests/test_aeroblade.py`
- `requirements-learned.txt`
- `backend/app/analysis/registry.py`
- `scripts/calibrate.py`
- `scripts/benchmark.py`
- `backend/app/analysis/calibration.json`
- `plan/reference/detector-catalog.yaml`

The external ONNX encoder/decoder pair was verified to load with onnxruntime
without torch. The focused detector tests passed (`3 passed`), including the
absent-model NOT_APPLICABLE path and the no-torch source check. The standard
corpus has no external TAESD pair at the configured path, so the required
benchmark correctly reports 0 applicable rows for both synthetic and real
subsets. Consequently its AUC and SE are null, not zero, and no performance
claim is made.

The implementation explicitly records the three substitutions/limits: TAESD
is distilled rather than the paper's exact autoencoder, mean L1 replaces the
paper's LPIPS distance, and the cue is latent-diffusion-only and useless for
splicing, copy-move, or GAN output. External weights remain outside the repo.

## R4 - implemented: scientific reference

Status: implemented after R1--R3 landed.

Changed:

- `docs/detection-principles.md`
- `docs/index.md`
- `plan/reference/detector-catalog.yaml`
- `plan/STATUS.yaml`

The reference covers every detector, including ELA (`~0.44`), entropy
(`~0.47`), qtable and CFA null applicability findings, and ZERO near chance.
It includes the physical/statistical principle, material formulas, signal
direction, citations, failure modes, the complete bibliography, limitations,
and the paper-versus-permissive-code licensing record. Its numerical links
point to the catalog's calibrated `measurements` section rather than copying
values into prose.

## Verification

Passed:

```text
.venv/bin/python -m pytest backend/tests -q
76 passed, 1 warning in 303.65s

.venv/bin/python plan/validate.py
All structural and shell-syntax checks passed.

cd frontend && npx tsc --noEmit
passed

cd frontend && npx biome check .
passed; one existing deprecated-config info for biome.json's `recommended` field

cd frontend && npm run test
2 test files, 4 tests passed

cd frontend && npm run build
passed; Vite emitted only the existing >500 kB chunk-size warning
```

The mandated all-corpus benchmark and calibration commands above also passed.
The ONNX runtime emitted macOS cache/telemetry warnings while calibrating; the
commands completed and the detector results were recorded normally.

No R8 item is blocked. No absolute AUC floor was introduced, no failing check
was weakened, and no commit was made.
