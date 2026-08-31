# Round 16C: native/parity dual-axis plumbing

Date: 2026-09-01  
Base: `5dc8e38`  
Status: mechanism complete; final dual-variant calibration deferred to consolidation

## Outcome

Native and parity are now explicit corpus variants in the manifest schema,
benchmark selection, calibration fitting, the detector catalog, and the
calibration artifact metadata.

The existing manifest has 816 local rows and no parity bytes. Its rows
therefore resolve to `native` through the explicit `default_variant: native`.
No parity paths, rows, or bytes were fabricated. A future parity row must set
`variant: parity` and point to an existing local parity file.

The committed numeric calibration was not refit. `calibration.json` only gains
the machine-readable variant policy and records that its existing numbers are
native-only.

## 1. Variant mechanism

`data/corpus/MANIFEST.yaml` is version 2 and declares:

```yaml
variants: [native, parity]
default_variant: native
```

Rows may override the default with `variant: parity`. The loaders normalize
every synthetic, real, matched, and external entry to a concrete variant.
Selection is explicit:

```sh
.venv/bin/python scripts/benchmark.py --corpus real --variant native --out /tmp/native.json
.venv/bin/python scripts/benchmark.py --corpus real --variant parity --out /tmp/parity.json
.venv/bin/python scripts/benchmark.py --corpus real --variant both --out /tmp/both.json
```

`benchmark.py` records `variant_selection`, the variants actually present, and
the variant on every result row. `--variant parity` has no native fallback;
with the current manifest it honestly produces zero selected rows.

Each detector has one declared scope in the runtime-ID map in
`plan/reference/detector-catalog.yaml`. The same map is present in both
scripts because the benchmark must enforce scope without importing the fit
driver. `calibration.json.variant_policy.detector_scope` makes the policy
inspectable from the serving artifact.

The provisional R16C assignment is:

| Variant scope | Detectors |
|---|---|
| parity only | `aeroblade`, `clip_probe`, `learned`, `npr`, `spectral`, `entropy` |
| native only | `c2pa`, `qtable`, `exif`, `cfa`, `ela` |
| both, pending 16A/16B evidence | `copy_move`, `double_jpeg`, `jpeg_ghosts`, `prnu`, `resampling`, `splicebuster`, `zero` |

ELA is the one explicit addition to the user-provided starting list. R15C
classified it with compression/history detectors, and its parity measurement
was blocked, so it is conservatively native-only until a completed measurement
changes that decision. This is a provisional scope, not an AUC-tuning result.

## 2. Fit and gate enforcement

`scripts/calibrate.py` now accepts `--variant native|parity|both`, defaulting to
`both` for the eventual one-command consolidation run. `_in_calibration_scope`
requires both conditions:

1. the row's validated corpus axis is in `VALIDATED_BY`; and
2. the row's concrete variant is in that detector's `VARIANT_SCOPES`.

The calibration code applies this predicate before threshold fitting, raw-score
probability conversion, source-local AUC, held-out AUC, the Hanley-McNeil
weight guard, and fusion fitting. Synthetic family names are normalized to the
existing axis names (`splice` to `synthetic_splice`, recompression families to
`synthetic_recompress`, and so on) before the axis check.

The held-out split still groups native and parity copies of one underlying
`source_image` together. Source-local comparisons use a
`source_image+variant` key, preventing a native positive from being compared
with a parity negative.

`benchmark.py` applies the same detector scope per row. It calls `run_all` only
for eligible detector IDs. An ineligible detector gets a checkable
`not_applicable` row with `scope_eligible: false`, the declared scope in its
reason, and an incremented `scope_violations` count. It is not merely excluded
from the metric after execution.

This closes the round 8c calibration-side failure mode as far as the permitted
files allow. The endpoint and detector modules are outside this round's file
boundary, so `run_all(ImageContext(...))` remains variant-blind at serving
time. That is an explicit known limitation, not an implied guarantee. Until a
serving orchestrator selects the bytes matching the artifact's scope, the
committed calibration must be treated as a native legacy model. Parity-only
detectors are not claimed safe on native input.

## 3. R15C evidence and cost of parity

The R15C byte-budget run used 402 AI rows and 12 strict camera negatives, a
120,000-byte target, and seed `20260831`. All 414 outputs were exactly 120,000
bytes, JPEG, 1024x1024, and EXIF-free. The metadata gate was exactly chance for
all groups:

| Feature group | Train AUC | Held-out AUC | Pooled AUC |
|---|---:|---:|---:|
| all | 0.500 | 0.500 | 0.500 |
| format | 0.500 | 0.500 | 0.500 |
| dimensions | 0.500 | 0.500 | 0.500 |
| file size | 0.500 | 0.500 | 0.500 |
| EXIF | 0.500 | 0.500 | 0.500 |

The cost is a class-correlated quality distribution. AI-generated outputs had
quality median 66 and mean 63.24; camera outputs had median 81.5 and mean
74.58. Equal file size therefore removed the R14 file-size shortcut while
creating a quality-factor signal.

R15C's complete detector cost table is reproduced below. `blocked` means the
command ran into the documented runtime/model time limit before serializing a
measurement. It is not a fabricated AUC.

| Detector | R14 native AUC +/- SE (n) | R15C parity AUC +/- SE (n) | R15C observation |
|---|---:|---:|---|
| aeroblade | 0.540 +/- 0.082 (402) | blocked | optional LPIPS path |
| c2pa | N/A (30) | N/A (0) | parity removes provenance |
| cfa | N/A (12) | N/A (0) | not applicable on parity |
| clip_probe | 1.000 +/- 0.000 (402) | blocked | optional model path |
| copy_move | 0.386 +/- 0.127 (82) | 0.561 +/- 0.117 (88) | material change |
| double_jpeg | 0.192 +/- 0.082 (42) | 0.373 +/- 0.088 (414) | JPEG history changed |
| ela | 0.303 +/- 0.095 (42) | blocked | compression-sensitive |
| entropy | 0.554 +/- 0.081 (402) | blocked | slow local-entropy path |
| exif | 0.083 +/- 0.058 (42) | N/A (0) | parity removes EXIF |
| jpeg_ghosts | 0.417 +/- 0.100 (42) | blocked | JPEG-history-sensitive |
| learned | 0.424 +/- 0.137 (114) | blocked | optional ONNX path |
| npr | 0.342 +/- 0.087 (402) | blocked | 1024x1024 path timed out |
| prnu | 0.588 +/- 0.078 (402) | blocked | noise cue needs follow-up |
| qtable | N/A (12) | N/A (0) | applicability disappears |
| resampling | 0.298 +/- 0.094 (360) | 0.240 +/- 0.082 (414) | material change |
| spectral | 0.602 +/- 0.077 (402) | 0.508 +/- 0.084 (414) | moved near chance |
| splicebuster | 0.720 +/- 0.110 (35) | blocked | manipulation-sensitive |
| zero | 0.275 +/- 0.084 (402) | blocked | compression/grid-sensitive |

### Recommendation

The final evidence-based recommendation remains separate from the provisional
plumbing assignment:

- Use parity for the AI-generation screen: `aeroblade`, `clip_probe`,
  `learned`, `npr`, `spectral`, and `entropy`. The measured spectral change
  from `0.602` native to `0.508` parity and the exact `0.500` metadata gates
  support this direction. The blocked parity runs remain unmeasured.
- Keep provenance and capture-history detectors on native: `c2pa`, `qtable`,
  `exif`, `cfa`, `ela`, `double_jpeg`, `jpeg_ghosts`, `splicebuster`, and
  `zero`. Parity either removes their evidence or changes the JPEG history;
  the completed double-JPEG shift from `0.192` to `0.373` is direct evidence
  of that cost.
- Keep `copy_move`, `prnu`, and `resampling` native until detector-specific
  parity gates complete. Their measured changes (`0.386` to `0.561`, blocked,
  and `0.298` to `0.240`) show that they cannot be silently treated as stable
  across encodings. Their code scope is left `both` this round so 16A/16B can
  confirm or narrow it without changing the plumbing again.

No threshold, weight, or fusion score was tuned to improve AUC in R16C.

## 4. Validation

Commands and results:

```text
.venv/bin/python -m py_compile scripts/benchmark.py scripts/calibrate.py
# pass

.venv/bin/python -m json.tool backend/app/analysis/calibration.json
# pass

.venv/bin/python scripts/benchmark.py --corpus real --variant parity \
  --detectors c2pa,entropy --out /tmp/r16c-empty-parity.json
# pass; n_images=0, no native fallback

.venv/bin/python scripts/benchmark.py --corpus synthetic --variant native \
  --sample 12 --seed 20260828 --detectors entropy,c2pa --out /tmp/r16c-native-a.json
.venv/bin/python scripts/benchmark.py --corpus synthetic --variant native \
  --sample 12 --seed 20260828 --detectors entropy,c2pa --out /tmp/r16c-native-b.json
cmp /tmp/r16c-native-a.json /tmp/r16c-native-b.json
# pass; deterministic repeat

.venv/bin/python -m pytest -q backend/tests/test_contract.py
# 8 passed, 1 warning

.venv/bin/pytest -q backend/tests
# 93 passed, 1 failed, 1 warning
```

The one full-suite failure is in the parallel agent's modified
`backend/app/analysis/entropy.py`, not in an R16C file:
`test_detect_ai_generated_original_images` observed `0.1189747667` for the
original landscape image while the test expects it above `0.35`. That module
and its test were not touched.

The eventual refit command is wired and documented:

```sh
.venv/bin/python scripts/calibrate.py --corpus all --variant both \
  --out backend/app/analysis/calibration.json --seed 20260828
```

An all-detector trial to `/tmp/r16c-calibration.json` was stopped during the
existing optional LPIPS/model-backed path after it reached the known long
runtime. It wrote no repository artifact and did not alter the committed
calibration values. The lightweight scope checks and the contract test passed.

No commit was made. The working tree also contains pre-existing uncommitted
changes in parallel-owned detector modules (`ela.py`, `entropy.py`, `ghosts.py`,
`npr.py`, and `prnu.py`); they were preserved and not edited.
