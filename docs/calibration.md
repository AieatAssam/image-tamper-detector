# Calibration

Run:

```sh
.venv/bin/python scripts/calibrate.py --corpus all --variant both \
  --out backend/app/analysis/calibration.json --seed 20260828
```

## Native and parity variants

The calibration command accepts `--variant native|parity|both`; the default is
`both` so one consolidation run can load both encodings. Each detector is then
filtered by both its existing `VALIDATED_BY` axis scope and its declared
`variant_scope`. Rows outside either scope do not contribute to threshold,
within-source AUC, held-out AUC, the Hanley-McNeil guard, or fusion weights.

```sh
.venv/bin/python scripts/calibrate.py --corpus all --variant both \
  --out backend/app/analysis/calibration.json --seed 20260828
```

`calibration.json.variant_policy.detector_scope` and each newly generated
detector config's `variant_scope` are the machine-readable declarations. The
current committed calibration numbers were not refit in Round 16C; its
`fitted_on.variants: ["native"]` records that they are the pre-parity model.
The current manifest has no parity rows, so `--variant both` is currently
equivalent to native data and cannot manufacture missing parity observations.

The held-out split still groups by the underlying `source_image`, keeping
native and parity encodings of one source together. Within-source comparisons
key the comparison by `source_image+variant`, so a native positive is never
paired with a parity negative. `benchmark.py` applies the same scope at
execution time: an ineligible detector row is `not_applicable`, carries
`scope_eligible: false`, and increments `scope_violations` rather than being
run on the wrong variant.

This round cannot make the upload endpoint variant-aware because detector
modules and the endpoint are outside scope. The explicit known limitation is
therefore that `run_all(ImageContext(...))` remains variant-blind at serving
time. A serving orchestrator must select the bytes matching
`calibration.json.variant_policy.detector_scope` before invocation; the
benchmark and calibration outputs make that precondition checkable. Until the
consolidation refit and serving change, the committed calibration is a native
legacy model, not a claim that parity-only detectors are safe on native input.

The fit reads every available synthetic entry and every checksum-verified real
entry. `fitted_on.corpora` records the corpus names actually present, so an
unavailable real corpus is not represented as if it supplied observations.
The calibration artifact also records the computed `weight_skill_spearman` at
the top level. The R6b fit includes the source-directory-stratified IMD2020
sample: 200 manipulated pairs and their 200 `_orig.jpg` counterparts.

Thresholds are selected from the training groups with Youden's J statistic.
Scales are half the interquartile range. Each applicable raw statistic is
converted with `base.to_probability()` before numpy L2 logistic regression fits
the fusion intercept and detector weights. The logit columns are z-scored for
the L2 fit, then the coefficients are translated back to the raw logit scale
used by runtime fusion. Missing detector values are omitted from the fit and
from runtime fusion. `within_source_auc` compares only
authentic/manipulated pairs sharing the same `source_image`; rows from
different source images are never compared.

The generator-specific AI axes do not ship their camera counterpart bytes. For the
AI-generation guard, `scripts/calibrate.py` therefore records an explicit
`ai_axis_auc` screen for `learned`, `npr`, and `clip_probe`, comparing applicable generated
rows from `sd35_flux` and `synthbuster` with applicable `real_camera` rows.
This is an unpaired cross-source screen, not a within-source claim. The
selected `weight_guard.metric` says which measurement controlled a detector's
weight. The CLIP probe's separate fit holds out complete generators; its OOD
report, rather than this pooled screen, is the primary generalization result.

The held-out split is deterministic, uses `source_image` groups, and reports
only the groups excluded from fitting. Both per-detector `heldout_auc` and the
fused held-out AUC are source-local comparisons, never pooled across source
images. Fusion uses applicable detectors only:
missing or errored detectors contribute neither a score nor a log-odds term.
The verdict is forced to `inconclusive` when fewer than three detectors apply.

The weight guard uses each detector's full source-local AUC and the
Hanley-McNeil standard error:

```text
SE = sqrt((A(1-A) + (n_pos-1)(Q1-A^2) + (n_neg-1)(Q2-A^2)) / (n_pos*n_neg))
Q1 = A/(2-A)
Q2 = 2A^2/(1+A)
```

Unless `within_source_auc > 0.5 + SE`, the detector is dropped to weight zero. Each
detector's `weight_guard` records the AUC, SE, class counts, rule, and drop
decision. The calibration command also asserts a positive Spearman correlation
between fitted weights and defined held-out AUCs; held-out AUC is used for this
ranking check because it is the independent skill estimate.

Positive statistical weights are capped so no one detector can cross the
`manipulated` verdict band alone. Fitted coefficients are constrained to be
non-negative: an anti-correlated detector is dropped at weight zero rather than
silently inverted in fusion. The fitted intercept also enforces the contract's `<= 0.10` manipulated rate
for the `authentic_recompress` and `resize_then_save` false-positive traps.

The committed numbers are valid for images resembling the corpus. The corpus
is small, partly synthetic, and not representative of the open web. The
manifest currently has 12 strict real-camera images, 12 real-AI images, two
C2PA fixtures, 400 source-balanced IMD2020 rows, 120 `sd35_flux` rows, and 270
`synthbuster` rows. Synthetic images cannot validate CFA, spectral, or PRNU
sensor-provenance detectors. The `double_jpeg` aggregate was sign-corrected
after its corpus measurement showed the raw direction was inverted. The Round
11 calibration reports a fused held-out AUC of 0.5784615384615385 on 916 rows
across 317 source groups. This is lower than the prior 0.6521739130434783 and
is reported without a tuning change; no AUC floor was introduced and no
weight was tuned to improve fusion. Round 12 adds optional TAESD/LPIPS
AEROBLADE and a frozen CLIP probe. AEROBLADE is zero-weighted by its
source-paired guard (`0.511013 +/- 0.025584`). CLIP's generator-held-out report
is `1.0000 +/- 0.0000` OOD and `1.0000 +/- 0.0000` ID with four camera
negatives in each test partition; all AI rows are PNG and the camera negatives
are JPEG, so this is not a universal performance claim.
