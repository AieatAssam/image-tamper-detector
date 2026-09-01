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

R15C's temporary parity corpus is the first encoding that passes the metadata
gate: every format, dimension, file-size, and EXIF ablation returned held-out
and pooled AUC `0.500`. It is nevertheless not interchangeable with native:
the exact 120,000-byte re-save strips EXIF and changes JPEG quality and history.
The R16A/B measurements are therefore recorded per variant in the catalog, and
the consolidation refit must be run only after the parity rows are available
to the manifest.

The latest committed detector measurements are the R15C/R16A/R16B results,
not the values currently serialized in `calibration.json`: AI-axis examples
are AEROBLADE parity `0.416 +/- 0.088`, learned parity `0.184 +/- 0.131`, NPR
parity `0.2803 +/- 0.0846`, and CLIP parity `0.999585 +/- 0.000757` on the
402-AI/12-camera screen. The complete per-detector table, including N/A rows,
corpus, variant, applicable count, and date, is in
[`docs/detection-principles.md`](detection-principles.md) and
[`plan/reference/detector-catalog.yaml`](../plan/reference/detector-catalog.yaml).

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
AI-generation guard, `scripts/calibrate.py` records an explicit unpaired
`ai_axis_auc` screen, now scoped to the detector's declared variant. The
screen compares applicable generated rows from `sd35_flux` and `synthbuster`
with applicable `real_camera` rows; it is not a within-source claim. A
calibration-time scope does not protect serving: the caller must provide the
matching native or parity bytes, and the current upload path remains
variant-blind as a documented limitation. The CLIP result is not a primary
generalization result while only 12 real-camera negatives are available.

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

The committed calibration artifact is still the native legacy fit from before
the R16 measurements. Its fused held-out AUC `0.5784615384615385` on 916 rows
across 317 source groups is an artifact description, not a current parity
performance claim; the older fused result is superseded and omitted. The corpus remains small, partly synthetic, and unlike the
open web. Synthetic rows cannot validate CFA, spectral, or sensor-provenance
claims. R16A's CLIP value still rounds to 1.000 after re-save, but the 12
negative rows and content/domain confound prevent treating it as generation
skill. The final native/parity refit is intentionally deferred to consolidation
after the 17A/17B detector results land.
