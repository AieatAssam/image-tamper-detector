# Calibration

Run:

```sh
.venv/bin/python scripts/calibrate.py --corpus all --out backend/app/analysis/calibration.json --seed 20260828
```

The fit reads every available synthetic entry and every checksum-verified real
entry. `fitted_on.corpora` records the corpus names actually present, so an
unavailable real corpus is not represented as if it supplied observations.

Thresholds are selected from the training groups with Youden's J statistic.
Scales are half the interquartile range. Each applicable raw statistic is
converted with `base.to_probability()` before numpy L2 logistic regression fits
the fusion intercept and detector weights. The logit columns are z-scored for
the L2 fit, then the coefficients are translated back to the raw logit scale
used by runtime fusion. Missing detector values are omitted from the fit and
from runtime fusion. `within_source_auc` compares only
authentic/manipulated pairs sharing the same `source_image`; rows from
different source images are never compared.

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
manifest currently has 12 strict real-camera images, nine real-AI images, and
two C2PA fixtures; three additional real-AI images are still missing. Synthetic
images cannot validate CFA, spectral, or PRNU sensor-provenance detectors. The
`double_jpeg` aggregate was sign-corrected after its corpus measurement showed
the raw direction was inverted.
