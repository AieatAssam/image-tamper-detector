# Calibration

Run `.venv/bin/python scripts/calibrate.py --corpus all --out backend/app/analysis/calibration.json --seed 20260828` after regenerating the synthetic corpus. The current output reports held-out AUC 0.855 on 30 source-image holdout rows. It stores thresholds, scales, weights, intercept, and held-out metrics. It is valid for images resembling the corpus; the corpus is small, partly synthetic, and not representative of the open web.
