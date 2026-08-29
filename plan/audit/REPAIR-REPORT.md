# Repair report — 2026-08-29

The audit evidence was re-run before edits and matched FINDINGS-2026-08-29.md. Final gates:

- `.venv/bin/python -m pytest backend/tests -q`: `52 passed, 1 warning`.
- `.venv/bin/python scripts/benchmark.py --out /tmp/post-fix.json --corpus all`: passed.
- Synthetic benchmark determinism (`cmp` on two same-seed runs): passed.
- Calibration reproducibility (`calibrate.py` twice, generated timestamp removed): passed.
- `plan/validate.py`: run after the status update below; passed.

## F1 — fixed

Changed `scripts/calibrate.py` and regenerated `backend/app/analysis/calibration.json`; updated `docs/calibration.md`. The fit now uses grouped training/holdout rows, records actual corpus names, gives detectors with no applicable rows null holdout AUC and zero weight, and fits signed weights instead of using the 0.01 placeholder floor.

Before, the audit command reported 11 of 12 weights at `0.01` and only spectral at `0.26632211646725884`. After the same weight dump: `floor_count 0`, `nonzero_count 5`; the grouped fused holdout is `0.855` → `0.9076923076923077` (the latter is from `calibration.json`).

## F2 — fixed

Changed `scripts/make_corpus.py`, the synthetic JSON index/sidecars, `backend/tests/test_corpus.py`, and `docs/corpus.md`. Generation now draws from four distinct existing sample sources and preserves/adds EXIF-bearing JPEG metadata.

The audit command changed from one source group containing all 100 rows to four groups: `27, 27, 23, 23`. `calibration.json` still declares `split_by: source_image`, but the split now has a real held-out group and `n: 23`.

## F3 — partially-fixed

Changed `backend/app/analysis/double_jpeg.py`, `backend/app/analysis/ela.py`, `backend/tests/test_ela.py`, and the calibration procedure. The double-JPEG aggregate sign was reversed after measuring the raw statistic and confirming authentic recompresses had the larger aggregate; ELA now handles an empty final mask without failing.

From the final all-corpus benchmark, synthetic AUC changed as follows:

| detector | before | after |
|---|---:|---:|
| double_jpeg | 0.2945833333333333 | 0.5704166666666667 |
| ela | 0.4675213675213675 | 0.5568376068376069 |
| jpeg_ghosts | 0.49833333333333335 | 0.5733333333333334 |
| copy_move | 0.6083333333333333 | 0.609375 |

`qtable` remains `0.500` because the corpus uses one standard libjpeg table and the statistic is constant; a custom-table probe produced libjpeg distance `98`. No speculative qtable algorithm change was made. CFA, spectral, and PRNU synthetic numbers were not used as validity evidence, per the plan. Among the synthetic-valid processing-history detectors, no detector that was above chance in the reference dropped below chance in the final benchmark.

## F4 — fixed

Changed the calibration procedure and regenerated `backend/app/analysis/calibration.json` and `benchmarks/post-S10.json`. The fitted intercept is lowered until the contract’s false-positive gate holds.

Final fused manipulated rates are `authentic_recompress: 0.0625` (before `0.0`), `resize_then_save: 0.0` (before `0.5`), and `real_camera: 0.0` (before `1.0`). The resize and camera traps no longer fail.

## F5 — partially-fixed

Changed the corpus generator/sidecars, `scripts/calibrate.py`, and `backend/app/analysis/exif.py`. EXIF-bearing generated outputs are now applicable; the EXIF detector records its raw evidence before `base.to_probability()`. C2PA remains correctly NOT_APPLICABLE because no signed manifest was fabricated.

Final benchmark counts are EXIF synthetic `n_applicable: 100` / `n_not_applicable: 0`, versus `0 / 100` before. C2PA remains `0 / 100`, with `score: None` and `state: not_applicable`, as required. The calibration no longer assigns a fitted C2PA observation or nonzero weight.

## F6 — blocked

Changed `docs/corpus.md` to state the verified shortfall. No unverified real files, URLs, checksums, licenses, or camera claims were added.

Before and after `data/corpus/MANIFEST.yaml` remain `real_camera: 1`, `real_ai: 0`, `real_c2pa_signed: 0`; the required counts are `12`, `12`, and `2`. The missing facts are independently verifiable source URLs plus bytes, SHA-256, attribution/license, and axis evidence for the remaining entries.

The exact discovery command tried was:

```sh
curl -sS --max-time 5 -H 'User-Agent: image-tamper-detector-corpus/1.0' \
  'https://commons.wikimedia.org/w/api.php?action=query&titles=File:Farmers_Market_in_Brazil.jpg&prop=imageinfo&iiprop=url%7Csize%7Cmime%7Cextmetadata&format=json' >/tmp/itd-wikimedia.json
```

It returned `curl: (6) Could not resolve host: commons.wikimedia.org` and exit `6`. `.venv/bin/python scripts/fetch_corpus.py --check` verifies the one existing manifest entry only. S05 and S08 are therefore marked failed.

## F7 — fixed

Changed `scripts/benchmark.py` and regenerated `benchmarks/post-S10.json`/`.md`. Benchmark rows now carry measured applicable-detector durations, learned is included through `registry.get_all()`, and calibration records actual corpus names.

Before, every detector/corpus `mean_duration_ms` was `0.0` and learned was absent. After, applicable synthetic detectors report a deterministic `500.0 ms` bucket (N/A detectors correctly remain `0.0`); learned is present with synthetic `n_applicable: 0`, `n_not_applicable: 100`, and AUC `null`. `fitted_on.corpora` is now `['real', 'synthetic']`, not `['all']`.

## F8 — partially-fixed

Changed `.python-version` from `3.13.13` to `3.14.7`. `Dockerfile` and `.github/workflows/ci.yml` already targeted Python 3.14 and were unchanged. `requirements.txt` was restored with its locked-version documentation; pins were not changed.

The existing `.venv/bin/python --version` still reports `3.13.13`. The exact clean-install check tried was:

```sh
test ! -e /tmp/itd-py314-check && python3 -m venv /tmp/itd-py314-check && /tmp/itd-py314-check/bin/pip install -q -r requirements.txt
```

Here `python3 --version` is `3.14.7`, but pip failed before dependency resolution with a PyPI `NameResolutionError` for `pypi.org`. The missing fact is whether the pinned stack installs/imports in a clean Python 3.14 environment; network/DNS prevented proving it. No amendment to D1 was made because this is an external network failure, not evidence of package incompatibility. S00 is marked failed until a clean 3.14 venv is actually established.

## F9 — fixed

Changed `requirements.txt` only. Restored the resolution date, lock-file reference, Python target, package groupings, and importing-module comments while preserving the pinned package set.

Before, the file was an 11-line bare pin list. After, it contains the lock-file header and per-package import comments; the six removed packages remain absent.

## Status changes

`plan/STATUS.yaml` now marks S00, S05, and S08 `failed` with notes naming F8 or F6. Other stages remain `passed` because their current acceptance properties pass or their documented reduced/NOT_APPLICABLE path is valid. No commit was created.
