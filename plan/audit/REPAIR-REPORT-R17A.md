# Round 17A repair report

Date: 2026-09-01

## Status

| item | status | evidence |
|---|---|---|
| expanded parity negative pool | COMPLETE | 402 AI rows and 212 authentic rows; all 614 exact 120,000-byte JPEGs |
| metadata shortcut acceptance gate | PASS | pooled and held-out AUC 0.500 for all five feature groups; every check exited 0 |
| seven-detector parity measurement | COMPLETE / N/A where scoped out | AEROBLADE, CLIP, learned, spectral, entropy, and NPR completed; CFA is native-only and therefore N/A on parity |
| calibration promotion | BLOCKED / not promoted | the parity pool is unpaired by source, and the all-detector calibration path did not yield a safe AEROBLADE calibration; the committed artifact remains unchanged |

No detector, threshold, weight, or AUC was tuned. No image bytes are tracked. `scripts/parity_encode.py` itself was reused unchanged.

## 1. Pool construction and acceptance gate

Before writing the first parity byte, this ignore check passed:

```text
git check-ignore -v data/corpus/real/r17a-parity-120k data/corpus/real/r17a-parity-120k/images
.gitignore:8:data/corpus/real/*  data/corpus/real/r17a-parity-120k
.gitignore:8:data/corpus/real/*  data/corpus/real/r17a-parity-120k/images
```

The final negative encode was:

```sh
.venv/bin/python scripts/parity_encode.py \
  --manifest /tmp/r17a-negative-input.jsonl \
  --out data/corpus/real/r17a-parity-120k \
  --target-bytes 120000 --seed 20260831 --canvas-size 1024 \
  --tolerance-bytes 100000
```

It produced 212 negative rows, all JPEG, 1024x1024, EXIF-free, and exactly 120,000 bytes. The chosen quality values ranged from 22 to 100 (mean 78.8915). The final ignored parity set contains 402 copied R15C AI parity rows plus those 212 newly encoded authentic rows. The twelve camera negatives retain `source_axis: real_camera`; their parity representation is listed under `axis: imd2020` because the existing real-camera fetch contract correctly rejects EXIF-free parity files. Native rows and native files remain intact.

A first attempt with the inherited 20,000-byte tolerance stopped before completion with the exact error `ValueError: target 120000 is not reachable within 20000 bytes (nearest quality=100, size=40469)`. The final tolerance is recorded in the command and is not a label-dependent selection.

The gate was run from the explicit manifest rows via temporary absolute-path JSONL files:

```sh
for feature in all format dimensions file_size exif; do
  .venv/bin/python scripts/check_format_shortcut.py \
    --manifest /tmp/r17a-manifest-gate-${feature}.jsonl \
    --features "$feature" --seed 20260828 --check \
    --out /tmp/r17a-gate-manifest-${feature}.json
done
```

Each JSONL has 614 parity rows: 402 positives and 212 negatives. The checker uses its fixed 70/30 source-group split and the existing 0.55 acceptance limit.

| feature group | train AUC ± SE | held-out AUC ± SE | pooled AUC ± SE | result |
|---|---:|---:|---:|---|
| all | 0.5000 ± 0.0294 | 0.5000 ± 0.0447 | 0.5000 ± 0.0245 | PASS |
| format | 0.5000 ± 0.0294 | 0.5000 ± 0.0447 | 0.5000 ± 0.0245 | PASS |
| dimensions | 0.5000 ± 0.0294 | 0.5000 ± 0.0447 | 0.5000 ± 0.0245 | PASS |
| file_size | 0.5000 ± 0.0294 | 0.5000 ± 0.0447 | 0.5000 ± 0.0245 | PASS |
| exif | 0.5000 ± 0.0294 | 0.5000 ± 0.0447 | 0.5000 ± 0.0245 | PASS |

The three per-axis gate runs (`real_ai`, `sd35_flux`, and `synthbuster`) were also exactly 0.500 pooled for every feature group. The gate therefore passed before detector measurement. `fetch_corpus.py --check` subsequently verified all 1,430 manifest entries.

The detector JSON artifacts are `/tmp/r17a-{aeroblade,clip_probe,learned,spectral,entropy,npr}-parity.json` and `/tmp/r17a-cfa-parity.json`.

## 2. Pooled AI-axis measurements

AUC is the existing tie-aware rank statistic; SE is the Hanley–McNeil standard error. The “before” column is Round 16A’s 12-negative parity measurement (for spectral/entropy/NPR, the completed R16B artifacts on the same R16A parity set). The “after” column uses the 212 authentic parity negatives. Learned counts are conditional on its image-derived face gate.

| detector | R16A before AUC ± SE | R17A after AUC ± SE | before nAI/nR | after nAI/nR |
|---|---:|---:|---:|---:|
| AEROBLADE | 0.416000 ± 0.088000 | 0.329931 ± 0.023574 | 402/12 | 402/212 |
| CLIP probe | 0.999585 ± 0.000757 | 0.801464 ± 0.017354 | 402/12 | 402/212 |
| learned | 0.183824 ± 0.130905 | 0.085647 ± 0.018639 | 136/4 | 134/125 |
| spectral | 0.508085 ± 0.084227 | 0.547111 ± 0.024151 | 402/12 | 402/212 |
| entropy | 0.761609 ± 0.055874 | 0.695227 ± 0.021225 | 402/12 | 402/212 |
| NPR | 0.280265 ± 0.084618 | 0.304715 ± 0.023105 | 402/12 | 402/212 |
| CFA | N/A (native-only scope) | N/A (native-only scope; 212 parity negatives not scoreable) | 0/12 applicable | 0/212 applicable |

The headline change is CLIP: `0.999585` with twelve negatives to `0.801464` with 212. It no longer has its old trivial result, but it still separates this set materially. This is a measurement result, not proof that it is generation-specific: all parity files use the same JPEG save pipeline, but the frozen model can still respond to content and residual quantization/composition differences. AEROBLADE remains below chance (`0.416` to `0.330`).

## 3. Per-generator AUC

Cells are `AUC ± SE (nAI/nR)`. `nAI` and `nR` are applicable scored rows, not merely manifest rows. The 14 named generator rows include the three one-image `real_ai` examples; their large SE is therefore expected. Nine additional `real_ai` positives have no generator field, so they remain in pooled AUC but cannot have an honest per-generator row. The new `nR=212` applies to every fully applicable detector. Learned uses the 125 face-detected authentic negatives; CFA has no parity-applicable rows.

### AEROBLADE

| axis | generator | R16A nR=12 | R17A nR=212 pool |
|---|---|---:|---:|
| real_ai | Midjourney v4 | 0.167 ± 0.173 (1/12) | 0.113 ± 0.103 (1/212) |
| real_ai | Stable Diffusion 3.5 Large | 0.583 ± 0.320 (1/12) | 0.429 ± 0.273 (1/212) |
| real_ai | xAI Aurora | 0.667 ± 0.316 (1/12) | 0.585 ± 0.301 (1/212) |
| sd35_flux | FLUX.1-schnell | 0.456 ± 0.093 (60/12) | 0.361 ± 0.038 (60/212) |
| sd35_flux | stable-diffusion-3.5-medium | 0.307 ± 0.091 (60/12) | 0.228 ± 0.030 (60/212) |
| synthbuster | dalle2 | 0.389 ± 0.100 (30/12) | 0.301 ± 0.045 (30/212) |
| synthbuster | dalle3 | 0.325 ± 0.097 (30/12) | 0.226 ± 0.038 (30/212) |
| synthbuster | firefly | 0.481 ± 0.100 (30/12) | 0.381 ± 0.051 (30/212) |
| synthbuster | glide | 0.756 ± 0.077 (30/12) | 0.679 ± 0.057 (30/212) |
| synthbuster | midjourney-v5 | 0.647 ± 0.090 (30/12) | 0.538 ± 0.057 (30/212) |
| synthbuster | stable-diffusion-1-3 | 0.189 ± 0.082 (30/12) | 0.122 ± 0.025 (30/212) |
| synthbuster | stable-diffusion-1-4 | 0.181 ± 0.081 (30/12) | 0.117 ± 0.024 (30/212) |
| synthbuster | stable-diffusion-2 | 0.264 ± 0.092 (30/12) | 0.190 ± 0.034 (30/212) |
| synthbuster | stable-diffusion-xl | 0.586 ± 0.096 (30/12) | 0.483 ± 0.056 (30/212) |

### CLIP probe

| axis | generator | R16A nR=12 | R17A nR=212 pool |
|---|---|---:|---:|
| real_ai | Midjourney v4 | 1.000 ± 0.000 (1/12) | 0.311 ± 0.227 (1/212) |
| real_ai | Stable Diffusion 3.5 Large | 1.000 ± 0.000 (1/12) | 0.552 ± 0.298 (1/212) |
| real_ai | xAI Aurora | 1.000 ± 0.000 (1/12) | 0.684 ± 0.297 (1/212) |
| sd35_flux | FLUX.1-schnell | 1.000 ± 0.000 (60/12) | 0.837 ± 0.034 (60/212) |
| sd35_flux | stable-diffusion-3.5-medium | 1.000 ± 0.000 (60/12) | 0.802 ± 0.036 (60/212) |
| synthbuster | dalle2 | 1.000 ± 0.000 (30/12) | 0.789 ± 0.051 (30/212) |
| synthbuster | dalle3 | 1.000 ± 0.000 (30/12) | 0.891 ± 0.040 (30/212) |
| synthbuster | firefly | 1.000 ± 0.000 (30/12) | 0.786 ± 0.051 (30/212) |
| synthbuster | glide | 1.000 ± 0.000 (30/12) | 0.850 ± 0.045 (30/212) |
| synthbuster | midjourney-v5 | 1.000 ± 0.000 (30/12) | 0.811 ± 0.049 (30/212) |
| synthbuster | stable-diffusion-1-3 | 1.000 ± 0.000 (30/12) | 0.768 ± 0.052 (30/212) |
| synthbuster | stable-diffusion-1-4 | 1.000 ± 0.000 (30/12) | 0.828 ± 0.048 (30/212) |
| synthbuster | stable-diffusion-2 | 1.000 ± 0.000 (30/12) | 0.764 ± 0.053 (30/212) |
| synthbuster | stable-diffusion-xl | 1.000 ± 0.000 (30/12) | 0.781 ± 0.052 (30/212) |

### learned

| axis | generator | R16A nR=12 | R17A nR=212 pool |
|---|---|---:|---:|
| real_ai | Midjourney v4 | N/A (no applicable face-positive) | N/A (no applicable face-positive) |
| real_ai | Stable Diffusion 3.5 Large | N/A (no applicable face-positive) | N/A (no applicable face-positive) |
| real_ai | xAI Aurora | N/A (no applicable face-positive) | N/A (no applicable face-positive) |
| sd35_flux | FLUX.1-schnell | 0.125 ± 0.119 (18/4) | 0.040 ± 0.016 (18/125) |
| sd35_flux | stable-diffusion-3.5-medium | 0.131 ± 0.120 (21/4) | 0.045 ± 0.016 (21/125) |
| synthbuster | dalle2 | 0.175 ± 0.141 (10/4) | 0.054 ± 0.022 (10/125) |
| synthbuster | dalle3 | 0.208 ± 0.148 (12/4) | 0.099 ± 0.032 (12/125) |
| synthbuster | firefly | 0.205 ± 0.148 (11/4) | 0.121 ± 0.039 (11/125) |
| synthbuster | glide | 0.200 ± 0.148 (10/4) | 0.058 ± 0.023 (10/125) |
| synthbuster | midjourney-v5 | 0.250 ± 0.162 (9/4) | 0.130 ± 0.044 (9/125) |
| synthbuster | stable-diffusion-1-3 | 0.091 ± 0.106 (11/4) | 0.034 ± 0.016 (11/125) |
| synthbuster | stable-diffusion-1-4 | 0.150 ± 0.133 (10/4) | 0.046 ± 0.020 (10/125) |
| synthbuster | stable-diffusion-2 | 0.227 ± 0.154 (11/4) | 0.113 ± 0.037 (11/125) |
| synthbuster | stable-diffusion-xl | 0.364 ± 0.173 (11/4) | 0.258 ± 0.065 (11/125) |

### spectral

| axis | generator | R16A nR=12 | R17A nR=212 pool |
|---|---|---:|---:|
| real_ai | Midjourney v4 | 0.000 ± 0.000 (1/12) | 0.019 ± 0.021 (1/212) |
| real_ai | Stable Diffusion 3.5 Large | 0.000 ± 0.000 (1/12) | 0.033 ± 0.034 (1/212) |
| real_ai | xAI Aurora | 0.500 ± 0.312 (1/12) | 0.547 ± 0.297 (1/212) |
| sd35_flux | FLUX.1-schnell | 0.426 ± 0.094 (60/12) | 0.477 ± 0.042 (60/212) |
| sd35_flux | stable-diffusion-3.5-medium | 0.419 ± 0.094 (60/12) | 0.467 ± 0.042 (60/212) |
| synthbuster | dalle2 | 0.419 ± 0.100 (30/12) | 0.475 ± 0.056 (30/212) |
| synthbuster | dalle3 | 0.464 ± 0.100 (30/12) | 0.509 ± 0.057 (30/212) |
| synthbuster | firefly | 0.539 ± 0.098 (30/12) | 0.572 ± 0.058 (30/212) |
| synthbuster | glide | 0.872 ± 0.054 (30/12) | 0.862 ± 0.044 (30/212) |
| synthbuster | midjourney-v5 | 0.572 ± 0.096 (30/12) | 0.608 ± 0.058 (30/212) |
| synthbuster | stable-diffusion-1-3 | 0.456 ± 0.101 (30/12) | 0.512 ± 0.057 (30/212) |
| synthbuster | stable-diffusion-1-4 | 0.417 ± 0.100 (30/12) | 0.483 ± 0.056 (30/212) |
| synthbuster | stable-diffusion-2 | 0.497 ± 0.100 (30/12) | 0.533 ± 0.057 (30/212) |
| synthbuster | stable-diffusion-xl | 0.708 ± 0.083 (30/12) | 0.714 ± 0.055 (30/212) |

### entropy

| axis | generator | R16A nR=12 | R17A nR=212 pool |
|---|---|---:|---:|
| real_ai | Midjourney v4 | 1.000 ± 0.000 (1/12) | 0.915 ± 0.193 (1/212) |
| real_ai | Stable Diffusion 3.5 Large | 0.917 ± 0.200 (1/12) | 0.858 ± 0.238 (1/212) |
| real_ai | xAI Aurora | 1.000 ± 0.000 (1/12) | 1.000 ± 0.000 (1/212) |
| sd35_flux | FLUX.1-schnell | 0.965 ± 0.020 (60/12) | 0.899 ± 0.028 (60/212) |
| sd35_flux | stable-diffusion-3.5-medium | 0.764 ± 0.065 (60/12) | 0.686 ± 0.041 (60/212) |
| synthbuster | dalle2 | 0.925 ± 0.040 (30/12) | 0.861 ± 0.044 (30/212) |
| synthbuster | dalle3 | 0.900 ± 0.047 (30/12) | 0.834 ± 0.047 (30/212) |
| synthbuster | firefly | 0.714 ± 0.083 (30/12) | 0.611 ± 0.058 (30/212) |
| synthbuster | glide | 0.997 ± 0.007 (30/12) | 0.982 ± 0.017 (30/212) |
| synthbuster | midjourney-v5 | 0.414 ± 0.100 (30/12) | 0.315 ± 0.046 (30/212) |
| synthbuster | stable-diffusion-1-3 | 0.911 ± 0.044 (30/12) | 0.850 ± 0.045 (30/212) |
| synthbuster | stable-diffusion-1-4 | 0.933 ± 0.037 (30/12) | 0.867 ± 0.043 (30/212) |
| synthbuster | stable-diffusion-2 | 0.272 ± 0.093 (30/12) | 0.227 ± 0.038 (30/212) |
| synthbuster | stable-diffusion-xl | 0.361 ± 0.099 (30/12) | 0.303 ± 0.045 (30/212) |

### NPR

| axis | generator | R16A nR=12 | R17A nR=212 pool |
|---|---|---:|---:|
| real_ai | Midjourney v4 | 0.167 ± 0.173 (1/12) | 0.264 ± 0.203 (1/212) |
| real_ai | Stable Diffusion 3.5 Large | 0.167 ± 0.173 (1/12) | 0.255 ± 0.198 (1/212) |
| real_ai | xAI Aurora | 1.000 ± 0.000 (1/12) | 0.901 ± 0.206 (1/212) |
| sd35_flux | FLUX.1-schnell | 0.258 ± 0.087 (60/12) | 0.288 ± 0.034 (60/212) |
| sd35_flux | stable-diffusion-3.5-medium | 0.100 ± 0.061 (60/12) | 0.120 ± 0.021 (60/212) |
| synthbuster | dalle2 | 0.200 ± 0.084 (30/12) | 0.238 ± 0.039 (30/212) |
| synthbuster | dalle3 | 0.150 ± 0.075 (30/12) | 0.144 ± 0.028 (30/212) |
| synthbuster | firefly | 0.358 ± 0.099 (30/12) | 0.434 ± 0.054 (30/212) |
| synthbuster | glide | 0.750 ± 0.077 (30/12) | 0.742 ± 0.054 (30/212) |
| synthbuster | midjourney-v5 | 0.553 ± 0.098 (30/12) | 0.609 ± 0.058 (30/212) |
| synthbuster | stable-diffusion-1-3 | 0.069 ± 0.053 (30/12) | 0.057 ± 0.015 (30/212) |
| synthbuster | stable-diffusion-1-4 | 0.067 ± 0.052 (30/12) | 0.060 ± 0.015 (30/212) |
| synthbuster | stable-diffusion-2 | 0.247 ± 0.090 (30/12) | 0.291 ± 0.044 (30/212) |
| synthbuster | stable-diffusion-xl | 0.475 ± 0.100 (30/12) | 0.515 ± 0.057 (30/212) |

### CFA

| axis | generator | R16A | R17A parity |
|---|---|---:|---:|
| real_ai | Midjourney v4 | N/A | N/A — native-only detector scope |
| real_ai | Stable Diffusion 3.5 Large | N/A | N/A — native-only detector scope |
| real_ai | xAI Aurora | N/A | N/A — native-only detector scope |
| sd35_flux | FLUX.1-schnell | N/A | N/A — native-only detector scope |
| sd35_flux | stable-diffusion-3.5-medium | N/A | N/A — native-only detector scope |
| synthbuster | dalle2 | N/A | N/A — native-only detector scope |
| synthbuster | dalle3 | N/A | N/A — native-only detector scope |
| synthbuster | firefly | N/A | N/A — native-only detector scope |
| synthbuster | glide | N/A | N/A — native-only detector scope |
| synthbuster | midjourney-v5 | N/A | N/A — native-only detector scope |
| synthbuster | stable-diffusion-1-3 | N/A | N/A — native-only detector scope |
| synthbuster | stable-diffusion-1-4 | N/A | N/A — native-only detector scope |
| synthbuster | stable-diffusion-2 | N/A | N/A — native-only detector scope |
| synthbuster | stable-diffusion-xl | N/A | N/A — native-only detector scope |

For learned, the R17A negative pool has 125 applicable faces out of 212. For generators with no applicable positive face rows, N/A is the precondition result, not an imputed score.

## 4. Calibration plumbing

`scripts/benchmark.py` now selects authentic `real_camera` rows and parity `imd2020` rows as the negative scope, yielding 212 negatives for the parity detector runs. `scripts/calibrate.py` makes the same parity-negative rows available to the AI-axis guard and records the scope dynamically as `real_camera+imd2020(parity)` when such rows are scored. No corpus-membership condition was added to any detector.

I ran:

```sh
.venv/bin/python scripts/calibrate.py --corpus real --variant parity --out /tmp/r17a-calibration.json --seed 20260828
```

The command completed, but it was not promoted to `backend/app/analysis/calibration.json`: the parity AI rows and the independent authentic pool have no paired `source_image`, so the source-local held-out AUC is necessarily unavailable; additionally the concurrent all-detector runner returned no applicable AEROBLADE raw values in that run, producing an invalid zero/one fallback configuration. Promoting it would fabricate calibration confidence. The committed calibration JSON is intentionally unchanged and still carries its earlier 12-negative native-legacy guard. A safe future promotion needs either a serial/precomputed raw-score calibration path for the optional models or parity positives and negatives with a valid calibration pairing design.

## 5. Files changed and verification

Changed by R17A:

- `data/corpus/MANIFEST.yaml`: 614 explicit parity rows, hashes, source groups, quality metadata, and source-axis provenance.
- `scripts/benchmark.py`: parity negative selection includes the 200 IMD2020 authentic rows plus the 12 parity camera-source rows.
- `scripts/calibrate.py`: parity negative rows participate in AI-axis guard accounting; no fallback fit is used when source pairing is absent.
- `plan/audit/REPAIR-REPORT-R17A.md`: this report.
- `scripts/parity_encode.py`: unchanged; existing deterministic encoder was sufficient.

Commands passing:

```text
.venv/bin/python scripts/fetch_corpus.py --check
# 1430 manifest entries verified
.venv/bin/python -m py_compile scripts/benchmark.py scripts/calibrate.py scripts/parity_encode.py
.venv/bin/python scripts/check_format_shortcut.py ... --check  # all five feature groups exit 0
```

No commit was made. Native rows and `data/samples/` were not edited; parity image bytes remain under the ignored `data/corpus/real/*` path.
