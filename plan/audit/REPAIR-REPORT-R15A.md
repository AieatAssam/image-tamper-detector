# Repair report R15A: matched-pair evaluation

Date: 2026-08-31

Status: **BLOCKED for trustworthy AI-generation measurement.** The
deterministic matcher and matched benchmark mode are implemented, but the
available corpus cannot supply a sufficiently metadata-balanced subset. The
largest tested subset whose held-out shortcut check returns pass still has
pooled metadata AUC 0.8000 +/- 0.1485.

No downloads were performed. data/corpus/MANIFEST.yaml,
backend/app/analysis/calibration.json, data/samples/, and all detector
modules were left unchanged.

## Files changed

- scripts/matched_eval.py
- scripts/benchmark.py, matched mode only
- backend/tests/test_matched_eval.py
- plan/audit/REPAIR-REPORT-R15A.md

## Matching method

The matcher reads the existing manifest through scripts.benchmark._real().
Its AI scope is the two Round 10 generation axes, sd35_flux and synthbuster,
containing 390 images. Its authentic pool is the imd2020 and real_camera
axes, containing 212 images. real_c2pa_signed was excluded because it is a
provenance fixture rather than a camera-original negative.

Format and EXIF presence are exact hard constraints. For candidates that meet
both, the distance is:

~~~text
D = 4 * abs(log2(file_size_ratio))
    + max(abs(log2(width_ratio)), abs(log2(height_ratio)))
    + 2 * abs(log2(aspect_ratio_ratio))
~~~

The file-size term is weighted four times, dimensions once, and aspect ratio
twice. This makes the R14-proven file-size shortcut the primary nearest-neighbor
criterion while retaining decoded width, height, and aspect ratio in the
distance. A candidate is accepted when D <= tolerance. The final tolerance is
4.0; the maximum accepted distance was 3.863217190912768.

AI rows are sorted by manifest ID, shuffled with NumPy's
default_rng(20260828), and each row receives the nearest unused authentic row.
Ties are resolved by authentic ID. A real row is removed immediately after
use. The output JSONL records both rows of each pair with a shared
source_image/matched_pair_id, so source-group calculations and later
reproduction are deterministic.

## Why the sample collapses

All 212 authentic candidates are JPEG. The 120 SD3.5/Flux images are PNG.
Synthbuster contributes 240 PNG images and 30 JPEG images; all 30 JPEG rows
are firefly and have no EXIF. Consequently, exact format and EXIF matching
can only consider those 30 Firefly positives against the EXIF-free authentic
JPEGs. At tolerance 4.0, only five pairs survive:

| AI row | Authentic row | Distance |
|---|---|---:|
| synthbuster_firefly_r1c4d4720t | imd2020_1bajww_c956yew_0_authentic | 2.314262 |
| synthbuster_firefly_r0515a051t | imd2020_1ayilh_c91z2wh_0_authentic | 1.644557 |
| synthbuster_firefly_r1bf00696t | imd2020_1cdvuj_c9fo87w_0_authentic | 3.696762 |
| synthbuster_firefly_r1426cde6t | imd2020_1cfo34_c9gd5k9_0_authentic | 3.863217 |
| synthbuster_firefly_r0b17fd57t | imd2020_1bs5ew_c99k5bh_0_authentic | 3.853688 |

At the initial tolerance 6.0, 15 pairs survived, but the metadata gate
selected file size with held-out AUC 0.8750 +/- 0.1367 and pooled AUC
0.9667 +/- 0.0345. Tightening to 4.0 is the largest tested tolerance that
makes the existing held-out-only --check return success. It does not make the
pooled metadata distribution near chance.

## Matched-subset shortcut gate

The final command was:

~~~text
.venv/bin/python scripts/matched_eval.py \
  --out /tmp/r15a-matched-t4.jsonl \
  --seed 20260828 \
  --tolerance 4.0

.venv/bin/python scripts/check_format_shortcut.py \
  --manifest /tmp/r15a-matched-t4.jsonl \
  --out /tmp/r15a-format-t4.json \
  --check
# exit 0
~~~

The gate result was:

| Population | AUC +/- SE | Positive | Negative | Selected feature |
|---|---:|---:|---:|---|
| train | 1.0000 +/- 0.0000 | 3 | 3 | file_size |
| held-out | 0.5000 +/- 0.3227 | 2 | 2 | same |
| pooled | 0.8000 +/- 0.1485 | 5 | 5 | same |

The script's current acceptance return code is based on held-out AUC, so it
returns zero here. The pooled result is not near chance and the training
result is perfect. Therefore this report does not call the corpus
metadata-balanced. The result is a blocked measurement with a very large
standard error, not a detector-performance result.

The five pair IDs are unique and the authentic IDs are unique. Running the
matcher twice with seed 20260828 produces the same pair mapping. The focused
regression suite verifies both properties.

## Detector evaluation

The matched benchmark command was:

~~~text
.venv/bin/python scripts/benchmark.py \
  --corpus matched \
  --matched-manifest /tmp/r15a-matched-t4.jsonl \
  --detectors spectral,entropy,cfa,learned,aeroblade,clip_probe,npr \
  --out /tmp/r15a-matched-t4-final.json
~~~

The benchmark evaluates all five pairs, but reports AUC only where at least
ten applicable image rows exist, matching the existing benchmark contract.
The only generator with any matched positives is Firefly, so aggregate and
Firefly values are identical:

| Detector | Matched AUC +/- SE | Applicable positives / negatives | Status |
|---|---:|---:|---|
| spectral | 0.640000 +/- 0.182647 | 5 / 5 | provisional only |
| entropy | 0.400000 +/- 0.187006 | 5 / 5 | provisional only |
| cfa | N/A | 0 / 0 | not applicable |
| learned | N/A | 2 / 3 | too few applicable positives |
| aeroblade | N/A | 4 / 5 | too few applicable positives |
| clip_probe | 0.760000 +/- 0.159833 | 5 / 5 | provisional only |
| npr | 1.000000 +/- 0.000000 | 5 / 5 | provisional only |

The N/A rows are abstentions or insufficient populations, not zero scores.
The NPR perfect rank separation is especially not interpretable on five pairs.

### Per-generator before and after

The before column is the R12 unpaired Round 10-style result, using the
real-camera negative scope. The after column is the five-pair matched result.
Because only Firefly survives, all other after values are N/A for zero matched
pairs. These are not causal before/after estimates because the negative pool
and sample size changed.

| Generator | spectral | entropy | cfa | learned | aeroblade | clip_probe | npr |
|---|---|---|---|---|---|---|---|
| FLUX.1-schnell | 0.558333 +/- 0.088786 -> N/A | 0.598611 +/- 0.085719 -> N/A | N/A -> N/A | 0.470588 +/- 0.151380 -> N/A | 0.668056 +/- 0.078699 -> N/A | 1.000000 +/- 0.000000 -> N/A | 0.534722 +/- 0.090242 -> N/A |
| stable-diffusion-3.5-medium | 0.545833 +/- 0.089588 -> N/A | 0.505556 +/- 0.091688 -> N/A | N/A -> N/A | 0.410000 +/- 0.149289 -> N/A | 0.536111 +/- 0.090163 -> N/A | 1.000000 +/- 0.000000 -> N/A | 0.281944 +/- 0.088854 -> N/A |
| dalle2 | 0.558333 +/- 0.097316 -> N/A | 0.661111 +/- 0.088967 -> N/A | N/A -> N/A | 0.533333 +/- 0.165241 -> N/A | 0.602778 +/- 0.094340 -> N/A | 1.000000 +/- 0.000000 -> N/A | 0.122222 +/- 0.068520 -> N/A |
| dalle3 | 0.572222 +/- 0.096489 -> N/A | 0.294444 +/- 0.094573 -> N/A | N/A -> N/A | 0.466667 +/- 0.159056 -> N/A | 0.500000 +/- 0.099768 -> N/A | 1.000000 +/- 0.000000 -> N/A | 0.308333 +/- 0.095668 -> N/A |
| firefly | 0.605556 +/- 0.094122 -> 0.640000 +/- 0.182647 | 0.536111 +/- 0.098445 -> 0.400000 +/- 0.187006 | N/A -> N/A | 0.433333 +/- 0.159139 -> N/A (2/3) | 0.547222 +/- 0.097910 -> N/A (4/5) | 1.000000 +/- 0.000000 -> 0.760000 +/- 0.159833 | 0.513889 +/- 0.099334 -> 1.000000 +/- 0.000000 |
| glide | 0.963889 +/- 0.026766 -> N/A | 0.811111 +/- 0.066955 -> N/A | N/A -> N/A | N/A -> N/A | 0.802778 +/- 0.068520 -> N/A | 1.000000 +/- 0.000000 -> N/A | 0.688889 +/- 0.085809 -> N/A |
| midjourney-v5 | 0.608333 +/- 0.093900 -> N/A | 0.369444 +/- 0.099118 -> N/A | N/A -> N/A | 0.383333 +/- 0.157600 -> N/A | 0.636111 +/- 0.091476 -> N/A | 1.000000 +/- 0.000000 -> N/A | 0.522222 +/- 0.099029 -> N/A |
| stable-diffusion-1-3 | 0.569444 +/- 0.096662 -> N/A | 0.680556 +/- 0.086798 -> N/A | N/A -> N/A | N/A -> N/A | 0.386111 +/- 0.099694 -> N/A | 1.000000 +/- 0.000000 -> N/A | 0.141667 +/- 0.072960 -> N/A |
| stable-diffusion-1-4 | 0.536111 +/- 0.098445 -> N/A | 0.586111 +/- 0.095568 -> N/A | N/A -> N/A | N/A -> N/A | 0.397222 +/- 0.099995 -> N/A | 1.000000 +/- 0.000000 -> N/A | 0.138889 +/- 0.072357 -> N/A |
| stable-diffusion-2 | 0.502778 +/- 0.099689 -> N/A | 0.552778 +/- 0.097621 -> N/A | N/A -> N/A | 0.257143 +/- 0.141712 -> N/A | 0.219444 +/- 0.086409 -> N/A | 1.000000 +/- 0.000000 -> N/A | 0.105556 +/- 0.064254 -> N/A |
| stable-diffusion-xl | 0.700000 +/- 0.084435 -> N/A | 0.508333 +/- 0.099519 -> N/A | N/A -> N/A | 0.555556 +/- 0.163804 -> N/A | 0.519444 +/- 0.099135 -> N/A | 1.000000 +/- 0.000000 -> N/A | 0.266667 +/- 0.092011 -> N/A |

The CLIP result drops from 1.000000 +/- 0.000000 to
0.760000 +/- 0.159833, but the matched subset is too small and still fails
the pooled shortcut diagnostic. It does not establish that CLIP has
generation signal.

## Verification

Passed:

~~~text
.venv/bin/python -m pytest backend/tests/test_matched_eval.py \
  backend/tests/test_corpus.py \
  backend/tests/test_format_shortcut.py -q
# 8 passed

.venv/bin/python plan/validate.py
# All structural and shell-syntax checks passed.

git diff --check
# clean
~~~

No detector or calibration values were changed. The next missing fact is a
real authentic PNG pool, or a real and AI JPEG population with overlapping
dimensions, EXIF state, and byte sizes. Without that, matched-pair evaluation
cannot produce a statistically useful, metadata-balanced result for the
current AI axes.
