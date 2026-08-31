# Round 12 repair report

Date: 2026-08-31

## Result

Round 12 activates the permitted optional torch path without changing the
default installation. The torch-free degradation gate passes. AEROBLADE now
uses distilled TAESD plus LPIPS and is measurable on all 402 AI rows, but its
calibration guard leaves it at zero fusion weight. The frozen CLIP ViT-L/14
probe generalizes across the held-out generators on this corpus, with
1.000000 +/- 0.000000 ID and OOD AUC. That perfect separation is explicitly
qualified: every AI row is PNG and every strict camera negative is JPEG, so a
format/domain shortcut remains plausible.

No image bytes, model bytes, or commit were added to the repository. The
pre-existing user change to plan/plan.yaml adding D5b was preserved.

The benchmark below uses all 916 available rows: 100 synthetic rows, 816 real
manifest rows, 402 generated AI rows across 11 named generators, and 12 strict
real_camera negatives. AUC is rank AUC. +/- is the Hanley-McNeil standard
error, not a confidence interval. AI rows and camera negatives are unpaired
across source images.

## Per-agent status

| Agent/item | Status | Result |
|---|---|---|
| AGENT-EXTRA | COMPLETE | torch==2.13.0, torchvision==0.28.0, diffusers==0.40.0, open-clip-torch==3.3.0, and lpips==0.1.4 are optional pins in requirements-learned.txt; base requirements, CI, and Docker remain unchanged. |
| AGENT-AEROBLADE | COMPLETE | Distilled MIT TAESD + LPIPS path runs end to end; missing optional pieces return NOT_APPLICABLE; calibrated fusion weight is 0.0. |
| AGENT-CLIP | COMPLETE | Frozen LAION ViT-L/14 backbone plus linear probe; source-image grouping and complete generator holdout are recorded; OOD is the primary result. |
| AGENT-NPR-TRAINED | SKIPPED / OPTIONAL | Not attempted after the stronger CLIP result. No unlicensed NPR weights were used. |
| D6 exclusions | UNCHANGED | TruFor, Noiseprint++, NPR released weights, and Splicebuster reference code remain excluded for licensing reasons; torch does not alter this. |

## Degradation and model-path checks

The download directories were checked before model bytes were fetched:

~~~text
.gitignore:13:models  models/taesd
.gitignore:13:models  models/clip
.gitignore:13:models  models/onnx
~~~

The base environment built from requirements-dev.txt alone passed 85 passed,
1 warning. The full optional environment passed the same 85 passed,
1 warning. In the base environment, the torch-backed detectors remain lazy
and report NOT_APPLICABLE when their optional files are absent. pip check
passed for the optional environment.

scripts/fetch_model.py pins and verifies:

- madebyollin/taesd revision 614f76814bbe30edbe2e627ace1c2234c81a2c0e,
  including the Diffusers config and safetensors file;
- laion/CLIP-ViT-L-14-laion2B-s32B-b82K revision
  1627032197142fbe2a7cfec626f4ced3ae60d07a;
- the torchvision AlexNet cache used by LPIPS, SHA-256
  7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02.

The fetched model paths are ignored and were not staged.

## AEROBLADE

The old ONNX/L1 substitute was replaced with the permitted optional path:

~~~text
RGB image -> bounded multiple-of-eight resize -> [-1,1]
-> TAESD encode/decode -> LPIPS(input, reconstruction)
~~~

The detector computes reconstruction_lpips; lower error is more suspicious,
so higher_is_worse=false is physically and mathematically consistent. The
runtime imports torch, diffusers, and lpips lazily, uses local files only, and
returns NOT_APPLICABLE for a missing optional dependency or artifact. Its
reason explicitly says latent-diffusion-specific; its limitations state that
distilled TAESD is not the exact paper autoencoder and that the cue is useless
for splicing, copy-move, and GAN output.

### AEROBLADE per-generator AUC

Before Round 12, TAESD weights were absent, so every AI generator was
N/A (0 applicable images). After activation:

| Generator | Before | After AUC +/- SE |
|---|---:|---:|
| FLUX.1-schnell | N/A | 0.668056 +/- 0.078699 |
| stable-diffusion-3.5-medium | N/A | 0.536111 +/- 0.090163 |
| dalle2 | N/A | 0.602778 +/- 0.094340 |
| dalle3 | N/A | 0.500000 +/- 0.099768 |
| firefly | N/A | 0.547222 +/- 0.097910 |
| glide | N/A | 0.802778 +/- 0.068520 |
| midjourney-v5 | N/A | 0.636111 +/- 0.091476 |
| stable-diffusion-1-3 | N/A | 0.386111 +/- 0.099694 |
| stable-diffusion-1-4 | N/A | 0.397222 +/- 0.099995 |
| stable-diffusion-2 | N/A | 0.219444 +/- 0.086409 |
| stable-diffusion-xl | N/A | 0.519444 +/- 0.099135 |

The AI-axis screen is 0.539957 +/- 0.082240 from 390 generated positives
and 12 camera negatives. The source-paired calibration result is
0.511013 +/- 0.025584 from 260 positive and 250 negative participating
rows. The guard rule is AUC > 0.5 + SE; it therefore drops AEROBLADE to
weight 0.0. This is not a claim that the paper's exact model would perform
the same way.

## CLIP probe

scripts/fit_clip_probe.py fits only the linear probe using the repository's
standardized logistic routine. The CLIP backbone is frozen in evaluation mode.
Rows are grouped by the manifest's source_image/source_group field. The fixed
seed is 20260828; complete held-out generators are:

~~~text
glide
stable-diffusion-1-4
stable-diffusion-3.5-medium
stable-diffusion-xl
~~~

Training uses the other seven named generators and 171 rows. The test split
contains 31 source groups, including four held-out camera groups; this
stratified group selection is fixed before feature extraction and was not
tuned to scores. ID contains test rows from seen generators and those four
camera negatives. OOD contains test rows from the four unseen generators and
the same four held-out camera negatives. No source group occurs in training
and evaluation.

### CLIP aggregate result

| Evaluation | AUC +/- SE | Positive rows | Negative rows |
|---|---:|---:|---:|
| In distribution: seen generators | 1.000000 +/- 0.000000 | 77 | 4 |
| Out of distribution: unseen generators | 1.000000 +/- 0.000000 | 47 | 4 |

The Hanley-McNeil SE is zero at this observed perfect rank separation; it is
not evidence of zero population uncertainty. The benchmark's pooled
real-camera comparison also gives 1.000000 +/- 0.000000 for every named
generator with 12 negatives:

| Generator | AUC +/- SE | Generator split |
|---|---:|---|
| FLUX.1-schnell | 1.000000 +/- 0.000000 | in distribution |
| stable-diffusion-3.5-medium | 1.000000 +/- 0.000000 | out of distribution |
| dalle2 | 1.000000 +/- 0.000000 | in distribution |
| dalle3 | 1.000000 +/- 0.000000 | in distribution |
| firefly | 1.000000 +/- 0.000000 | in distribution |
| glide | 1.000000 +/- 0.000000 | out of distribution |
| midjourney-v5 | 1.000000 +/- 0.000000 | in distribution |
| stable-diffusion-1-3 | 1.000000 +/- 0.000000 | in distribution |
| stable-diffusion-1-4 | 1.000000 +/- 0.000000 | out of distribution |
| stable-diffusion-2 | 1.000000 +/- 0.000000 | in distribution |
| stable-diffusion-xl | 1.000000 +/- 0.000000 | out of distribution |

This is a measured corpus result, not an acceptance floor. Every AI row in
the measured axes is PNG, while every strict camera negative is JPEG. The
probe may be using a format/domain cue or another corpus shortcut. The next
scientifically useful check is re-encoded AI and camera material with matched
format and dimensions.

### CLIP runtime result

The registered clip_probe detector is image-side only. It checks decoded
format, minimum dimensions, and local probe/backbone files; it does not inspect
manifest axis, generator, or corpus membership. Missing dependencies or
weights return NOT_APPLICABLE. The fitted probe is ignored by git.

The recalibrated runtime clip_probe weight is 0.0021987867770255362
(unclipped 0.0023172509366955318). This small weight came from the existing
fusion calibration on source-paired rows; no weight was tuned from the CLIP
OOD result.

## All AI-capable detector results

The following is the full per-generator table from /tmp/post-r12.json.
cfa is correctly N/A on every AI row because its camera-JPEG/EXIF
precondition rejects these PNGs. The learned detector's N/A entries have fewer
than 10 applicable positive rows after its face gate; the parenthesized values
are applicable-positive / camera-negative counts.

| Generator | spectral | entropy | cfa | learned | aeroblade | clip_probe | npr |
|---|---:|---:|---:|---:|---:|---:|---:|
| FLUX.1-schnell | 0.558333 +/- 0.088786 | 0.598611 +/- 0.085719 | N/A (0/12) | 0.470588 +/- 0.151380 | 0.668056 +/- 0.078699 | 1.000000 +/- 0.000000 | 0.534722 +/- 0.090242 |
| stable-diffusion-3.5-medium | 0.545833 +/- 0.089588 | 0.505556 +/- 0.091688 | N/A (0/12) | 0.410000 +/- 0.149289 | 0.536111 +/- 0.090163 | 1.000000 +/- 0.000000 | 0.281944 +/- 0.088854 |
| dalle2 | 0.558333 +/- 0.097316 | 0.661111 +/- 0.088967 | N/A (0/12) | 0.533333 +/- 0.165241 | 0.602778 +/- 0.094340 | 1.000000 +/- 0.000000 | 0.122222 +/- 0.068520 |
| dalle3 | 0.572222 +/- 0.096489 | 0.294444 +/- 0.094573 | N/A (0/12) | 0.466667 +/- 0.159056 | 0.500000 +/- 0.099768 | 1.000000 +/- 0.000000 | 0.308333 +/- 0.095668 |
| firefly | 0.605556 +/- 0.094122 | 0.536111 +/- 0.098445 | N/A (0/12) | 0.433333 +/- 0.159139 | 0.547222 +/- 0.097910 | 1.000000 +/- 0.000000 | 0.513889 +/- 0.099334 |
| glide | 0.963889 +/- 0.026766 | 0.811111 +/- 0.066955 | N/A (0/12) | N/A (0/5) | 0.802778 +/- 0.068520 | 1.000000 +/- 0.000000 | 0.688889 +/- 0.085809 |
| midjourney-v5 | 0.608333 +/- 0.093900 | 0.369444 +/- 0.099118 | N/A (0/12) | 0.383333 +/- 0.157600 | 0.636111 +/- 0.091476 | 1.000000 +/- 0.000000 | 0.522222 +/- 0.099029 |
| stable-diffusion-1-3 | 0.569444 +/- 0.096662 | 0.680556 +/- 0.086798 | N/A (0/12) | N/A (2/5) | 0.386111 +/- 0.099694 | 1.000000 +/- 0.000000 | 0.141667 +/- 0.072960 |
| stable-diffusion-1-4 | 0.536111 +/- 0.098445 | 0.586111 +/- 0.095568 | N/A (0/12) | N/A (2/5) | 0.397222 +/- 0.099995 | 1.000000 +/- 0.000000 | 0.138889 +/- 0.072357 |
| stable-diffusion-2 | 0.502778 +/- 0.099689 | 0.552778 +/- 0.097621 | N/A (0/12) | 0.257143 +/- 0.141712 | 0.219444 +/- 0.086409 | 1.000000 +/- 0.000000 | 0.105556 +/- 0.064254 |
| stable-diffusion-xl | 0.700000 +/- 0.084435 | 0.508333 +/- 0.099519 | N/A (0/12) | 0.555556 +/- 0.163804 | 0.519444 +/- 0.099135 | 1.000000 +/- 0.000000 | 0.266667 +/- 0.092011 |

The full post-activation benchmark had no detector errors. Applicability over
all 916 rows was: spectral 916, entropy 916, cfa 12, learned 346, AEROBLADE
915, CLIP 916, and NPR 916.

## Fusion before and after

| Calibration | Held-out fused AUC | Held-out rows | Source groups |
|---|---:|---:|---:|
| Round 11 / before R12 | 0.5784615384615385 | 557 | 317 |
| Round 12 / after optional detectors | 0.5784615384615385 | 557 | 317 |

The fused result is unchanged at reported precision. AEROBLADE is zero
weighted by the guard; CLIP receives a small calibrated weight. No fused AUC
target or absolute floor was used.

## NPR-trained option

Skipped as the explicitly lowest-priority optional item. The paper-only NPR
weights remain unavailable and unlicensed; no equivalent CNN was trained in
this round. This is a scope decision, not fabricated coverage.

## Files changed

- requirements-learned.txt: optional torch, torchvision, Diffusers, open-clip-
  torch, and LPIPS pins.
- scripts/fetch_model.py: pinned TAESD, CLIP, and LPIPS-cache fetches with
  checksum checks.
- backend/app/analysis/aeroblade.py and backend/tests/test_aeroblade.py:
  TAESD + LPIPS implementation, lazy runtime, local-weight checks, and tests.
- backend/app/analysis/clip_probe.py, scripts/fit_clip_probe.py, and
  backend/tests/test_clip_probe.py: frozen backbone adapter,
  source/generator-held-out fitting, and tests.
- backend/app/analysis/registry.py, scripts/calibrate.py,
  scripts/benchmark.py, and backend/app/analysis/calibration.json: registry,
  raw metrics, AI-axis guard, benchmark metadata, and refit.
- docs/calibration.md, docs/detection-principles.md, and
  docs/learned-detector.md: optional-runtime, method, licensing, and confound
  limitations.
- plan/reference/detector-catalog.yaml, plan/reference/versions.lock.yaml,
  and plan/STATUS.yaml: catalog, pin, and round status updates.
- plan/plan.yaml: pre-existing user-owned D5b change preserved, not authored
  by this round's edits.

.github/workflows/ci.yml, Dockerfile, requirements.txt, and data/samples/
were not changed. models/ remains ignored.

## Verification

~~~text
/tmp/base/bin/python -m pytest backend/tests -q
85 passed, 1 warning in 322.02s

.venv/bin/python -m pytest backend/tests -q
85 passed, 1 warning in 331.44s

.venv/bin/python scripts/benchmark.py --out /tmp/post-r12.json --corpus all
wrote /tmp/post-r12.json and /tmp/post-r12.md

.venv/bin/python scripts/calibrate.py --corpus all --out backend/app/analysis/calibration.json --seed 20260828
weight/heldout-skill Spearman=0.5458137240480705

.venv/bin/python plan/validate.py
All structural and shell-syntax checks passed.
~~~

The optional environment also passed pip check; git diff --check passed.
