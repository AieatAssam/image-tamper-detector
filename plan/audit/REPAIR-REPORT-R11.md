# Round 11 repair report

Date: 2026-08-30

## Result

Round 11 completes the NPR direction audit, adds an image-side face
precondition to the optional learned detector, and investigates CFA absence on
PNG. The NPR sign remains unchanged because it agrees with the physical
premise. The learned detector is now zero-weighted on the AI axes. The CFA
gate remains strict because PNG absence cannot distinguish generation from a
camera JPEG re-encoded as PNG. No image bytes or model bytes were committed,
and no commit was created.

The AI benchmark below uses the fixed manifest sample selected with seed
`20260830`: 60 images per `sd35_flux` generator, 30 per `synthbuster`
generator, and 12 `real_camera` negatives. AUC is rank AUC. `+/-` is the
Hanley-McNeil standard error, not a confidence interval. The generated rows
are not image-level paired with the camera negatives, so these are explicitly
unpaired cross-source screens.

## Per-item status

| Item | Status | Result |
|---|---|---|
| P1 NPR direction and calibration | COMPLETE | The implementation computes a training-free 2x2 relative-difference composite. Its higher-is-worse direction agrees with the physics; the AI-axis result is negative and the weight remains zero. |
| P2 learned applicability and refit | COMPLETE | OpenCV's bundled Haar face cascade gates inference. No detected face returns `NOT_APPLICABLE`. The AI-axis guard changes the weight from `0.11950275731498387` to `0.0`. |
| P3 CFA absence investigation | INVESTIGATED / UNCHANGED | Relaxing the PNG gate would confuse generated PNG with camera JPEG re-encoded as PNG. The strict camera-JPEG and EXIF gate remains. |
| AEROBLADE carry-forward | BLOCKED | `models/taesd/encoder.onnx` and `models/taesd/decoder.onnx` are still absent. All AI-axis rows remain `NOT_APPLICABLE`. |

## P1: NPR physics and direction

The statistic actually computed by `backend/app/analysis/npr.py` is not the
paper's trained NPR classifier. For every overlapping 2x2 RGB patch it
subtracts the fourth pixel from the other three and computes:

- the mean intra-patch variance divided by the variance of the patch reference
  pixels, `intra_inter_variance_ratio`;
- the fraction of patches below the one-level variance threshold,
  `near_constant_fraction`; and
- Shannon entropy of the quantized relative differences, plus its normalized
  form.

The returned raw statistic is:

```text
(near_constant_fraction
 + 1 / (1 + intra_inter_variance_ratio)
 + 1 - normalized_difference_entropy) / 3
```

Tan et al.'s physical premise predicts that upsampling creates local
neighbouring-pixel dependence. Relative differences in generated images should
therefore be more structured than in camera images: lower intra-patch
variance, more near-constant patches, and lower difference entropy. The first
and third raw quantities move lower when suspicion increases. The
implementation explicitly inverts those two terms before averaging; the
near-constant fraction already moves higher. Thus the final composite's
`higher_is_worse=True` calibration declaration is physically consistent. The
sign was not flipped to chase AUC.

The data disagrees with that prediction on most current generators. The
Round 11 AI-axis AUC is `0.3416666666666667 +/- 0.08720542226749045`, with
`390` applicable generated positives and `12` applicable `real_camera`
negatives. Its within-source calibration fields are now populated as
`0.4793926247288503` overall and `0.4676923076923077` held out, but its guard
uses the explicit AI-axis screen and leaves the weight at `0.0`. This is a
negative result about this training-free statistic on the measured generators,
not a declaration that its physics or sign should be inverted.

### NPR per-generator before and after calibration

The R10 baseline used a different sampled positive count for the S1 benchmark;
R11 uses all 60 S1 rows and all 30 S2 rows in the manifest sample. The change
is therefore reported as a calibration/measurement change, not as a
paired-image improvement.

| Generator | R10 benchmark AUC +/- SE | R11 benchmark AUC +/- SE |
|---|---:|---:|
| `FLUX.1-schnell` | 0.534722 +/- 0.090242 | 0.534722 +/- 0.090242 |
| `stable-diffusion-3.5-medium` | 0.281944 +/- 0.088854 | 0.281944 +/- 0.088854 |
| `dalle2` | 0.122222 +/- 0.068520 | 0.122222 +/- 0.068520 |
| `dalle3` | 0.308333 +/- 0.095668 | 0.308333 +/- 0.095668 |
| `firefly` | 0.513889 +/- 0.099334 | 0.513889 +/- 0.099334 |
| `glide` | 0.688889 +/- 0.085809 | 0.688889 +/- 0.085809 |
| `midjourney-v5` | 0.522222 +/- 0.099029 | 0.522222 +/- 0.099029 |
| `stable-diffusion-1-3` | 0.141667 +/- 0.072960 | 0.141667 +/- 0.072960 |
| `stable-diffusion-1-4` | 0.138889 +/- 0.072357 | 0.138889 +/- 0.072357 |
| `stable-diffusion-2` | 0.105556 +/- 0.064254 | 0.105556 +/- 0.064254 |
| `stable-diffusion-xl` | 0.266667 +/- 0.092011 | 0.266667 +/- 0.092011 |

## P2: learned face applicability and refit

`LearnedDetector` now runs OpenCV's bundled
`haarcascade_frontalface_default.xml` on the input image's downscaled RGB
representation. If no face is detected, `applicable()` returns false and
`run()` returns `NOT_APPLICABLE` before loading ONNX Runtime or checking model
weights. This is an image-side precondition and does not inspect the corpus
axis, generator, or source group. No torch or new dependency was added.

The full calibration was refit over 916 rows and 317 source groups with seed
`20260828`. For this detector the guard now uses `ai_axis_auc`, not its old
source-paired score, because the face model's scope is the AI-generation axis.
The fitted values are:

| Measurement | Before R11 | After R11 |
|---|---:|---:|
| within-source AUC | not recorded in R10 calibration | 0.6235294117647059 |
| held-out AUC | not recorded in R10 calibration | 0.6363636363636364 |
| AI-axis AUC +/- SE | not recorded in R10 calibration | 0.42385321100917434 +/- 0.13664179659732437 |
| AI-axis applicable positives | not recorded in R10 calibration | 109 |
| applicable camera negatives | not recorded in R10 calibration | 5 |
| fusion weight | 0.11950275731498387 | 0.0 |

The prior R10 learned benchmark was below chance on every named generator.
The post-gate R11 benchmark is below, with the applicable populations shown so
the abstentions are not mistaken for negative scores.

| Generator | R10 before face gate AUC +/- SE | R11 after face gate AUC +/- SE |
|---|---:|---:|
| `FLUX.1-schnell` | 0.4314 +/- 0.0978 | 0.470588 +/- 0.151380 (17/5 applicable) |
| `stable-diffusion-3.5-medium` | 0.4024 +/- 0.0979 | 0.410000 +/- 0.149289 (20/5 applicable) |
| `dalle2` | 0.4333 +/- 0.1005 | 0.533333 +/- 0.165241 (9/5 applicable) |
| `dalle3` | 0.5028 +/- 0.0997 | 0.466667 +/- 0.159056 (12/5 applicable) |
| `firefly` | 0.3306 +/- 0.0972 | 0.433333 +/- 0.159139 (12/5 applicable) |
| `glide` | 0.3944 +/- 0.0999 | N/A (0/5 applicable; fewer than 10 positives) |
| `midjourney-v5` | 0.3167 +/- 0.0963 | 0.383333 +/- 0.157600 (12/5 applicable) |
| `stable-diffusion-1-3` | 0.2778 +/- 0.0931 | N/A (2/5 applicable; fewer than 10 positives) |
| `stable-diffusion-1-4` | 0.3611 +/- 0.0988 | N/A (2/5 applicable; fewer than 10 positives) |
| `stable-diffusion-2` | 0.2944 +/- 0.0946 | 0.257143 +/- 0.141712 (14/5 applicable) |
| `stable-diffusion-xl` | 0.3833 +/- 0.0996 | 0.555556 +/- 0.163804 (9/5 applicable) |

The changed applicability makes the post-gate comparison noisier, not better:
the R11 negative pool has only five detectable-face camera images. The
detector is still below chance overall, so zero weight is the required result.

## P3: CFA absence investigation

The CFA implementation estimates a dominant Bayer arrangement with
intermediate-value masks and reports local disagreement with that arrangement.
It does not compute a generic “absence of CFA” score. A missing dominant
pattern returns zero from `measure()`, while the runtime gate correctly
requires a strict JPEG with camera Make/Model EXIF and matching capture
dimensions.

The investigation tested representative S1 and S2 PNGs and a genuine camera
JPEG. The generated PNGs returned no dominant pattern (`_pattern_measure()`
was `None`, so `measure()` returned `(0.0, -1, ...)`). A camera JPEG encoded
again as PNG in memory returned the same result. Therefore a relaxed PNG path
would assign the same absence signal to two different causes. That is not a
sound AI-generation claim, so the gate is unchanged. The new regression test
ensures a large PNG remains `NOT_APPLICABLE`.

The complete post-change AI table is:

| Generator | spectral | entropy | cfa | learned | aeroblade | npr |
|---|---:|---:|---:|---:|---:|---:|
| `FLUX.1-schnell` | 0.558333 +/- 0.088786 | 0.598611 +/- 0.085719 | N/A | 0.470588 +/- 0.151380 | N/A | 0.534722 +/- 0.090242 |
| `stable-diffusion-3.5-medium` | 0.545833 +/- 0.089588 | 0.505556 +/- 0.091688 | N/A | 0.410000 +/- 0.149289 | N/A | 0.281944 +/- 0.088854 |
| `dalle2` | 0.558333 +/- 0.097316 | 0.661111 +/- 0.088967 | N/A | 0.533333 +/- 0.165241 | N/A | 0.122222 +/- 0.068520 |
| `dalle3` | 0.572222 +/- 0.096489 | 0.294444 +/- 0.094573 | N/A | 0.466667 +/- 0.159056 | N/A | 0.308333 +/- 0.095668 |
| `firefly` | 0.605556 +/- 0.094122 | 0.536111 +/- 0.098444 | N/A | 0.433333 +/- 0.159139 | N/A | 0.513889 +/- 0.099334 |
| `glide` | 0.963889 +/- 0.026766 | 0.811111 +/- 0.066955 | N/A | N/A | N/A | 0.688889 +/- 0.085809 |
| `midjourney-v5` | 0.608333 +/- 0.093900 | 0.369444 +/- 0.099118 | N/A | 0.383333 +/- 0.157600 | N/A | 0.522222 +/- 0.099029 |
| `stable-diffusion-1-3` | 0.569444 +/- 0.096662 | 0.680556 +/- 0.086798 | N/A | N/A | N/A | 0.141667 +/- 0.072960 |
| `stable-diffusion-1-4` | 0.536111 +/- 0.098445 | 0.586111 +/- 0.095568 | N/A | N/A | N/A | 0.138889 +/- 0.072357 |
| `stable-diffusion-2` | 0.502778 +/- 0.099689 | 0.552778 +/- 0.097621 | N/A | 0.257143 +/- 0.141712 | N/A | 0.105556 +/- 0.064254 |
| `stable-diffusion-xl` | 0.700000 +/- 0.084435 | 0.508333 +/- 0.099519 | N/A | 0.555556 +/- 0.163804 | N/A | 0.266667 +/- 0.092011 |

`cfa` is N/A for all 11 generated groups because all downloaded AI images are
PNG. It was applicable on 12 camera negatives but has no positive class.
`aeroblade` is N/A for all rows because the TAESD ONNX pair is absent. The
spectral and entropy values are measured on all 60/30 generated rows and 12
camera negatives. Learned uses five applicable camera negatives after the
face gate. NPR uses all 12 camera negatives.

## Fusion before and after

The prior committed R10 calibration reported held-out fused AUC
`0.6521739130434783` on 526 rows and 227 source groups. The R11 refit reports
held-out fused AUC `0.5784615384615385` on 916 rows and 317 source groups.
The decrease is expected after removing the learned detector from a scope
where it is below chance. No AUC floor, sign inversion, or fusion tuning was
introduced to hide it.

## Corpus and licensing note

The calibration manifest contains 916 rows: 400 IMD2020, 120 `sd35_flux`, 270
`synthbuster`, 12 strict `real_camera`, 12 `real_ai`, and two C2PA fixtures.
Generator names are copied from dataset structure. S1 is Zenodo DOI
`10.5281/zenodo.22166280`, CC-BY-4.0. S2 is Zenodo DOI
`10.5281/zenodo.10066460`, CC-BY-NC-SA-4.0. The S2 NC and share-alike terms
apply to those rows; the entire corpus must not be described as CC-BY.
Archives, extracted images, and model bytes remain ignored and untracked.

## Files changed

- `backend/app/analysis/learned.py`: Haar face precondition and image-side
  `NOT_APPLICABLE` path.
- `backend/app/analysis/calibration.json` and `scripts/calibrate.py`: explicit
  AI-axis screen for `learned` and `npr`, refit thresholds, weights, and
  held-out results.
- `backend/tests/test_learned.py` and `backend/tests/test_cfa.py`: face-gate
  and PNG-abstention checks.
- `docs/calibration.md`, `docs/detection-principles.md`,
  `docs/learned-detector.md`, `README.md`: scope, physics, licensing, and
  calibration updates.
- `plan/reference/detector-catalog.yaml`: NPR physics direction, learned gate,
  CFA limitation, and R11 measurements.
- `plan/STATUS.yaml`: R11 status.

## Verification

The post-change benchmark completed without detector errors:

```text
.venv/bin/python scripts/benchmark.py --out /tmp/r11-ai-all.json \
  --corpus real --axes sd35_flux,synthbuster,real_camera --sample 402 \
  --seed 20260830 --detectors spectral,entropy,cfa,learned,aeroblade,npr --profile
402 rows; learned applicable 114, not_applicable 288; cfa applicable 12;
aeroblade applicable 0; npr applicable 402
```

Verification passed:

```text
.venv/bin/python -m pytest backend/tests/test_cfa.py backend/tests/test_learned.py backend/tests/test_npr.py -q
10 passed in 1.16s

.venv/bin/python -m pytest -q
81 passed, 1 warning in 312.09s

.venv/bin/python scripts/fetch_corpus.py --check
816 manifest entries verified

json/yaml parse passed
git diff --check
```

The full test run creates the ignored `:memory:.ses` test artifact; it was
removed after verification. The vault checks also passed after adding the two
new discoveries:

```text
Scripts/validate.py
notes: 165
problems: 0
Scripts/verify-quotes.py
quotes checked: 463 | too short to test: 12 | not found: 0
```
