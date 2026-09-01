# Round 18A repair report — close the compose gap

Date: 2026-09-01  
HEAD at start: `5a7342a`  
Status: measurement and direction audit complete; no detector declaration or
corpus file was changed.

## Result

The exact R17A parity corpus still has a large content shortcut. The
metadata-only result is not the problem anymore, but the blurred 32×32 image
content check predicts the label at `0.8472 +/- 0.0276` held out and
`0.8024 +/- 0.0173` pooled. These are Hanley–McNeil standard errors.

CLIP's R17A result on the same rows was `0.8015 +/- 0.0174`. The near equality
is not a decomposition of CLIP's score, but it is decisive evidence against
calling that number forensic: a much simpler content representation reaches
the same separation. The current CLIP result is primarily a content/corpus
result, not the project's first trustworthy AI-generation measurement.

## 1. Exact R17A content check

The input was made from `scripts.benchmark._real()` by retaining every row
whose manifest `variant` is `parity`, giving 402 `ai_generated` rows and 212
`authentic` rows. The labels, paths, axes, and `source_image` values were
copied without changing the images. The check used seed `20260828`, the
script's 30% grouped split, and its standardised nearest-centroid classifier
over RGB 32×32 LANCZOS thumbnails followed by Gaussian blur radius 1.0.

Command:

```text
.venv/bin/python scripts/check_content_shortcut.py \
  --manifest /tmp/r18a-parity-content.jsonl \
  --out /tmp/r18a-content-parity.json
```

| configuration | n positive | n negative | held-out AUC +/- SE | pooled AUC +/- SE | gate |
|---|---:|---:|---:|---:|---|
| all exact R17A parity rows | 402 | 212 | 0.8472 +/- 0.0276 | 0.8024 +/- 0.0173 | FAIL |

The split contained 122 positive and 64 negative test rows. The per-axis
pooled/held-out results were:

| AI axis | n positive | pooled AUC +/- SE | held-out AUC +/- SE |
|---|---:|---:|---:|
| `real_ai` | 12 | 0.9057 +/- 0.0586 | 0.8516 +/- 0.1224 |
| `sd35_flux` | 120 | 0.9098 +/- 0.0190 | 0.8971 +/- 0.0369 |
| `synthbuster` | 270 | 0.7558 +/- 0.0216 | 0.7215 +/- 0.0417 |

This is a content shortcut, not a metadata shortcut: the format, dimensions,
file size, and EXIF gate had already passed on these parity rows. The result
also shows that the residual is not confined to one axis.

## 2. Content-matched diagnostic

Because the content gate failed, I ran one fixed, reproducible matching
experiment. It was not used to tune a detector score.

### Selection rule

* Candidate pool: the same 402 positive and 212 negative parity rows.
* Representation: the exact blurred 32×32 RGB vector used by
  `check_content_shortcut.py`.
* Standardisation: feature-wise standard deviation over the pooled candidate
  rows, with the script's `1e-6` floor.
* Matching: rectangular minimum-cost one-to-one assignment, using mean
  squared standardised pixel distance. The assignment selects 212 distinct
  positive rows and all 212 negative rows.
* Reproducibility: RNG seed `20260828` permuted both candidate orders before
  assignment to make tie handling deterministic; output rows were then sorted
  by image ID. No CLIP score was used for selection.

The selected set has 212 pairs / 424 rows. Its mean and median assignment
distances were 0.6593 and 0.6319 in the stated standardised mean-squared units.
Selected positive counts were: `midjourney-v5` 30, `stable-diffusion-2` 29,
`stable-diffusion-xl` 28, `firefly` 25, `dalle3` 20,
`stable-diffusion-1-4` 19, `stable-diffusion-1-3` 18,
`FLUX.1-schnell` 15, `stable-diffusion-3.5-medium` 12, `dalle2` 5,
`glide` 5, `xAI Aurora` 1, and 5 rows without a generator name. This is a
selection diagnostic, not a new balanced corpus axis.

The exact temporary row list and result were:

```text
/tmp/r18a-content-matched.jsonl
sha256 f8980d66d9d8789af2c73726c07e086ca250748b84296a6c5440d4305ea1a7a0
/tmp/r18a-content-matched-gate.json
sha256 09278ea02f31ae71c5c56cdd3d153e50cccd600270758041b4b9d797bb6ab57b
```

Command:

```text
.venv/bin/python scripts/check_content_shortcut.py \
  --manifest /tmp/r18a-content-matched.jsonl \
  --out /tmp/r18a-content-matched-gate.json
```

| configuration | n positive | n negative | held-out AUC +/- SE | pooled AUC +/- SE | gate |
|---|---:|---:|---:|---:|---|
| 212 content-matched pairs | 212 | 212 | 0.7021 +/- 0.0507 | 0.7060 +/- 0.0251 | FAIL |

Matching reduced the held-out shortcut from 0.8472 to 0.7021 and pooled
shortcut from 0.8024 to 0.7060, but it did not approach the repository's
`0.55` acceptance limit. The 212-pair sample is not too small; it is large
enough to show that this particular content control did not close the leak.
Shrinking it further to obtain a more flattering gate result would be tuning
the evaluation and was not done. This subset is therefore not an accepted
content-controlled corpus.

### CLIP on the diagnostic subset

Using the already recorded R17A CLIP scores for exactly the selected image
IDs (no refit and no threshold change):

| rows | n positive | n negative | CLIP AUC +/- SE |
|---|---:|---:|---:|
| exact R17A parity | 402 | 212 | 0.8015 +/- 0.0174 |
| 212 matched pairs | 212 | 212 | 0.7896 +/- 0.0220 |

The matched subset costs about 27% more Hanley–McNeil SE (`0.0174` to
`0.0220`) while retaining a substantial content shortcut. The small decrease
in CLIP AUC is not evidence of a forensic remainder because the acceptance
gate still fails.

For completeness, the selected-subset CLIP rows were:

| generator | selected n | AUC +/- SE |
|---|---:|---:|
| `FLUX.1-schnell` | 15 | 0.8465 +/- 0.0638 |
| `stable-diffusion-3.5-medium` | 12 | 0.7697 +/- 0.0813 |
| `dalle2` | 5 | 0.7085 +/- 0.1320 |
| `dalle3` | 20 | 0.9175 +/- 0.0430 |
| `firefly` | 25 | 0.7753 +/- 0.0566 |
| `glide` | 5 | 0.7755 +/- 0.1242 |
| `midjourney-v5` | 30 | 0.8107 +/- 0.0491 |
| `stable-diffusion-1-3` | 18 | 0.7233 +/- 0.0699 |
| `stable-diffusion-1-4` | 19 | 0.8260 +/- 0.0595 |
| `stable-diffusion-2` | 29 | 0.7573 +/- 0.0540 |
| `stable-diffusion-xl` | 28 | 0.7732 +/- 0.0538 |
| `xAI Aurora` | 1 | 0.6840 +/- 0.2973 |
| unnamed | 5 | 0.5943 +/- 0.1356 |

Each per-generator AUC uses the 212 selected authentic rows as negatives.
Small selected generator counts have correspondingly large standard errors.

## 3. Direction audit

No direction was changed. An AUC below 0.5 is reported as an anti-correlated
result, not inverted into an acceptance result.

### AEROBLADE

**Premise.** A latent-diffusion image is decoded by a VAE and should lie near
that autoencoder's reconstruction manifold. The same autoencoder should
therefore reconstruct it with *lower* perceptual error than an ordinary camera
image.

**Actual statistic.** `backend/app/analysis/aeroblade.py` computes

```text
LPIPS(input, TAESD.decode(TAESD.encode(input)))
```

after bounded resizing to a multiple-of-eight shape. The runtime uses
distilled TAESD rather than the paper's exact autoencoder and an LPIPS AlexNet
distance. The raw metric is `reconstruction_lpips`.

**Direction.** `higher_is_worse=False` in the detector and calibration is the
physically correct declaration for this premise: lower raw reconstruction
error maps to higher suspicion. The declaration was not changed.

**Evidence.** On the exact R17A parity rows, the final correctly directed
score gives AUC `0.3299 +/- 0.0236` (402 AI, 212 authentic; all 614
applicable). The raw LPIPS means were 0.09892 for AI and 0.07526 for
authentic. Thus the observed AI images have *higher*, not lower, error on this
implementation/corpus, producing the below-chance result under the correct
direction. The stable-diffusion-family examples are also below chance:
SD 1.3 `0.1219 +/- 0.0251`, SD 1.4 `0.1171 +/- 0.0244`, SD 2
`0.1898 +/- 0.0339`, SD 3.5 `0.2275 +/- 0.0302`, and FLUX
`0.3612 +/- 0.0381`.

**Conclusion.** The data contradict the effective AEROBLADE premise for
this TAESD/LPIPS implementation on this parity corpus. It does not establish
that the paper's method is wrong, nor does it identify whether distilled-model
mismatch, JPEG parity processing, corpus differences, or an implementation
error is responsible. There is no physics-based justification for flipping
the sign. The detector remains latent-diffusion-specific and unvalidated here.

### `learned`

**Premise.** This is an ONNX face-deepfake classifier, not a general
AI-generation detector. For an image that is actually a face deepfake, a
higher probability of the model's `Deepfake` class is more suspicious. The
Haar face check is an image-side applicability precondition only.

**Actual statistic.** After the face gate, the adapter resizes RGB to 224×224,
normalizes it, reads the first two ONNX outputs, and uses output 1 as the
`Deepfake` probability (or the two-class softmax probability when logits are
returned). The raw metric is `deepfake_probability`, then calibration maps it
with `higher_is_worse=True`.

**Direction.** `higher_is_worse=True` matches the model's intended class
semantics. `scripts/fetch_model.py` verifies the model metadata as
`{"0": "Realism", "1": "Deepfake"}`. The declaration was not changed.

**Evidence.** On R17A parity rows, only 261 rows were applicable after the
image-side face gate (136 AI and 125 authentic). The AUC was
`0.0856 +/- 0.0186`. Mean raw deepfake probability was 0.6226 for AI versus
0.7847 for authentic. This is systematic anti-correlation on this evaluation,
not noise, but it is not evidence for a reversed sign: these labels are
AI-generated images, not labels for face deepfakes. The result is a scope and
domain failure when the face model is treated as an AI-origin detector. It
does not show that higher `Deepfake` probability should mean less suspicious
for the task the model was trained to solve.

## Decision

The compose gap remains open. CLIP's `0.8015` and the content check's
`0.8024` cannot be used as independent evidence of AI-generation detection.
The 212-pair content match reduced, but did not remove, the shortcut and is
not accepted as a replacement evaluation. A future corpus or matching design
must pass the same content gate before any CLIP/AEROBLADE headline is treated
as forensic. AEROBLADE and `learned` keep their existing physically/model
semantically correct directions; their below-chance results are recorded as
method/domain failures rather than repaired by sign flipping.
