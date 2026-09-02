# Round 19 repair report — aigen

Date: 2026-09-01  
Scope: `spectral`, `npr`, `aeroblade`, and `entropy`. No calibration, corpus,
model, or unrelated detector files were changed by this agent.

## Result summary

| detector | R17 audit grade | R19 grade against the cited source | resolution | result |
|---|---|---|---|---|
| `spectral` | MAJOR-DRIFT | MAJOR-DRIFT, now explicitly an honest repository variant | (b), plus bounded mask removal | no classifier or paper training protocol was fabricated |
| `npr` | MAJOR-DRIFT | MAJOR-DRIFT, now explicitly a proxy | (b), plus paper-shaped representation repair | no trained classifier or weights were fabricated |
| `aeroblade` | MAJOR-DRIFT | MAJOR-DRIFT, now explicitly a single-TAESD variant | (b) | lower-error direction retained; no model assets were fabricated |
| `entropy` | UNVERIFIED | UNVERIFIED, now explicitly blog-derived | (b), plus blog-constant and arithmetic repair | no paper claim is made |

The grades remain method-fidelity grades. Correcting a catalog claim does not
turn a method without its published classifier or model family into a faithful
reimplementation.

## Sources fetched

- [Zhang, Karaman, and Chang, “Detecting and Simulating Artifacts in GAN Fake Images”](https://arxiv.org/pdf/1907.06515)
- [Durall, Keuper, and Keuper, “Watch your Up-Convolution”](https://arxiv.org/pdf/2003.01826)
- [Tan et al., “Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection”](https://arxiv.org/pdf/2312.10461)
- [Ricker et al., “AEROBLADE”](https://arxiv.org/html/2401.17879)
- [Rohrer, “Detecting AI-Generated Images Using Entropy Analysis”](https://blog.frohrer.com/detecting-ai-generated-images-using-entropy-analysis/)

The first four sources were obtained from arXiv. The entropy source is a blog
post, not a paper; it remains UNVERIFIED under R1.

## `spectral`

### Grade and resolution

Audit grade: **MAJOR-DRIFT**. Post-change grade against Zhang/Durall:
**MAJOR-DRIFT**, with the catalog corrected to say “repository spectral-peak
variant ... not a paper reimplementation” (`plan/reference/detector-catalog.yaml:372-391`).

Resolution (b) was chosen for the P0. Zhang's method requires a spectrum-input
classifier; Durall's method requires a classifier after its radial feature
extraction. This repository has no paper-trained classifier, training corpus,
or model weights for either source, and adding one would violate the round's
no-model/no-data constraint. The catalog now names the missing components
explicitly (`plan/reference/detector-catalog.yaml:391`).

### Paper specification

Zhang et al. say they “apply the 2D DFT to each of the RGB channels,” discard
phase, compute the logarithmic spectrum, normalize it to `[-1, 1]`, and use it
as input to a fake-image classifier. Their implementation details specify a
256x256 input, random 224x224 training crop, central 224x224 test crop, and an
ImageNet-pretrained ResNet-34.

Durall et al. define a 1-D representation by azimuthal integration of the DFT
power spectrum. Their detection pipeline says inputs are “converted to
grey-scale before DFT,” then uses a basic SVM for supervised detection (and
K-Means for their unsupervised experiment), normalizing by the zero-frequency
coefficient and scaling the 1-D feature to a fixed size.

### Implementation and deltas

The code still computes the existing repository heuristic at
`backend/app/analysis/spectral.py:61-86`: fixed 512x512 resize, Gaussian
high-pass with `sigmaX=1.0`, Hann window, FFT magnitude, `log1p`, radial-average
subtraction, radius-5 exclusion, maximum standardized residual, and local
maxima at `4.0` sigma. These values are repository values; neither cited paper
specifies this peak statistic, `sigma=1.0`, radius 5, 4-sigma threshold, or
512x512 resize.

The R17 P1 JPEG-grid mask was removed. `measure` now uses only
`radius > 5` for valid spectrum pixels (`backend/app/analysis/spectral.py:70-74`);
the `_jpeg_grid_mask` helper is gone. The papers do not prescribe masking a
JPEG lattice. The catalog consequently says JPEG processing remains a
processing sensitivity of this variant rather than a published correction
(`plan/reference/detector-catalog.yaml:401-403`).

The shared 1600-side cap followed by the detector's 512 resize remains. It is
now explicitly recorded as an operational gap rather than a paper requirement
(`plan/reference/detector-catalog.yaml:389-391`). No threshold was tuned.

### Signal direction

The physical premise is that upsampling can replicate or distort frequency
content. The repository statistic is the maximum non-radial spectral residual
in standard-deviation units, so a larger value means a stronger periodic or
anisotropic cue. `higher_is_worse=true` remains physically consistent. The
papers' final classifier directions are learned from labels; this direction
check applies only to the repository peak statistic.

### Measurement

Reproduction image: `data/samples/tampered/landscape_copy_paste.jpg`.

Before, direct measurement:

```text
spectral (2.8077352046966553, 0)
```

After, direct measurement:

```text
spectral (2.8077783584594727, 0)
```

The raw value moved by `+0.0000431537628174`; the peak count stayed zero.
Removing the unsupported mask did not materially change this image.

### Prioritised follow-up

1. **P0 remains open:** implement a complete Zhang or Durall classifier only
   when the project has an approved paper-matched training protocol and model
   assets, or keep this cataloged as the present heuristic variant.
2. **P1 completed, resolution (a):** the unsupported JPEG-grid mask was
   removed because the cited papers do not prescribe it.
3. **P1 resolution (b), documented but not calibrated:** the 1600-to-512 serving path and
   heuristic thresholds remain operational choices. A paper-matched experiment
   would need a separately fitted calibration; this round did not touch
   calibration files.

### Calibration impact

The existing raw keys `peak_to_sigma` and `peak_count` were retained. The
mask removal changes the statistic in principle, so the existing spectral
calibration should be refit by the human after all round repairs. No
calibration file was changed here.

## `npr`

### Grade and resolution

Audit grade: **MAJOR-DRIFT**. Post-change grade against Tan et al.:
**MAJOR-DRIFT**, with the catalog corrected to call this a “training-free
NPR-inspired proxy, not the paper's trained detector”
(`plan/reference/detector-catalog.yaml:584-600`).

Resolution (b) was chosen for the P0 classifier gap. Tan et al. require a
1.44-million-parameter CNN/ResNet classifier trained on their data. This
repository has no such weights or training pipeline, so the code does not
pretend to reproduce the paper's final detector. The bounded representation
portion was repaired to the paper's stated grid/reference construction
(resolution (a) for that sub-step).

### Paper specification

Tan et al. divide the output into `W x H` grids of `l x l` patches, construct
the relationship representation per color channel, and form every `w_i-w_j`,
including the zero reference element. They state `l=2` and `j=1`, while also
noting that the reference element may be any element. They then use the NPR
representation to train a lightweight CNN with a ResNet block; the paper
specifies 1.44M parameters, Adam learning rate `2 x 10^-4`, and batch size 32.

### Implementation and deltas

`backend/app/analysis/npr.py:110-115` now constructs aligned,
non-overlapping 2x2 grids, uses the first element as the reference, preserves
the RGB channels, and includes the zero relationship. `measure` consumes all
four relationships per channel (`backend/app/analysis/npr.py:79-87`). This
removes the prior one-pixel-stride overlap, bottom-right reference, RGB sum
collapse, synthetic histogram zeros, and the inconsistent `/12` formula.

The remaining scalar is deliberately repository-defined:
`np.var` over the 12 relationship values, an intra/inter variance ratio,
near-constant fraction, 511-bin quantized relationship entropy, equal-weight
combination, and calibration are not paper constants
(`backend/app/analysis/npr.py:80-106`). The 1024 longest-side cap and 4x4
applicability floor are also operational choices and are now called out in the
catalog (`plan/reference/detector-catalog.yaml:594-600`). The paper's trained
classifier, ForenSynths training data, and weights remain absent.

### Signal direction

The paper's physical premise is local dependence after upsampling: lower local
relationship variance, more near-constant 2x2 relationships, and lower
relationship entropy are the suspicious-side cues. The proxy inverts the
variance ratio and entropy terms before averaging them, so higher
`npr_statistic` remains the suspicious direction. `higher_is_worse=true` was
not changed. Tan's learned classifier direction is not established by this
hand-built scalar.

### Measurement

Reproduction image: `data/samples/tampered/landscape_copy_paste.jpg`.

Before, direct measurement:

```text
npr 0.7653611162477771
```

After, direct measurement:

```text
npr 0.7661189691552025
```

The raw value moved by `+0.0007578529074254`. This is a small movement in the
direction of the existing score mapping, not an AUC claim.

### Prioritised follow-up

1. **P0 classifier gap resolved by claim correction:** keep the detector out of
   paper-accuracy comparisons until an approved training protocol and weights
   exist.
2. **P0 deterministic representation/math repaired, resolution (a):** retain the aligned
   four-value per-channel relationships and refit the proxy calibration.
3. **P1 resolution (b), documented but not calibrated:** the 1024 cap and proxy thresholds have
   no paper basis. The human must refit or explicitly retain the exploratory
   calibration after all repairs.

### Calibration impact

The existing keys, including `npr_statistic`, were retained, but their
distribution changed because the patch layout, reference, and relationship
population changed. The current NPR calibration is not valid evidence for the
new representation and must be refit by the human. No calibration file was
changed.

## `aeroblade`

### Grade and resolution

Audit grade: **MAJOR-DRIFT**. Post-change grade against Ricker et al.:
**MAJOR-DRIFT**, with the catalog corrected to call the implementation a
“repository AEROBLADE-style single-TAESD/LPIPS variant, not the paper's
complete method” (`plan/reference/detector-catalog.yaml:534-553`).

Resolution (b) was chosen. The published method needs the external SD1, SD2,
and KD2.1 autoencoder family and the paper's LPIPS configuration. This
repository has only one optional distilled TAESD artifact and an AlexNet LPIPS
cache; adding the missing model family or weights is prohibited in this round.
No source change was justified or made in `aeroblade.py`.

### Paper specification

Ricker et al. define `Delta_AE_i(x) = d(x, D_i(E_i(x)))` and then define
`Delta_Min` as the minimum reconstruction error across a set of autoencoders.
They write that generated images have consistently lower error, and that
`LPIPS_2` captures the most meaningful differences. Their main experiments use
SD1, SD2, and KD2.1 AEs; their real-image protocol uses a 512x512 center crop.

### Implementation and deltas

The source still loads one `AutoencoderTiny` and one standard
`lpips.LPIPS(net="alex")` (`backend/app/analysis/aeroblade.py:104-119`), then
computes one encode/decode LPIPS distance (`backend/app/analysis/aeroblade.py:66-77`).
The paper's multi-AE minimum is therefore absent. Standard AlexNet LPIPS is
also not the paper's strongest reported `LPIPS_2`/VGG16 setup.

The serving input remains an aspect-preserving multiple-of-eight resize after
the shared 1600-side cap (`backend/app/analysis/aeroblade.py:135-146`), rather
than the paper's 512x512 center crop. The catalog now records both the missing
`Delta_Min`/LPIPS family and this preprocessing gap
(`plan/reference/detector-catalog.yaml:544-549`). The class defaults and
calibrated threshold are repository values, not paper constants.

### Signal direction

The physical/model premise is unambiguous: a generated image from the relevant
latent-diffusion family should reconstruct with lower error. The implementation
reports `LPIPS(input, TAESD.decode(TAESD.encode(input)))`; lower raw error is
more suspicious, so `higher_is_worse=False` at
`backend/app/analysis/aeroblade.py:39-41` remains correct. R18A recorded the
below-chance result under that direction; the missing AE minimum is a plausible
cause, not permission to invert the sign.

### Measurement

No raw AEROBLADE distance was available on the sample: the optional TAESD and
LPIPS weights are not installed, so the detector correctly returns
`NOT_APPLICABLE` rather than a fabricated score. The existing test covers this
path (`backend/tests/test_aeroblade.py:13-18`). R18A's recorded parity result
remains the relevant prior measurement: AUC `0.3299 +/- 0.0236`, with AI mean
raw LPIPS `0.09892` versus authentic `0.07526`, under the correct lower-error
direction.

### Prioritised follow-up

1. **P0 resolved by claim correction:** do not compare the single-TAESD result
   with the paper's `Delta_Min` headline evidence.
2. **P1 resolution (b), documented but open:** a paper-matched run needs square center-crop
   evaluation and the cited AE/LPIPS family. Serving can retain its current
   operational preprocessing only while labeled as a separate variant.
3. **P1 resolution (b), calibration deferred intentionally:** refit only after the raw metric
   representation is changed; the human's post-round calibration is required.

### Calibration impact

No AEROBLADE raw metric, key, direction, or calibration value changed. The
catalog claim changed, so existing measurements must continue to be described
as the single-TAESD variant rather than as paper AEROBLADE results.

## `entropy`

### Grade and resolution

Audit grade: **UNVERIFIED**. Post-change grade: **UNVERIFIED**, with the catalog
now explicitly describing a blog-derived scalarization rather than a
peer-reviewed method (`plan/reference/detector-catalog.yaml:108-133`).

Resolution (b) was chosen for the P0 because the cited source is a blog post,
not a paper. No paper-level claim, threshold, confidence mapping, or
classifier is made. The blog's bounded deterministic procedure was still
implemented where its specification was explicit.

### Source specification and implementation

Rohrer's source loads RGB channels, uses `filters.rank.entropy` over a disk of
`radius=5`, compares raw channel entropy values with `tolerance=0.1`, and
highlights the matching mask. It describes real images as having cohesive red
regions and AI images as having small scattered red regions; it does not
specify a scalar threshold or confidence mapping.

The implementation now uses the source's radius and tolerance defaults at
`backend/app/analysis/entropy.py:45-69`. It retains normalized uint8 maps only
for the returned visualization features, but computes the matching mask from
the raw entropy maps in float32 at `backend/app/analysis/entropy.py:141-168`.
This fixes the prior uint8 subtraction wrap, where `0-255` became `1`.

The unpublished uniformity mask, color-consistency mask, and morphology were
removed. `detect_ai_generated` now scalarizes only the matching mask area at
`backend/app/analysis/entropy.py:195-203`. This area is a repository API
scalarization of a qualitative source, not a source-published classifier.

The 1024 ndarray cap and shared 1600 adapter cap remain operational choices;
path/bytes analysis remains uncapped by this module. The catalog records those
facts (`plan/reference/detector-catalog.yaml:120-122`). The normal adapter
still passes calibrated threshold `0.6979472477`, while direct analyzer
construction still reads legacy default `0.35`; the catalog now states both and
states that neither is from the source (`plan/reference/detector-catalog.yaml:118`).
Calibration-file changes were out of scope.

### Signal direction

The source's qualitative premise is fewer cohesive matching entropy regions in
AI-generated images. Lower matching-mask area therefore maps toward greater
suspicion in the repository, and `higher_is_worse=False` remains consistent.
The source does not validate the repository's scalar threshold or sigmoid
confidence mapping.

### Measurement

Reproduction image: `data/samples/tampered/landscape_copy_paste.jpg`.

Before, direct analyzer with `matching_threshold=0.35`:

```text
entropy (False, 0.5508637152777778)
```

After, same direct command:

```text
entropy (False, 0.5922202932098766)
```

The matching proportion moved by `+0.0413565779320988`; the direct boolean
remained `False`. This movement is a changed raw metric, not evidence that the
blog heuristic is valid.

### Prioritised follow-up

1. **P0 resolved by claim correction:** keep the detector UNVERIFIED and blog-
   derived; do not describe its scalar as paper-level evidence.
2. **P0 completed, resolution (a):** raw entropy comparison and the unsigned arithmetic defect
   were repaired; the cited radius/tolerance are now the defaults.
3. **P1 completed, resolution (a):** extra masks and morphology were removed from the decision.
   Connectedness remains absent from the scalar API and is still an open method
   limitation because the source describes it qualitatively rather than
   specifying a reproducible connected-component score.
4. **P1 resolution (b), documented:** catalog/runtime threshold paths are explicit. The direct
   legacy default and adapter calibration remain distinct until the human's
   planned calibration/configuration pass decides whether to unify them.

### Calibration impact

The existing adapter metric key `matching_proportion` was retained, but its
distribution changed from normalized/multi-mask area to raw-entropy matching
area. The current `0.6979472477` adapter threshold and sigmoid scale are not
validated for this changed statistic and require human refitting. No
calibration file was changed.

## Verification

Commands and real outputs:

1. Baseline before repairs:

   ```text
   .venv/bin/python -m pytest backend/tests -q
   .................................................... [then completed]
   104 passed, 1 warning in 94.59s (0:01:34)
   ```

   The requested baseline said 102; this shared worktree already included two
   concurrent-round tests, so the observed baseline was 104.

2. Focused repaired detector tests:

   ```text
   .venv/bin/python -m pytest backend/tests/test_spectral.py backend/tests/test_npr.py backend/tests/test_entropy.py backend/tests/test_aeroblade.py -q
   22 passed in 81.31s (0:01:21)
   ```

3. Required full suite after repairs:

   ```text
   .venv/bin/python -m pytest backend/tests -q
   131 passed, 1 warning in 128.53s (0:02:08)
   ```

   The total includes tests added by concurrent rounds in this shared
   worktree; all tests were green.

   A final repeat after the last source cleanup also passed:

   ```text
   .venv/bin/python -m pytest backend/tests -q
   132 passed, 1 warning in 102.59s (0:01:42)
   ```

4. Catalog parse after the final catalog patch:

   ```text
   .venv/bin/python -c "import yaml,sys; yaml.safe_load(open('plan/reference/detector-catalog.yaml')); print('yaml ok')"
   yaml ok
   ```

5. `git diff --check` over the owned detector, test, and catalog paths produced
   no output and exit status 0.

6. The concise API reproduction was successful before the repair, returning
   HTTP 200 with verdict `inconclusive` and score
   `0.37818789798181013`; its normal adapter values included spectral raw
   `2.8077352046966553`, entropy matching proportion
   `0.19838460286458334`, and entropy score `0.9999471186800619`.

   The same endpoint reproduction after the repair returned HTTP 504 after the
   service's 60-second analysis timeout in the concurrently changing shared
   worktree. The wrapper then raised `KeyError` when it tried to read a verdict
   from the 504 body. No post-repair endpoint detector result is claimed; the
   direct same-image measurements above are the comparable before/after values.

## Still open

- Complete paper classifiers for Zhang/Durall and Tan NPR remain unavailable
  without an approved training protocol/data/model asset set.
- Complete AEROBLADE `Delta_Min` over the cited AE family and paper-matched
  LPIPS variant remain unavailable without the external model family.
- Entropy has no paper citation; its connected-region qualitative cue is not
  represented by the repository scalar.
- Human post-round calibration is required for the changed spectral/NPR/entropy
  raw distributions. No sign was inverted, and no calibration file was edited.
