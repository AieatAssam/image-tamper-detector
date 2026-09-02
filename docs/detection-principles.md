# Detection principles

This is the scientific reference for the detector set. The machine-readable
registry is [`plan/reference/detector-catalog.yaml`](../plan/reference/detector-catalog.yaml).
It is the source of truth for detector names, signal directions, applicability,
limitations, citations, and measured performance. Each current measurement is
identified by AUC, Hanley–McNeil SE, corpus, variant, applicable count, and
date; the catalog is authoritative when this prose is abbreviated.

## How to read the measurements

For source-paired measurements, the benchmark compares manipulated rows only
with authentic rows sharing the same `source_image`; this prevents the image
source from becoming a hidden class label. AI-axis AUCs instead compare
applicable generated rows with the available camera-negative pool and are
explicitly unpaired. In either case, AUC is the probability that a randomly
selected positive row scores above the comparison negative, with ties counted
as one half.
The reported uncertainty is the Hanley-McNeil estimate:

```text
SE(A) = sqrt([A(1-A) + (n+ - 1)(Q1 - A^2) + (n- - 1)(Q2 - A^2)] / (n+ n-))
Q1 = A / (2 - A)
Q2 = 2A^2 / (1 + A)
```

The catalog stores the AUC, standard error, participating count, corpus,
variant, scope, and date for every runtime detector ID. A null means that the
corpus did not contain a valid comparison. It is not a zero score. The current
measurements below are the latest committed observations; they are not the
unrefitted `calibration.json` artifact.

## Corpus shortcut acceptance gate

The first question for an AI-generation corpus is whether metadata alone can
predict its label. [`scripts/check_format_shortcut.py`](../scripts/check_format_shortcut.py)
fits a single threshold stump over decoded container format, width, height, file
size, and EXIF presence. It reports pooled and grouped held-out AUC, standard
error, and per-axis results. Run it on a few hundred real sample rows before
downloading or ingesting a proposed axis:

```text
.venv/bin/python scripts/check_format_shortcut.py --manifest sample.jsonl --check
```

The acceptance diagnostic uses `0.55` as its maximum held-out shortcut AUC.
This is not an absolute performance floor for a detector. A future axis must
pass its own `per_axis` result, and a failed result blocks the axis regardless
of its size, labels, or detector scores.

The current AI screen fails this check at held-out AUC `0.8750 +/- 0.0598`
(pooled `0.9583 +/- 0.0137`), selected on width. Round 10's per-generator
AI table therefore remains a metadata-confounded exploratory result. Round
12's CLIP `1.0000 +/- 0.0000` seen and unseen generator results are also not
generation evidence, because the probe can separate the corpus's metadata
domains. Round 14's 400-row WildFake sample fails independently at
`1.0000 +/- 0.0000` held-out and pooled metadata AUC, selected on decoded
format: generated DDIM images are PNG and CelebA-HQ negatives are JPEG.

Until an AI axis passes this gate, its per-generator and fused AI-generation
numbers must be labelled exploratory and must not be used to claim general
generation-detection skill.

## Native and parity are separate evidence axes

`native` preserves the supplied bytes and therefore retains capture metadata,
EXIF, quantisation tables, and JPEG history. `parity` is the deterministic
R15C byte-budget re-save: 1024×1024 RGB, optimized non-progressive 4:2:0 JPEG,
EXIF removed, and exactly 120,000 bytes. Its metadata gate is exactly chance
for format, dimensions, file size, and EXIF, but it changes JPEG quality and
history. The parity encoder therefore removes the size shortcut while adding a
quality distribution that remains class-correlated.

The R16C machine-readable policy is the measurement contract:

| variant scope | detectors |
|---|---|
| parity only | `aeroblade`, `clip_probe`, `learned`, `npr`, `spectral`, `entropy` |
| native only | `c2pa`, `qtable`, `exif`, `cfa`, `ela` |
| both | `copy_move`, `double_jpeg`, `jpeg_ghosts`, `prnu`, `resampling`, `splicebuster`, `zero` |

Calibration applies this scope while fitting and gating. Benchmarking applies
the same scope while serving a selected variant. The upload path is still
variant-blind, so a serving orchestrator must choose bytes matching the fitted
variant; calibration-time scope alone does not make a parity-only model safe on
native input. This is the explicit R16C train/serve limitation.

## Current evidence snapshot (R17C)

The table gives one primary, latest completed measurement per detector. AUCs
are not pooled across native and parity. `AI-axis` is an unpaired screen using
402 AI rows and 12 strict camera negatives; it is not a within-source claim.
The comparison values are retained only where both variants were measured in
R16A/B; the catalog records them under `comparison`.

| detector | AUC ± SE | corpus / variant / date | scope |
|---|---:|---|---|
| `aeroblade` | 0.416 ± 0.088 | R15C byte budget / parity / 2026-09-01 | AI-axis |
| `c2pa` | N/A | local manifest / native / 2026-08-31 | no valid paired AUC |
| `cfa` | N/A | local manifest / native / 2026-08-31 | no valid AI comparison |
| `clip_probe` | 0.999585 ± 0.000757 | R15C byte budget / parity / 2026-09-01 | AI-axis; 12 negatives |
| `copy_move` | 0.386 ± 0.127 | R15C / native / 2026-08-31 | AI-axis diagnostic |
| `double_jpeg` | 0.192 ± 0.082 | R15C / native / 2026-08-31 | AI-axis diagnostic |
| `ela` | 0.4190 ± 0.0985 | R16B bounded / native / 2026-09-01 | AI-axis diagnostic |
| `entropy` | 0.7616 ± 0.0559 | R16B bounded / parity / 2026-09-01 | AI-axis |
| `exif` | 0.083 ± 0.058 | R15C / native / 2026-08-31 | AI-axis diagnostic |
| `jpeg_ghosts` | 0.4667 ± 0.0984 | R16B bounded / native / 2026-09-01 | AI-axis diagnostic |
| `learned` | 0.184 ± 0.131 | R15C byte budget / parity / 2026-09-01 | face-applicable AI-axis |
| `npr` | 0.2803 ± 0.0846 | R16B bounded / parity / 2026-09-01 | AI-axis |
| `prnu` | 0.6282 ± 0.0743 | R16B bounded / native / 2026-09-01 | AI-axis; blind residual |
| `qtable` | N/A | local manifest / native / 2026-08-31 | no valid paired AUC |
| `resampling` | 0.298 ± 0.094 | R15C / native / 2026-08-31 | exploratory |
| `spectral` | 0.508 ± 0.084 | R15C byte budget / parity / 2026-08-31 | AI-axis |
| `splicebuster` | 0.720 ± 0.110 | R15C / native / 2026-08-31 | AI-axis diagnostic |
| `zero` | 0.275 ± 0.084 | R15C / native / 2026-08-31 | AI-axis diagnostic |

The numbers are traceable to [R15C](../plan/audit/REPAIR-REPORT-R15C.md),
[R16A](../plan/audit/REPAIR-REPORT-R16A.md), and
[R16B](../plan/audit/REPAIR-REPORT-R16B.md). Agents 17A and 17B are expected
to supersede some rows during this round; until those reports are committed,
no replacement value is claimed here.

The complete calibration corpus currently contains 916 rows in 317 source
groups: 100 synthetic processing-history rows and 816 rows from the local
manifest, including 400 source-directory-stratified IMD2020 rows, 12 strict
real-camera rows, 12 real-AI rows, two C2PA fixtures, 120 `sd35_flux` rows,
and 270 `synthbuster` rows. The synthetic rows are useful for compression,
geometry, and controlled processing experiments; they cannot establish sensor
provenance. The synthetic and real portions must therefore be read as
different validation populations, not pooled evidence of one universal
detector skill.

The generator-specific AI axes do not ship
the genuine camera counterpart bytes, so `ai_axis_auc` compares applicable
generated rows with applicable `real_camera` rows as an explicitly unpaired
cross-source screen. It is not a within-source paired claim. The generator
field is copied from the archive directory and is never inferred.

### Corpus roles

The benchmark has separate source-paired manipulation measurements and an
unpaired AI-axis screen. The latter uses a small, heterogeneous camera-negative
class and is not an open-web error rate. Do not quote a detector number without
the corpus and variant in the current-evidence table or catalog.

Every applicable raw statistic is mapped to a probability using

```text
score = sigmoid((s - t) / w)                  higher statistic is worse
score = sigmoid((t - s) / w)                  lower statistic is worse
sigmoid(x) = 1 / (1 + exp(-x))
```

`NOT_APPLICABLE` is an abstention: its score is null and it contributes
nothing to fusion. A detector result is evidence, not a proof of origin or
intent.

## Applicability is an image property

A detector's applicability precondition must be checkable from the image itself
at inference time. Calibration scope is a statement about which measured rows
were trusted for fitting; it is not a substitute for an inference-time gate,
because an incoming image does not carry its corpus label. A detector may be
calibrated on a narrower validation population, but it must abstain whenever
the image does not expose the physical evidence its method requires.

## Error Level Analysis (`existing.ela`)

### Principle

JPEG quantisation makes the distortion introduced by a second encode depend on
the block's prior compression history. A region pasted from another source can
therefore have a different response to a controlled re-encode from the host
region. ELA looks for spatial discontinuities in that response. It is a
processing-history cue, not a general image-edit detector.

### Method

The implementation re-encodes a JPEG at a controlled quality, decodes the
result, and compares it with the input. It extracts an edge-discontinuity raw
statistic from the absolute difference map, along with texture, noise, and
compression features. The production adapter maps that statistic with the
calibrated sigmoid above. Unsigned arithmetic must be widened or use a
saturating absolute difference before subtraction; otherwise differences wrap
modulo 256.

### Citation and provenance

N. Krawetz, “A Picture's Worth: Digital Image Analysis and Forensics,” Black
Hat DC, 2008. This repository implements the detector through its own adapter;
no third-party source is retained.

### Signal direction

Higher edge discontinuity and higher compression-artifact evidence are more
suspicious.

### Measured performance

The current primary result is AUC `0.4190 +/- 0.0985` on 47 applicable
R16B-bounded native rows, dated 2026-09-01. The matched parity diagnostic was
`0.3601 +/- 0.0875` on 414 rows. ELA remains a negative, compression-sensitive
finding; the native value is the one used for its declared scope.

### Failure modes

ELA is meaningful only for an already JPEG-compressed source. It manufactures a
re-encoding comparison for PNG and other never-JPEG-compressed inputs, and a
globally re-saved JPEG can have a uniform error level despite a prior edit.

## Noise residual inconsistency (`existing.prnu`)

### Principle

True PRNU is a camera-specific multiplicative sensor pattern. It requires a
reference fingerprint estimated from many images of the same camera. This
repository has no such reference. Its production detector instead runs the
Noisesniffer a-contrario test for locally improbable noise structure, which is
a legitimate blind noise-inconsistency cue but is not camera attribution.

### Method

The production path is the Noisesniffer a-contrario test, not a local-variance
statistic. The image is split into blocks, the blocks are ordered by their
low-frequency energy, and the number of low-noise blocks falling inside a
candidate region is compared against the count expected by chance under the
background model. The raw statistic is the resulting significance,
`-log10(NFA)`; the runtime reports it as `noisesniffer_significance` and maps
it through the calibrated score. The public detector name describes the actual
capability rather than calling the statistic PRNU.

### Citation and provenance

For true PRNU attribution: J. Lukas, J. Fridrich, and M. Goljan, “Digital
Camera Identification from Sensor Pattern Noise,” IEEE Transactions on
Information Forensics and Security, 2006. That method is cited only to say what
this detector is not; it is not implemented here. The implemented method is
M. Gardella, P. Musé, M. Colom, and J.-M. Morel, “Image Forgery Detection Based on Noise
Inspection: Analysis and Refinement of the Noisesniffer Method,” *Image
Processing On Line* 14, article 462, 2024, under Apache-2.0. The
implementation follows that paper but is not a copy of the reference source;
`plan/audit/PAPER-AUDIT-residual.md` and `plan/audit/REPAIR-REPORT-R19-residual.md`
record the remaining deviations.

### Signal direction

Higher local residual variance is more suspicious for a manipulation or
generated image, but it does not identify the camera or the editor.

### Measured performance

The current native result is AUC `0.6282 +/- 0.0743` on 414 applicable rows,
dated 2026-09-01; the matched parity result is `0.6490 +/- 0.0719`. These are
AI-axis measurements of the blind residual, not camera-attribution skill.

### Failure modes

Resizing, heavy JPEG recompression, and denoising can destroy the cue. A
different camera, a normal denoiser, and an edit can all change residual
variance. Sensor attribution remains unavailable without a reference
fingerprint.

## Entropy heuristic (`existing.entropy`)

### Principle

The detector counts pixels where local Shannon entropy agrees across RGB
channels, the entropy field is locally uniform, and local colour is
consistent. Its underlying heuristic is that natural photographs contain more
of these mutually consistent neighborhoods than generated images.

### Method

For each channel and a disk neighborhood, Shannon entropy is estimated from
local intensity histograms. The three maps are compared under a tolerance,
then uniformity and colour-consistency masks produce a matching proportion

```text
matching_proportion = matching_pixels / analyzed_pixels
```

## Neighboring Pixel Relationships statistic (`new.npr`)

### Scope

This is a training-free statistic derived from the NPR representation in Tan
et al., “Rethinking the Up-Sampling Operations in CNN-based Generative Network
for Generalizable Deepfake Detection,” CVPR 2024, arXiv:2312.10461. It is not
the paper's trained classifier, and its measurements must not be compared with
the paper's headline accuracy.

### Method

For each overlapping 2x2 RGB patch, the last pixel is subtracted from the other
three pixels. The detector reports the ratio of mean intra-patch variance to
inter-patch variance, the fraction of near-constant relative patches, and the
Shannon entropy of the quantized difference distribution. Its exploratory
statistic is the fixed equal-weight combination of near-constant fraction,
`1 / (1 + intra_inter_variance_ratio)`, and one minus normalized difference
entropy. It uses only the uploaded image and a bounded RGB derivation, never
corpus membership.

Round 11 measured this statistic but left its fusion weight at zero. The
current implementation is intentionally not a reproduction of NPR's learned
model.

### Physics and direction decision

Tan et al.'s premise is that upsampling creates locally interdependent
neighbouring pixels. In a generated image, relative differences inside a 2x2
patch should therefore be more structured: lower intra-patch variance, more
near-constant patches, and lower difference entropy than in a camera image.
The raw variance ratio and entropy consequently move lower when suspicion
increases. The implementation maps both lower-moving terms to higher suspicion
(`1 / (1 + ratio)` and `1 - normalized_entropy`), while the near-constant
fraction already moves higher. Therefore the final composite's
`higher_is_worse=True` declaration in calibration agrees with the physics.
It was not flipped to chase the AUC. The current R16B parity result is a
genuine below-chance finding on this screen: `0.2803 +/- 0.0846` on 414 rows,
dated 2026-09-01. The matched native result is `0.3696 +/- 0.0876`.

Round 11 measured this statistic and left its fusion weight at zero. It is
training-free, but is intentionally not a reproduction of NPR's learned model.

### Citation and provenance

F. Rohrer, “Detecting AI-Generated Images Using Entropy Analysis,” 2023,
<https://blog.frohrer.com/detecting-ai-generated-images-using-entropy-analysis/>.
This is a single blog heuristic, not a peer-reviewed detector.

### Signal direction

Lower matching proportion is more suspicious for AI generation.

### Measured performance

The current parity result is AUC `0.7616 +/- 0.0559` on 414 applicable rows,
dated 2026-09-01. The matched native result is `0.5305 +/- 0.0828`. This is a
controlled AI-axis screen, not a universal AI-generation accuracy claim.

### Failure modes

Smooth authentic subjects such as sky, blank paper, and studio backgrounds can
have low entropy agreement. The method is not a general splice detector and
its source is not peer reviewed.

## C2PA provenance (`existing.c2pa`)

### Principle

C2PA is declarative signed provenance, unlike the statistical detectors here.
A valid manifest can attest to a creation or editing action, and a broken
signature or hash is evidence of post-signing change. Absence of a manifest is
uninformative because most images have none.

### Method

The detector reads the embedded C2PA store with the pinned Python binding. A
valid generative assertion is positive AI-origin evidence; a failed validation
is a failed-integrity signal. No-manifest input returns `NOT_APPLICABLE`, not a
tampered result, and contributes no fusion term.

### Citation and provenance

Coalition for Content Provenance and Authenticity, *C2PA Technical
Specification 2.x*, <https://c2pa.org/specifications/>; Python bindings,
<https://github.com/contentauth/c2pa-python>. The adapter targets the installed
binding's actual API. It does not copy C2PA source.

### Signal direction

A valid generative assertion or failed validation is stronger evidence than a
statistical cue. No manifest has no direction and is an abstention.

### Measured performance

The native corpus has no valid source-paired AUC for this declarative
capability (30 applicable rows, dated 2026-08-31). Parity has zero applicable
rows because the re-save removes the provenance it measures.

### Failure modes

Manifests can be absent, stripped, or deliberately forged. A valid manifest is
evidence about the signed claims and not a guarantee that the pixels represent
the user's intended event.

## JPEG quantisation tables (`new.qtable_fingerprint`)

### Principle

JPEG DQT markers preserve the quantisation table used by the encoder. Camera
firmware and generic software often use different tables. A camera-labelled
image whose table exactly matches a standard libjpeg table is consistent with a
software re-save, although it is not conclusive because some cameras also use
standard tables.

### Method

Pillow reads the luminance and chrominance tables in zig-zag order. For quality
`Q`, the standard table scaling is

```text
scale = 5000 / Q,             Q < 50
scale = 200 - 2Q,             Q >= 50
value = clamp((base * scale + 50) / 100, 1, 255)
```

The detector reports the quality minimizing the absolute table distance from
the Annex K libjpeg baseline, the minimum `libjpeg_distance`, and a SHA-256
fingerprint of the table bytes. An exact distance of zero is suspicious only
when camera provenance is also claimed.

### Citation and provenance

H. Farid, “Digital Image Ballistics from JPEG Quantization,” Dartmouth
Technical Report TR2006-583, 2006; ITU-T T.81, Annex K. The implementation is
an independent metadata reader.

### Signal direction

Lower distance, especially zero with EXIF Make/Model, is more suspicious.

### Measured performance

The native corpus has no valid source-paired AUC (12 applicable rows, dated
2026-08-31). Parity has zero applicable rows because its uniform encoder
replaces the native quantisation tables. This is `null`, not zero.

### Failure modes

The detector does not apply to PNG or WebP and cannot identify every camera
table. A table can be copied, and generic software can preserve a camera table.
No bundled camera-table database is used.

## Double JPEG / generalized Benford (`new.double_jpeg_benford`)

### Principle

Single JPEG compression produces DCT coefficient magnitudes with a smooth
leading-digit distribution. A second compression with a different quantizer
creates a periodic comb: quantizer multiples are overrepresented and nearby
bins are depleted. That comb and the departure from a fitted generalized
Benford model are evidence of recompression.

### Method

The image is converted to YCbCr, the Y plane is split into 8x8 blocks, and
OpenCV recomputes DCT coefficients after subtracting 128. For AC positions 1
through 20, nonzero magnitudes supply leading digits for the generalized model

```text
p(d) = N log10(1 + 1 / (s + d^q)),  d in {1,...,9}
```

The fitted divergence and the non-DC DFT peak-to-mean ratio of coefficient
histograms over `[-50, 50]` are aggregated. A block map preserves spatial
evidence. The combined raw direction is higher for recompression.

### Citation and provenance

D. Fu, Y. Q. Shi, and W. Su, “A Generalized Benford's Law for JPEG
Coefficients and Its Applications in Image Forensics,” SPIE, 2007; T. Bianchi
and A. Piva, “Image Forgery Localization via Block-Grained Analysis of JPEG
Artifacts,” IEEE TIFS, 2012. This repository recomputes coefficients because
Pillow does not expose JPEG DCT arrays.

### Signal direction

Higher generalized-Benford chi-square divergence and higher periodicity ratio
are more suspicious.

### Measured performance

The current native result is AUC `0.192 +/- 0.082` on 42 applicable rows, dated
2026-08-31. The parity diagnostic is `0.373 +/- 0.088` on 414 rows. Parity
overwrites the history that this detector is intended to inspect, so native is
the primary measurement.

### Failure modes

The method is weak or blind when both encodes use the same quality and
alignment, and it needs enough 8x8 blocks for a stable histogram. Flat images
can have no usable coefficient positions.

## JPEG ghosts (`new.jpeg_ghosts`)

### Principle

If a region was originally compressed at quality `q0`, its squared error from a
controlled re-save tends to have a local minimum near `q0`. A composite with
different compression histories can therefore produce spatially coherent
minima at multiple qualities.

### Method

The detector sweeps qualities 50 through 100 in steps of two. For each quality
it re-encodes and decodes the image, computes mean channel MSE, averages over
16x16 blocks, and min-max normalizes each block's 26-point curve. The per-block
argmin gives the local `q0` map. Distinct spatially coherent modes form the raw
evidence.

### Citation and provenance

H. Farid, “Exposing Digital Forgeries from JPEG Ghosts,” IEEE Transactions on
Information Forensics and Security 4(1), 2009. The implementation is an
independent bounded re-encoding sweep.

### Signal direction

More distinct, spatially coherent quality modes are more suspicious.

### Measured performance

The current native result is AUC `0.4667 +/- 0.0984` on 47 applicable rows,
dated 2026-09-01. The matched parity diagnostic is `0.5220 +/- 0.0834` on 414
rows. The native value is the primary result because parity overwrites JPEG
history.

### Failure modes

The pasted region must be large enough to survive 16x16 averaging. Flattening
the composite at a low final quality can erase both histories. The sweep is
bounded to a 1024px longest side for runtime safety.

## Copy-move (`new.copy_move`)

### Principle

In copy-move forgery the source and destination share the same camera,
compression, and noise history. The distinguishing cue is geometric: two
regions match under a translation or affine transform even though their
content is duplicated within one image.

### Method

SIFT keypoints are detected on a bounded grayscale image. Descriptors are
self-matched with a third neighbor so a keypoint does not select itself. Pairs
with offsets below 32px are rejected, surviving offsets are clustered on an
8px grid, and each cluster is verified with `estimateAffinePartial2D` and
RANSAC at a 3px reprojection threshold. Clusters with at least eight inliers
produce the source/destination mask.

### Citation and provenance

I. Amerini, L. Ballan, R. Caldelli, A. Del Bimbo, and G. Serra, “A SIFT-Based
Forensic Method for Copy-Move Attack Detection and Transformation Recovery,”
IEEE TIFS 6(3), 2011. The implementation uses OpenCV's built-in SIFT and an
independent clustering/verification path.

### Signal direction

At least one verified affine cluster is more suspicious. Fewer than 100
keypoints is `NOT_APPLICABLE`, not a clean result.

### Measured performance

The current native result is AUC `0.386 +/- 0.127` on 82 applicable rows, dated
2026-08-31. The parity diagnostic is `0.561 +/- 0.117` on 88 rows. These are
unpaired AI-axis diagnostics, not evidence that the detector generalises beyond
the tested corpus.

### Failure modes

Smooth pasted regions produce too few keypoints. Blur, large appearance changes,
and severe post-processing reduce descriptor matches. The detector cannot see
a cross-image splice because its match is restricted to one image.

## CFA periodicity (`new.cfa_periodicity`)

### Principle

A Bayer camera samples one colour at each sensor position and interpolates the
other colours. Interpolated samples can leave a 2x2 arrangement in the
intermediate-value masks. This is a camera-capture cue, not a general
AI-origin cue: the absence of that arrangement in a generated PNG is also
consistent with a camera image that was re-encoded as PNG.

### Method

The implementation estimates the dominant Bayer arrangement from per-channel
intermediate-value masks, then checks bounded local windows for a different
arrangement. The raw `cfa_ratio` is the mean confidence of those inconsistent
windows and a map marks their locations. No dominant arrangement returns zero
and is not treated as evidence of AI generation. Full-resolution dimensions
must agree with camera EXIF, and the file must be a strict camera JPEG, before
this detector is allowed to speak.

### Citation and provenance

A. C. Popescu and H. Farid, “Exposing Digital Forgeries in Color Filter Array
Interpolated Images,” IEEE TSP 53(10), 2005; P. Ferrara et al., “Image Forgery
Localization via Fine-Grained Analysis of CFA Artifacts,” IEEE TIFS 7(5), 2012.
The implementation is independently derived from the paper; the AGPL IPOL
reference source is not included.

### Signal direction

Higher local inconsistency is more suspicious for a splice or other
CFA-breaking operation. Absence of a dominant pattern is `NOT_APPLICABLE` at
the input gate, not a generated-image score.

### Measured performance

The current native population has no valid AI-comparison AUC (12 applicable
rows, dated 2026-08-31). Parity has zero applicable rows by design: its JPEG
re-save destroys the capture evidence. A probe also found the same
no-dominant-pattern result on a camera JPEG re-encoded in memory as PNG, so
relaxing the gate would confound generation with re-encoding.

### Failure modes

Downscaling destroys the original sampling relationship. Foveon, monochrome,
and multi-shot pixel-shift sensors do not have a Bayer pattern. Synthetic
splices cannot validate this sensor-provenance cue.

## Spectral peaks (`new.spectral_peaks`)

### Principle

GAN and diffusion decoders repeatedly upsample. Upsampling layers replicate
spectral energy on a regular lattice, whereas natural-image high-pass spectra
are usually smooth and anisotropic without that lattice.

### Method

The grayscale image is resized to 512x512, high-pass filtered with a Gaussian
blur subtraction, multiplied by a 2-D Hann window, and transformed with a
shifted 2-D FFT. The azimuthal mean is subtracted. Outside a radius-five DC
disc and the excluded JPEG grid, the raw statistic is the maximum flattened
peak divided by its standard deviation; a second metric counts local maxima
above four standard deviations.

### Citation and provenance

X. Zhang, S. Karaman, and S.-F. Chang, “Detecting and Simulating Artifacts in
GAN Fake Images,” WIFS, 2019; R. Durall, M. Keuper, and J. Keuper, “Watch your
Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to
Reproduce Spectral Distributions,” CVPR, 2020. The detector is an independent
NumPy/OpenCV implementation.

### Signal direction

Higher peak-to-sigma ratio and more four-sigma maxima are more suspicious for
GAN or diffusion generation.

### Measured performance

The current parity result is AUC `0.508 +/- 0.084` on 414 applicable rows,
dated 2026-08-31. This is an AI-axis screen; synthetic processing-history rows
are not used as sensor-provenance evidence. Round 10's generator-specific
result was strongly non-uniform (`glide` 0.964 +/- 0.027 versus
`stable-diffusion-3.5` 0.529 +/- 0.095), which is why no pooled generator
claim is made here.

### Failure modes

Modern learned upsamplers can leave weak peaks. JPEG recompression can create
its own 8x8 lattice, which is why those frequencies are excluded. A spectral
peak alone does not identify a generator.

## EXIF consistency (`new.exif_consistency`)

### Principle

Metadata is easy to forge and its absence proves nothing. Internal
inconsistency is different: editors often update the main image but leave an
embedded thumbnail, or update some dates and dimensions but not others.

### Method

The detector compares an embedded JPEG thumbnail with a resized full image,
records an editor software tag, checks for a weak missing-camera block, compares
DateTimeOriginal with later date fields, and compares EXIF pixel dimensions with
decoded dimensions. The strongest observed evidence is the maximum of these
signals. No-EXIF input is `NOT_APPLICABLE`.

### Citation and provenance

ExifTool, “EXIF Tags,” <https://exiftool.org/TagNames/EXIF.html>. Tag semantics
are implemented through Pillow and direct TIFF parsing; no metadata library
source is copied.

### Signal direction

Thumbnail mismatch, editor software, date disagreement, and dimension
disagreement are more suspicious. EXIF absence is neutral.

### Measured performance

The latest native result is AUC `0.083 +/- 0.058` on 42 applicable rows, dated
2026-08-31. Parity has zero applicable rows because EXIF is intentionally
removed; EXIF absence itself is not a tampering result.

### Failure modes

Platforms can strip all EXIF, tags can be forged, and an editor can regenerate
the thumbnail. A missing block is weak evidence and is never treated as proof.

## Splicebuster residual co-occurrence (`new.splicebuster`)

### Principle

A camera processing chain leaves a statistical fingerprint in high-frequency
residuals. A splice from a different chain changes the local residual
distribution, even when ordinary compression and noise cues are weak.

### Method

The detector converts the image to grayscale, applies third-order horizontal
and vertical residual filters, and quantizes each residual to one of three
symbols. Four-symbol co-occurrences are accumulated in overlapping 128x128
blocks. A regularized single-Gaussian Mahalanobis model scores the maximum
block distance. This is a bounded implementation of the paper's feature family,
not the paper's two-component EM posterior.

### Citation and provenance

A. Cozzolino, G. Poggi, and L. Verdoliva, “Splicebuster: A New Blind Image
Splicing Detector,” IEEE WIFS, 2015,
<https://doi.org/10.1109/WIFS.2015.7368565>. The implementation is
reconstructed from the paper; the GRIP-UNINA source is not copied.

### Signal direction

Higher maximum block Mahalanobis distance is more suspicious for a change in
processing-chain population.

### Applicability

Splicebuster uses the existing JPEG quantisation-table estimator as a measurable
recompression-strength proxy. It is applicable only to JPEGs whose lowest
estimated libjpeg quality is at least 80; non-JPEG input, missing tables, and
lower-quality JPEGs return `NOT_APPLICABLE`. The cutoff is a data-derived
guard, not a corpus label or calibration scope.

### Measured performance

The latest native result is AUC `0.720 +/- 0.110` on 35 applicable rows, dated
2026-08-31. The native result is the only current primary measurement: the
parity re-save overwrites the processing history this detector is meant to
inspect. A quality gate does not make internet-sourced histories equivalent to
controlled splices.

### Failure modes

Heavy recompression below the measured quality cutoff, resizing, denoising, or a
common web processing chain can erase the cue. Strong texture boundaries and
legitimate multiple pipelines can also resemble a splice. The JPEG quality
proxy is an estimate of the final table, not proof of the complete processing
history.

## Local resampling inconsistency (`new.resampling`)

### Principle

Interpolation creates deterministic relationships among neighboring pixels and
periodic structure in prediction residuals. A global resize is benign; a local
resized or rotated region is suspicious when its periodic signal disagrees with
the surrounding blocks.

### Method

The detector uses a fixed 3x3 linear predictor, takes the absolute prediction
residual, and measures the non-DC peak-to-background ratio of a windowed 2-D
DFT in bounded 128x128 blocks. The raw score is the 75th-percentile absolute
deviation from the block-median peak ratio, so a uniform global resize is not
treated as tampering.

### Citation and provenance

A. C. Popescu and H. Farid, “Exposing Digital Forgeries by Detecting Traces
of Resampling,” IEEE TSP 53(2), 2005; M. Kirchner, “Fast and Reliable
Resampling Detection by Spectral Analysis of Fixed Linear Predictor Residue,”
ACM MM&Sec, 2008. Both are cited in the bibliography and the implementation is
independent.

### Signal direction

Higher local block disagreement is more suspicious for local resampling.

### Measured performance

The current native result is AUC `0.298 +/- 0.094` on 360 applicable rows,
dated 2026-08-31. The parity diagnostic is `0.240 +/- 0.082` on 414 rows.
These are exploratory because the corpus has no labeled local-resampling
positive family; the two encodings are reported separately and no universal
skill is claimed.

### Failure modes

Small, heavily compressed, or smoothly textured regions may not preserve the
periodic signal. A global web resize can be indistinguishable from benign
processing and is intentionally not flagged.

## ZERO JPEG grid origin (`new.zero`)

### Principle

JPEG block grids are phase-aligned. A pasted or independently compressed region
can retain a different 8x8 origin, producing a locally foreign phase even when
the image's dominant grid is coherent.

### Method

Overlapping 8x8 windows vote for one of 64 possible grid phases. Global and
foreign local regions are tested with binomial a-contrario tail probabilities.
For support `k` among `n` votes, the log-NFA combines the binomial tail with
the number of tested grid positions and image locations. Significant foreign
regions become the localization mask. Sampling is bounded per cell to avoid an
exhaustive DCT runtime cliff.

### Citation and provenance

Nikoukhah et al., “ZERO: A Local JPEG Grid Origin Detector Based on the Number
of DCT Zeros and its Applications in Image Forensics,” *Image Processing On
Line* 11, 2021, article 390, <https://doi.org/10.5201/ipol.2021.390>.
The IPOL reference is AGPL-3.0-or-later; this repository reimplements the
paper and retains no AGPL source.

### Signal direction

More significant foreign-grid evidence, represented by a lower log-NFA, is
more suspicious.

### Measured performance

The latest native result is AUC `0.275 +/- 0.084` on 402 applicable rows, dated
2026-08-31. Parity was not used as a primary result because the uniform JPEG
re-save overwrites the grid history; the detector remains a negative finding
on this screen and has zero fusion weight.

### Failure modes

The cue disappears when the foreign grid is erased or realigned. Small images,
flat blocks, and strong final recompression reduce useful votes. The bounded
sampling is an explicit runtime tradeoff.

## Face deepfake ONNX model (`new.learned_onnx`)

### Principle

This is a learned classifier, not a classical forensic measurement. Its model
was trained to classify face deepfakes, so its output may carry evidence for
that task but does not generalize to arbitrary splices, documents, or scenes.
The adapter first runs OpenCV's bundled
`haarcascade_frontalface_default.xml` on the uploaded image. No detected face
returns `NOT_APPLICABLE`; this is an inference-time image precondition, not a
corpus-membership check.

### Method

The optional ONNX model converts RGB input to 224x224, scales by 1/255,
normalizes with mean and standard deviation 0.5, and returns the softmax
probability of the `Deepfake` label. ONNX Runtime is imported lazily. Missing
runtime or weights returns `NOT_APPLICABLE`.

### Citation and provenance

The model artifact and its Apache-2.0 license are recorded in
`new.learned_onnx` in the catalog and fetched by the repository's pinned model
script. The detector adapter is local code. No torch is part of this system.

### Signal direction

Higher `Deepfake` probability is more suspicious only for face-deepfake inputs.

### Measured performance

The optional model was present for R16A. After the face gate, the current
parity AI-axis AUC is `0.184 +/- 0.131` on 140 applicable rows, dated
2026-09-01; the native comparison is `0.423 +/- 0.137` on 116 rows. The
per-generator parity results include `0.091 +/- 0.106` for
`stable-diffusion-1-3` and `0.125 +/- 0.119` for FLUX, so the model is not a
general AI detector and its fusion weight remains zero. Runs without external
weights must report null rather than treating `NOT_APPLICABLE` as negative.

### Failure modes

It is not a general splice detector, receipt/document forgery detector, or
image-origin oracle. It is opt-in and is image-gated to faces; even when its
weights are installed, it must remain scoped to that task.

## Frozen CLIP linear probe (`new.clip_probe`)

### Principle

A frozen vision-language backbone supplies a broad image representation. A
linear probe fitted on this repository's corpus can use that representation
for an AI-generation screen without updating the backbone. This is a learned
classifier, not a forensic proof and not a face or splice detector.

### Method

The optional adapter uses open-clip-torch's `ViT-L-14` architecture with the
MIT-licensed LAION `CLIP-ViT-L-14-laion2B-s32B-b82K` weights. The backbone is
loaded locally, put in evaluation mode, and all parameters are frozen. Only
the external linear probe is fitted. `scripts/fit_clip_probe.py` uses the
repository's standardized logistic calibration routine, groups rows by
`source_image`, and holds out complete generator names. It reports both
in-distribution performance on generators seen during fitting and the
out-of-distribution result on generators never seen during fitting; the latter
is the meaningful generalization measurement.

The probe is image-side only: it checks decoded format, image size, and the
presence of its local backbone and probe files. Missing optional dependencies
or weights return `NOT_APPLICABLE`; no manifest axis or generator name is
consulted at inference.

### Measured performance

R12's `1.0000 +/- 0.0000` result is retired as forensic evidence: it was a
container-format artifact. R16A still measured `0.999585 +/- 0.000757` on the
parity rows (rounds to 1.000) and `0.999793 +/- 0.000532` on native rows, dated
2026-09-01. Both use only 12 real-camera negatives against 402 AI rows. The
parity re-save removes the format shortcut, but the CLIP separation can still
be a content or corpus confound; it is not generation proof.

### Failure modes

The probe can learn corpus, format, content, or generator-family shortcuts and
must be re-evaluated on re-encoded and genuinely cross-domain images. It is
not an origin oracle, splice localizer, or proof that an image came from a
particular generator.

## AEROBLADE-style reconstruction (`new.aeroblade`)

### Principle

Latent diffusion generates images by decoding a VAE latent. An image produced
by that same autoencoder lies near its reconstruction manifold and can be
reconstructed with unusually low error. This is a generator-family cue, not a
general manipulation detector.

### Method

The optional adapter resizes an image to a bounded multiple-of-eight shape,
maps RGB to `[-1,1]`, encodes with the distilled TAESD autoencoder, decodes,
and computes the paper's perceptual distance with LPIPS:

```text
reconstruction_lpips = LPIPS(input, decoded)
```

Lower error is mapped as more suspicious. TAESD and the LPIPS AlexNet cache are
external and are never bundled in this tree. The implementation uses a
distilled approximation, not the exact Stable Diffusion autoencoder used for
the paper's headline results.

### Citation and provenance

J. Ricker, D. Lukovnikov, and A. Fischer, “AEROBLADE: Training-Free Detection
of Latent Diffusion Images Using Autoencoder Reconstruction Error,” CVPR,
2024, arXiv:2401.17879. The algorithm is independently reimplemented. The
runtime uses the MIT-licensed `madebyollin/taesd` Diffusers artifact and LPIPS,
both pinned and fetched by `scripts/fetch_model.py`. The distilled model is
not the exact autoencoder used for the paper's headline results.

### Signal direction

Lower LPIPS reconstruction error is more suspicious for latent-diffusion
output.

### Measured performance

R16A's current parity AI-axis AUC is `0.416 +/- 0.088` on 414 rows, dated
2026-09-01; the native comparison is `0.547 +/- 0.082`. Per-generator parity
falls from native `0.668 +/- 0.079` to `0.456 +/- 0.093` for FLUX, and from
`0.386 +/- 0.100` to `0.189 +/- 0.082` for `stable-diffusion-1-3`.

This contradicts the paper's reported mean AP of `0.992` across its tested
latent-diffusion generators; AP and this pooled AUC are not identical metrics,
but the direction and magnitude still warrant the discrepancy being recorded.
It does not disprove the paper. Honest candidate
explanations are the distilled TAESD stand-in rather than the paper's exact
autoencoder, corpus and post-processing differences, or an implementation
error in this repository. The current adapter is latent-diffusion-specific,
and its fusion weight remains zero.

### Failure modes

This detector is latent-diffusion-only and is useless against splicing,
copy-move, or GAN output. Distillation can materially lower performance
relative to the paper. Missing optional dependencies or model artifacts
produce an abstention rather than a clean verdict.

## What this system cannot do

- It cannot prove that an image is authentic, identify the human editor, or
  establish what event a photograph depicts.
- It cannot perform true camera attribution: the noise detector has no PRNU
  reference fingerprint, and CFA evidence is unavailable when capture
  dimensions or sensor assumptions are missing.
- A global resize is not itself tampering. The resampling detector looks for
  local block disagreement and abstains on images too small for a stable
  comparison; it does not label every web-sized holiday photo as forged.
- It cannot reliably detect edits that preserve the relevant cue, including
  same-quality JPEG recompression, erased or realigned JPEG grids, smooth
  copy-move regions, or carefully counter-forensic resampling.
- It cannot generalize the optional face model beyond face deepfakes or the
  AEROBLADE-style model beyond latent-diffusion output. Neither is a universal
  AI-image oracle.
- C2PA absence is not evidence of tampering, EXIF absence is not evidence of
  tampering, and any statistical score is only evidence from one measurement
  family.
- The current corpus is small and heterogeneous, the synthetic rows simulate
  processing history rather than sensor provenance, and the catalog's
  within-source estimates must not be presented as open-web error rates.

## Licensing and implementation provenance

| Method | Repository treatment | Upstream license/provenance | Source retained? |
|---|---|---|---|
| Splicebuster | Reimplemented from paper | GRIP-UNINA source is not permissively licensed | No |
| Kirchner/Popescu-Farid resampling | Reimplemented from papers | Paper method only | No |
| ZERO | Reimplemented from paper | IPOL reference is AGPL-3.0-or-later | No |
| CFA periodicity | Reimplemented from paper | IPOL reference is AGPL-3.0-or-later | No |
| Noisesniffer residual | Adapted implementation | IPOL 2024/462, Apache-2.0 | No reference source; local adaptation only |
| AEROBLADE score | Reimplemented from paper | AEROBLADE repository has no project license | No |
| TAESD runtime | Adapted external artifact | MIT-licensed `madebyollin/taesd`, pinned revision and checksums in the fetch script | No weights |
| CLIP linear probe | Reimplemented from paper direction | MIT `open-clip-torch`; MIT LAION ViT-L/14 weights; local probe trained from this corpus | No weights |
| Face learned model | Adapted runtime integration | Cataloged ONNX artifact, Apache-2.0 | No weights |
| JPEG, ELA, EXIF, spectral, copy-move, double-JPEG | Local implementations | Primary papers and tool specifications cited above | No third-party source |

The table records a legal boundary: a published algorithm may be derived from,
but a source file may not be copied or translated into this repository unless
its license permits that use. Model weights are external, optional artifacts
and are not committed. Under decision D6, TruFor, Noiseprint++, and Comprint
remain excluded: their available implementations are nonprofit-use-only, do
not provide a verified compatible ONNX export/direct weight path for this
runtime, and are incompatible with this repository's distribution terms. No
substitute implementation or weights are claimed.

## Bibliography

1. N. Krawetz, “A Picture's Worth: Digital Image Analysis and Forensics,”
   Black Hat DC, 2008.
2. J. Lukas, J. Fridrich, and M. Goljan, “Digital Camera Identification from
   Sensor Pattern Noise,” IEEE TIFS, 2006.
3. F. Rohrer, “Detecting AI-Generated Images Using Entropy Analysis,” 2023,
   <https://blog.frohrer.com/detecting-ai-generated-images-using-entropy-analysis/>.
4. Coalition for Content Provenance and Authenticity, *C2PA Technical
   Specification 2.x*, <https://c2pa.org/specifications/>.
5. H. Farid, “Digital Image Ballistics from JPEG Quantization,” Dartmouth TR2006-583, 2006; ITU-T T.81 Annex K.
6. D. Fu, Y. Q. Shi, and W. Su, “A Generalized Benford's Law for JPEG
   Coefficients and Its Applications in Image Forensics,” SPIE, 2007.
7. T. Bianchi and A. Piva, “Image Forgery Localization via Block-Grained
   Analysis of JPEG Artifacts,” IEEE TIFS, 2012.
8. H. Farid, “Exposing Digital Forgeries from JPEG Ghosts,” IEEE TIFS 4(1),
   2009.
9. I. Amerini, L. Ballan, R. Caldelli, A. Del Bimbo, and G. Serra, “A
   SIFT-Based Forensic Method for Copy-Move Attack Detection and
   Transformation Recovery,” IEEE TIFS 6(3), 2011.
10. A. C. Popescu and H. Farid, “Exposing Digital Forgeries in Color Filter
    Array Interpolated Images,” IEEE TSP 53(10), 2005.
11. P. Ferrara et al., “Image Forgery Localization via Fine-Grained Analysis
    of CFA Artifacts,” IEEE TIFS 7(5), 2012.
12. X. Zhang, S. Karaman, and S.-F. Chang, “Detecting and Simulating Artifacts
    in GAN Fake Images,” WIFS, 2019.
13. R. Durall, M. Keuper, and J. Keuper, “Watch your Up-Convolution: CNN Based
    Generative Deep Neural Networks are Failing to Reproduce Spectral
    Distributions,” CVPR, 2020.
14. ExifTool, “EXIF Tags,” <https://exiftool.org/TagNames/EXIF.html>.
15. Nikoukhah et al., “ZERO: A Local JPEG Grid Origin Detector Based on the
    Number of DCT Zeros and its Applications in Image Forensics,” IPOL 11,
    2021, <https://doi.org/10.5201/ipol.2021.390>.
16. M. Gardella, P. Musé, M. Colom, and J.-M. Morel, “Image Forgery Detection
    Based on Noise Inspection: Analysis and Refinement of the Noisesniffer
    Method,” IPOL 14, article 462, 2024,
    <https://www.ipol.im/pub/art/2024/462/>.
17. A. Cozzolino, G. Poggi, and L. Verdoliva, “Splicebuster: A New Blind Image
    Splicing Detector,” IEEE WIFS, 2015, DOI 10.1109/WIFS.2015.7368565.
18. A. C. Popescu and H. Farid, “Exposing Digital Forgeries by Detecting Traces
    of Resampling,” IEEE TSP 53(2), 2005, DOI 10.1109/TSP.2004.839932.
19. M. Kirchner, “Fast and Reliable Resampling Detection by Spectral Analysis
    of Fixed Linear Predictor Residue,” ACM MM&Sec, 2008, DOI 10.1145/1411328.1411333.
20. J. Ricker, D. Lukovnikov, and A. Fischer, “AEROBLADE: Training-Free
    Detection of Latent Diffusion Images Using Autoencoder Reconstruction
    Error,” CVPR, 2024, arXiv:2401.17879.
21. V. Novozámský, B. Mahdian, and J. Saic, “IMD2020: A Large-Scale Annotated
    Dataset Tailored for Detecting Manipulated Images,” IEEE WACV Workshops,
    2020.
