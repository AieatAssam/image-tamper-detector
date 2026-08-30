# Detection principles

This is the scientific reference for the detector set. The machine-readable
registry is [`plan/reference/detector-catalog.yaml`](../plan/reference/detector-catalog.yaml).
It is the source of truth for detector names, signal directions, applicability,
limitations, citations, and measured performance. In particular, the current
`within_source_auc` and `auc_standard_error` values are not copied into this
document: follow each detector's catalog key to read the current value.

## How to read the measurements

The benchmark compares manipulated or AI-generated rows only with authentic
rows sharing the same `source_image`. This prevents the image source from
becoming a hidden class label. AUC is the probability that a randomly selected
positive row scores above its paired negative, with ties counted as one half.
The reported uncertainty is the Hanley-McNeil estimate:

```text
SE(A) = sqrt([A(1-A) + (n+ - 1)(Q1 - A^2) + (n- - 1)(Q2 - A^2)] / (n+ n-))
Q1 = A / (2 - A)
Q2 = 2A^2 / (1 + A)
```

The catalog stores the AUC, standard error, participating class counts, and
corpus revision under each calibrated detector entry. A null means that the
corpus did not contain a valid paired comparison. It is not a zero score.

The complete calibration corpus currently contains 526 rows in 227 source
groups: 100 synthetic processing-history rows and 426 rows from the local
manifest, including 400 source-directory-stratified IMD2020 rows, 12 strict
real-camera rows, 12 real-AI rows, and two C2PA fixtures. The synthetic rows
are useful for compression, geometry, and controlled processing experiments;
they cannot establish sensor provenance. The synthetic and real portions must
therefore be read as different validation populations, not pooled evidence of
one universal detector skill.

Every applicable raw statistic is mapped to a probability using

```text
score = sigmoid((s - t) / w)                  higher statistic is worse
score = sigmoid((t - s) / w)                  lower statistic is worse
sigmoid(x) = 1 / (1 + exp(-x))
```

`NOT_APPLICABLE` is an abstention: its score is null and it contributes
nothing to fusion. A detector result is evidence, not a proof of origin or
intent.

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

See `measurements.detectors.ela` in the catalog for the current
within-source AUC, standard error, applicable population, and corpus revision.
The result is a documented negative finding in the current calibration, near
0.44 AUC, and its fusion weight is consequently zero. It is retained so later
corpora can test whether the weakness is corpus-specific.

### Failure modes

ELA is meaningful only for an already JPEG-compressed source. It manufactures a
re-encoding comparison for PNG and other never-JPEG-compressed inputs, and a
globally re-saved JPEG can have a uniform error level despite a prior edit.

## Noise residual inconsistency (`existing.prnu`)

### Principle

True PRNU is a camera-specific multiplicative sensor pattern. It requires a
reference fingerprint estimated from many images of the same camera. This
repository has no such reference. Its production detector instead measures
whether a denoising residual has unusually different local variance, which is a
legitimate blind noise-inconsistency cue but is not camera attribution.

### Method

A wavelet/Gaussian residual is formed from the image and its local variance is
aggregated into a global raw statistic. Higher residual variance is passed
through the calibrated score mapping. The public detector name describes the
actual capability rather than calling the statistic PRNU.

### Citation and provenance

For true PRNU attribution: J. Lukas, J. Fridrich, and M. Goljan, “Digital
Camera Identification from Sensor Pattern Noise,” IEEE Transactions on
Information Forensics and Security, 2006. The blind residual implementation
also adapts the block-selection and a-contrario ideas of M. Gardella, P.
Musé, M. Colom, and J.-M. Morel, “Image Forgery Detection Based on Noise
Inspection: Analysis and Refinement of the Noisesniffer Method,” *Image
Processing On Line* 14, article 462, 2024, under Apache-2.0. The adaptation is
not a copy of the reference source.

### Signal direction

Higher local residual variance is more suspicious for a manipulation or
generated image, but it does not identify the camera or the editor.

### Measured performance

See `measurements.detectors.prnu` in the catalog. The value is the
within-source result for the blind noise residual, not a claim of PRNU
identification skill.

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

Lower proportion is mapped as more suspicious. The decision direction is
correct, but a matching proportion is not itself a calibrated confidence.

### Citation and provenance

F. Rohrer, “Detecting AI-Generated Images Using Entropy Analysis,” 2023,
<https://blog.frohrer.com/detecting-ai-generated-images-using-entropy-analysis/>.
This is a single blog heuristic, not a peer-reviewed detector.

### Signal direction

Lower matching proportion is more suspicious for AI generation.

### Measured performance

See `measurements.detectors.entropy` in the catalog. The current
within-source result is a documented negative finding, near 0.47 AUC, so the
calibration guard assigns zero fusion weight.

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

See `measurements.detectors.c2pa` in the catalog. The current corpus
does not provide a valid source-paired AUC for this declarative capability.

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

See `measurements.detectors.qtable` in the catalog. The current
IMD2020 population is entirely abstaining because the required camera
provenance gate is absent, so its AUC is null rather than zero. The catalog
also records the applicable synthetic and camera populations where present.

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

See `measurements.detectors.double_jpeg` in the catalog. Its
within-source result is the current strongest single calibrated cue, but the
value remains corpus-dependent and is not a universal JPEG-forgery guarantee.

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

See `measurements.detectors.jpeg_ghosts` in the catalog for the source-paired
AUC and standard error.

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

See `measurements.detectors.copy_move` in the catalog for the within-source
AUC, standard error, and applicable count.

### Failure modes

Smooth pasted regions produce too few keypoints. Blur, large appearance changes,
and severe post-processing reduce descriptor matches. The detector cannot see
a cross-image splice because its match is restricted to one image.

## CFA periodicity (`new.cfa_periodicity`)

### Principle

A Bayer camera samples one colour at each sensor position and interpolates the
other colours. Interpolated samples are statistically smoother in a 2x2
periodic arrangement. Rendered, resampled, or inserted content does not retain
the same arrangement.

### Method

For each colour plane and each Bayer phase, interpolation residuals are formed
from Bayer-subsampled positions. Sliding 32x32 windows estimate the variance in
the four 2x2 classes. The raw ratio is

```text
cfa_ratio = mean(two lowest class variances) /
            mean(two highest class variances)
```

The phase with strongest CFA structure is selected. A local ratio map is
returned. Full-resolution dimensions must agree with camera EXIF before this
detector is allowed to speak.

### Citation and provenance

A. C. Popescu and H. Farid, “Exposing Digital Forgeries in Color Filter Array
Interpolated Images,” IEEE TSP 53(10), 2005; P. Ferrara et al., “Image Forgery
Localization via Fine-Grained Analysis of CFA Artifacts,” IEEE TIFS 7(5), 2012.
The implementation is independently derived from the paper; the AGPL IPOL
reference source is not included.

### Signal direction

A ratio closer to one means CFA structure is absent and is more suspicious.

### Measured performance

See `measurements.detectors.cfa` in the catalog. The current
IMD2020 slice has zero applicable rows because strict camera-original evidence
is missing, so AUC and standard error are null. That abstention is a finding,
not a zero-performance result.

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

See `measurements.detectors.spectral` in the catalog. Its real-camera
and real-AI scope is distinct from the synthetic splice scope; synthetic rows
are not used as sensor-provenance evidence.

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

See `measurements.detectors.exif` in the catalog. Only two
IMD2020 rows currently have applicable EXIF evidence and they do not form a
valid source-paired comparison, so the AUC is null.

### Failure modes

Platforms can strip all EXIF, tags can be forged, and an editor can regenerate
the thumbnail. A missing block is weak evidence and is never treated as proof.

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

See `measurements.detectors.zero` in the catalog. The current result is a
documented near-chance finding, around 0.50 AUC, and the guard gives it zero
fusion weight.

### Failure modes

The cue disappears when the foreign grid is erased or realigned. Small images,
flat blocks, and strong final recompression reduce useful votes. The bounded
sampling is an explicit runtime tradeoff.

## Face deepfake ONNX model (`new.learned_onnx`)

### Principle

This is a learned classifier, not a classical forensic measurement. Its model
was trained to classify face deepfakes, so its output may carry evidence for
that task but does not generalize to arbitrary splices, documents, or scenes.

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

See `measurements.detectors.learned` in the catalog. The optional model
was present for this calibration and has a source-paired measurement; future
runs without its external weights must report null rather than treating
NOT_APPLICABLE as a negative result.

### Failure modes

It is not a general splice detector, receipt/document forgery detector, or
image-origin oracle. It is opt-in and must remain low-weight even when its
weights are installed.

## AEROBLADE-style reconstruction (`new.aeroblade`)

### Principle

Latent diffusion generates images by decoding a VAE latent. An image produced
by that same autoencoder lies near its reconstruction manifold and can be
reconstructed with unusually low error. This is a generator-family cue, not a
general manipulation detector.

### Method

The optional adapter resizes an image to the external TAESD ONNX encoder's
shape, maps RGB to `[0,1]`, encodes to a latent, decodes, clips the reconstruction
to `[0,1]`, and computes

```text
reconstruction_l1 = mean(abs(decoded - input))
```

The paper uses LPIPS. This implementation uses mean L1 because it does not add
a perceptual backbone. Lower error is mapped as more suspicious. The ONNX
encoder and decoder paths are external and are never bundled in this tree.

### Citation and provenance

J. Ricker, D. Lukovnikov, and A. Fischer, “AEROBLADE: Training-Free Detection
of Latent Diffusion Images Using Autoencoder Reconstruction Error,” CVPR,
2024, arXiv:2401.17879. The algorithm is independently reimplemented. The
runtime uses the distilled MIT-licensed TAESD ONNX conversion pinned in
`backend/app/analysis/aeroblade.py`, derived from the MIT
`madebyollin/taesd` project. The distilled model is not the exact autoencoder
used for the paper's headline results.

### Signal direction

Lower mean L1 reconstruction error is more suspicious for latent-diffusion
output.

### Measured performance

See `measurements.detectors.aeroblade` in the catalog. The detector is opt-in
and the external model files are absent from the standard calibration corpus,
so no corpus AUC is claimed. Any future number must state that it uses distilled
TAESD and L1 rather than exact-AE LPIPS.

### Failure modes

This detector is latent-diffusion-only and is useless against splicing,
copy-move, or GAN output. Distillation and L1 substitution can materially
lower performance relative to the paper. Missing ONNX artifacts produce an
abstention rather than a clean verdict.

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
| TAESD runtime pair | Adapted external artifact | MIT-licensed distilled TAESD conversion, pinned in code | No weights |
| Face learned model | Adapted runtime integration | Cataloged ONNX artifact, Apache-2.0 | No weights |
| JPEG, ELA, EXIF, spectral, copy-move, double-JPEG | Local implementations | Primary papers and tool specifications cited above | No third-party source |

The table records a legal boundary: a published algorithm may be derived from,
but a source file may not be copied or translated into this repository unless
its license permits that use. Model weights are external, optional artifacts
and are not committed.

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
