# Paper-fidelity audit: aigen

Date: 2026-09-01  
Scope: read-only audit of `spectral.py`, `npr.py`, `aeroblade.py`, and `entropy.py` plus their catalog entries. No product files were changed.

## Grades

| Detector | Grade | Short reason |
|---|---|---|
| `spectral` | **MAJOR-DRIFT** | The implementation is a new handcrafted peak detector, not either cited paper's spectrum-plus-classifier method. |
| `npr` | **MAJOR-DRIFT** | The representation, patch layout, statistics, and classifier differ materially from the NPR paper. |
| `aeroblade` | **MAJOR-DRIFT** | The AE reconstruction cue is present, but the published multi-AE minimum, selected LPIPS variant, and paper preprocessing are absent. |
| `entropy` | **UNVERIFIED** | The catalog citation is a blog post, not a paper. The blog implementation also differs materially from this code. |

The grades are against the published methods, not against whether a simplified
cue can still be useful. The primary sources are [Zhang et al., arXiv:1907.06515](https://arxiv.org/pdf/1907.06515), [Durall et al., CVPR 2020 / arXiv:2003.01826](https://arxiv.org/pdf/2003.01826), [Tan et al., CVPR 2024 / arXiv:2312.10461](https://arxiv.org/pdf/2312.10461), and [Ricker et al., CVPR 2024 / arXiv:2401.17879](https://arxiv.org/html/2401.17879). The entropy citation resolves to [Rohrer's blog post](https://blog.frohrer.com/detecting-ai-generated-images-using-entropy-analysis/), not a paper.

## `spectral` — MAJOR-DRIFT

### Paper specification

The catalog cites both the Zhang WIFS paper and the Durall CVPR paper, but they
specify different spectrum-based classifiers.

Zhang et al. compute the 2-D DFT independently on all three RGB channels,
discard phase, take the logarithmic spectrum, normalize it to `[-1, 1]`, and
feed the resulting three-channel spectrum to a classifier. Their stated
implementation uses ImageNet-pretrained ResNet-34, a random `224x224` crop from
`256x256` training inputs, and a central `224x224` test crop. The paper's
relevant wording is: “apply the 2D DFT to each of the RGB channels” and
“normalize the logarithmic spectrum to [-1, 1].”

Durall et al. first convert input images to grayscale, compute the DFT power
spectrum, azimuthally integrate it into a 1-D radial representation, normalize
by the zero-frequency coefficient, scale the feature vector to a fixed size,
and classify with SVM (or K-Means for their unsupervised experiment). Their
published pipeline explicitly says input images are “converted to grey-scale
before DFT” and uses “a basic SVM” after azimuthal integration.

The shared physical premise is sound: upsampling can replicate or distort
frequency content, so generated images can have stronger non-natural spectral
structure. This supports a higher generated-image score for a correctly
defined peak/deviation statistic.

### Our implementation

- `spectral.py:42-45` converts the already downscaled RGB input to grayscale and
  calibrates a scalar `peak_sigma` with `higher_is_worse`.
- `spectral.py:61-68` resizes every image to `512x512`, subtracts a Gaussian
  blur with `sigmaX=1.0`, applies a 2-D Hann window, computes log magnitude, and
  subtracts a radial average.
- `spectral.py:70-88` removes a radius-5 DC disc and a custom JPEG-grid mask,
  then reports the maximum standardized residual and local maxima above `4.0`
  sigma.
- The catalog records exactly those choices at
  `plan/reference/detector-catalog.yaml:362-368` and says the expected direction
  is higher peak-to-sigma at `:368`. Runtime calibration currently has
  `higher_is_worse: true` at `backend/app/analysis/calibration.json:363-373`.

### Deltas

1. **The detector is not either published detector.** Zhang's RGB log-spectrum
   and classifier are absent. Durall's power-spectrum azimuthal integration,
   DC normalization, fixed-length feature, and SVM/K-Means are also absent.
   The code instead implements a grayscale high-pass/Hann/radial-flattening
   peak heuristic (`spectral.py:61-88`). This is a major method substitution,
   not a parameter drift.

2. **The constants are repository choices, not paper values.** The `512x512`
   resize, Gaussian `sigma=1.0`, radius-5 exclusion, JPEG-mask neighborhood
   `3`, and local-max threshold `4.0` are all in `spectral.py:61-85`. The Zhang
   paper specifies `256x256` inputs with `224x224` crops; Durall instead scales
   the 1-D radial feature after azimuthal integration. Neither cited method
   specifies this set of peak-detector constants.

3. **The JPEG-grid rule is not supported by the cited papers.** The catalog
   says JPEG frequencies “MUST” be excluded at
   `plan/reference/detector-catalog.yaml:377-379`, implemented as the custom
   `64`-period mask at `spectral.py:101-107`. Zhang's paper instead reports that
   JPEG compression and resize destroy the upsampling artifact and that
   retraining on post-processed data improves performance. It does not prescribe
   masking a JPEG lattice. The mask can also remove genuine generator peaks,
   so it needs an ablation before being treated as a fidelity repair.

4. **Preprocessing is added and materially changes the signal.** The shared
   route first caps the longest side at `1600` with Lanczos resizing
   (`backend/app/analysis/base.py:77-87`), then `spectral.py` resizes again to
   `512x512`. Neither paper requires this two-stage path. Since the cue is a
   frequency representation, resizing changes the frequency coordinates and
   attenuates the high-frequency artifact being measured.

### Signal direction

The direction is **consistent with the physical premise**. Upsampling artifacts
are spectral replicas/distortions; the code's `peak_sigma` increases when a
non-radial peak dominates its background, and `higher_is_worse: true` is
therefore the correct direction for this custom statistic. This is not an
inversion like the double-JPEG incident. It does not make the statistic a
faithful implementation of either paper, whose final classifier direction is
learned from labels.

### Prioritised fixes

1. **P0:** Choose one cited method and implement its complete published path,
   including its spectrum representation and classifier. Otherwise rename and
   document this as a repository-specific spectral heuristic rather than a
   paper implementation. This is the main fidelity failure.
2. **P1:** Remove or experimentally justify the JPEG-grid mask. The cited
   evidence supports post-processing-aware training/evaluation, not this mask.
3. **P1:** Re-evaluate the `1600 -> 512` resizing and all heuristic thresholds
   on a paper-matched input protocol before interpreting AUC as method evidence.

## `npr` — MAJOR-DRIFT

### Paper specification

Tan et al. model an image as the output of an upsampling layer followed by
convolution. They divide the output into `W x H` grids, each an `l x l` patch,
and construct NPR separately for each color channel. For a patch vector
`v_I^c = {w_1, ..., w_n}`, they form every `w_i - w_j`, including the zero
reference element, and state that their chosen values are `l=2` and `j=1`.
The paper says: “The NPR is employed to train detector as artifact
representation.” It then trains a lightweight CNN with convolution and a
ResNet block, `1.44` million parameters, Adam learning rate `2e-4`, and batch
size `32`.

The physical direction is that nearest-neighbor upsampling makes local `2x2`
values equal before later convolution, so generated output can have stronger
local dependence and smaller relative differences. The paper's final decision
direction is learned by the classifier; it does not define the variance/entropy
composite used here.

### Our implementation

- `npr.py:79-84` creates every one-pixel-stride adjacent `2x2` window and uses
  the bottom-right pixel as reference.
- `npr.py:85-91` collapses RGB into scalar sums, computes a custom
  intra/inter-variance ratio, and defines a near-constant fraction with the
  threshold `(1/255)^2`.
- `npr.py:93-104` quantizes differences to integer bins, appends synthetic zero
  values, computes entropy over `511` bins, and averages three hand-built
  suspicion terms with equal weight.
- `npr.py:53-60` applies a fixed calibration and `higher_is_worse`; the catalog
  describes this as a “training-free statistic derived from the paper” at
  `plan/reference/detector-catalog.yaml:552-580`.

### Deltas

1. **The paper's classifier is absent.** There is no NPR CNN/ResNet block,
   training source, loss, or learned decision function in `npr.py`. The code is
   an exploratory scalar proxy, as the catalog admits at `:557` and `:579`.
   It cannot inherit the paper's generalization or headline accuracy.

2. **Patch topology differs.** The paper divides the image into `2x2` grids
   aligned to the upsampling factor (`2312.10461`, Sec. 3.3); the implementation
   scans overlapping windows at every pixel (`npr.py:79-84`). The catalog's
   “overlapping 2x2” description at `:563` is itself a paper-fidelity mismatch.

3. **Per-channel NPR is collapsed.** The paper's `v_I^c` and `\hat v_I^c` are
   per-channel representations. The implementation sums all three channels
   before variance (`npr.py:85-87`) and therefore does not preserve the paper's
   `NPR-R`, `NPR-G`, and `NPR-B` representation.

4. **The reference and fourth element are inconsistent.** The paper sets
   `j=1` but permits any element as reference. The code uses the fourth/bottom-
   right element (`npr.py:79-84`), so this is a minor constant drift rather than
   the main failure. The paper's permission for any `w_j` makes the choice
   defensible only if the rest of the representation is preserved.

5. **The complete difference vector is not used for the local statistics.**
   The paper forms all four values `w_i-w_j`; the implementation has only three
   neighbor tensors in `npr.py:80-84`. It appends zeros only to the entropy
   histogram (`:93-96`), not to `intra_variance` or `near_constant`. The
   variance is nevertheless divided by `12` at `:87`, although the preceding
   sums contain only three neighbors times three channels. This is not the
   paper's representation and is internally inconsistent even as a proxy.

6. **The statistics and thresholds are invented.** The paper specifies `l=2`,
   `j=1`, and the classifier training settings, but not `(1/255)^2`, `511`
   entropy bins, equal weights, inverse-ratio mapping, a `4x4` applicability
   floor, or the `1024` analysis-side cap (`npr.py:12-24`, `:90-104`).

### Signal direction

For the final custom composite, the direction is **physically consistent**:
low ratio, more near-constant patches, and lower difference entropy are each
mapped toward a higher suspicion value by `npr.py:102-104`, and calibration
keeps `higher_is_worse: true` (`calibration.json:267-277`). However, the paper
does not publish this composite or assert that these three terms should be
equal-weighted. The direction check passes for the proxy, not for fidelity to
the paper's learned detector.

### Prioritised fixes

1. **P0:** Decide whether this detector is meant to be NPR or a proxy. For NPR,
   implement non-overlapping `2x2` per-channel relationship maps and the
   trained classifier protocol. If the proxy is intentional, keep it clearly
   labeled as such and stop comparing it with the paper's accuracy.
2. **P0:** If the proxy remains, correct the missing reference element and the
   `12`-element variance calculation before calibrating it. The current numbers
   are not the stated set of values.
3. **P1:** Treat the `1024` downscale and all custom thresholds as an explicit
   operational variant and remeasure after any change. The paper gives no basis
   for these values.

## `aeroblade` — MAJOR-DRIFT

### Paper specification

Ricker et al. define reconstruction error for an LDM autoencoder as the distance
between an image and `D(E(x))`. The premise is exactly that latent-diffusion
images lie closer to the autoencoder manifold and therefore have lower error.
For practical detection, however, the paper computes the error for a set of
LDM autoencoders and uses the minimum reconstruction error across them. The
paper also reports that the second LPIPS layer captures the most meaningful
differences, and its main experiments use three AEs: Stable Diffusion 1, Stable
Diffusion 2, and Kandinsky 2.1. Its data protocol uses a `512x512` center crop
for real images and `512x512` generated images.

The paper's wording is: “use the smallest reconstruction error” across the AE
set, “the second LPIPS layer” is most meaningful, and real inputs use a “center
crop of size 512x512.”

### Our implementation

- `aeroblade.py:14-19` defines one TAESD path, one AlexNet LPIPS cache, defaults
  `threshold=0.05`, `scale=0.02`, and `MAX_ANALYSIS_SIDE=512`.
- `aeroblade.py:60-71` loads one `AutoencoderTiny`, encodes once, decodes once,
  and computes one LPIPS distance.
- `aeroblade.py:111-119` uses `AutoencoderTiny` and `lpips.LPIPS(net="alex")`.
- `aeroblade.py:135-146` preserves aspect ratio, resizes to a bounded
  multiple-of-eight shape, and normalizes to `[-1,1]`.
- The catalog openly calls it an “opt-in distilled TAESD + LPIPS approximation”
  at `plan/reference/detector-catalog.yaml:505-527`. Runtime calibration
  currently overrides the class defaults with threshold `0.0958355889`, scale
  `0.0214183377`, and `higher_is_worse: false` at
  `backend/app/analysis/calibration.json:3-13`.

### Deltas

1. **The core single-AE cue is present, but `Delta_Min` is omitted.** The paper
   evaluates a set of AEs and uses the minimum error. The code loads exactly one
   AE and returns its error (`aeroblade.py:60-71`, `:111-119`). This is a major
   omission because the minimum is the paper's mechanism for generalizing to
   unknown LDMs.

2. **The autoencoder is not the paper's AE set.** `AutoencoderTiny`/TAESD is a
   distilled approximation, not the Stable Diffusion 1, Stable Diffusion 2, and
   Kandinsky 2.1 AEs used in the paper's experiments. The catalog records this
   limitation, so it is an acknowledged approximation, but it still prevents a
   faithful paper claim.

3. **The LPIPS configuration differs.** The paper's best detector variant is
   single-layer `LPIPS_2`; the code requests standard `lpips.LPIPS(net="alex")`
   (`aeroblade.py:117-118`), which computes the standard AlexNet LPIPS rather
   than the paper's second-layer variant. The paper mainly uses VGG16 and finds
   AlexNet materially weaker in its alternative-metric table. The choice is a
   runtime approximation, not a paper constant.

4. **The thresholds are ours, not theirs.** The paper describes threshold-based
   detection but does not prescribe `0.05`, `0.02`, `0.0958355889`, or
   `0.0214183377`. The class defaults and calibration values are therefore
   repository calibration, not published method constants.

5. **The input protocol differs.** The code applies the shared longest-side
   `1600` cap (`base.py:77-87`) and then an aspect-preserving resize to at most
   `512` (`aeroblade.py:138-144`). The paper avoids resize distortion with a
   square `512x512` center crop. A wide or tall upload reaches the AE at a
   non-square shape, unlike the paper's evaluation protocol.

6. **The qualitative error map is omitted.** The paper presents local
   reconstruction-error maps for qualitative analysis and inpainting
   localization. This class declares `produces_map=False` and returns `None`
   (`aeroblade.py:31`, `:87`). That omission is not required for the scalar
   detection cue, but it means the published qualitative capability is absent.

### Signal direction

The direction is **correct**. The paper's physical premise is generated image
closer to its latent-diffusion AE manifold, hence lower reconstruction error.
The code sets `higher_is_worse=False` (`aeroblade.py:41`) and maps lower LPIPS
distance to higher suspicion. No direction flip is indicated.

### Prioritised fixes

1. **P0:** Do not present this as faithful AEROBLADE until it either computes
   `Delta_Min` over the paper's AE family and uses the paper-matched LPIPS
   variant, or is explicitly named an AEROBLADE-style TAESD approximation.
2. **P1:** Separate operational serving preprocessing from paper evaluation
   preprocessing. Re-run evidence with the paper's square crop or record the
   aspect-preserving route as a new variant.
3. **P1:** Keep the lower-error direction, but refit thresholds only after the
   AE and LPIPS variant are fixed. Existing calibration cannot repair a changed
   representation.

## `entropy` — UNVERIFIED

### Paper/source status

No paper was obtained for this detector. The catalog citation at
`plan/reference/detector-catalog.yaml:139` is a personal blog post dated
27 October 2024, not a paper or peer-reviewed publication. Under R1, this
detector is **UNVERIFIED** rather than being graded against the catalog's
second-hand summary.

The cited blog describes a much smaller procedure: load RGB channels, compute
local Shannon entropy with a circular disk (`radius=5`), compare the three raw
entropy maps with `tolerance=0.1`, and highlight matching pixels. It interprets
real images as having cohesive matching regions and AI images as having “many
small, scattered red areas.” It does not publish a scalar threshold, a
confidence mapping, extra uniformity/color masks, or a trained classifier.

### Our implementation and observed deltas against the cited blog

1. **Constants differ.** The blog code uses `radius=5` and `tolerance=0.1`
   (`blog` code). This implementation uses `radius=4`, `tolerance=0.12`, plus
   `matching_threshold=0.35`, `uniformity_threshold=0.2`, and
   `color_consistency_threshold=0.15` (`entropy.py:48-83`). The extra values are
   repository choices, not published constants.

2. **Normalization changes the statistic.** The blog compares the raw output of
   `filters.rank.entropy` with tolerance `0.1`. This code min-max normalizes
   each channel independently to `uint8` (`entropy.py:85-94`, `:209-212`) and
   compares with `0.12*255 = 30.6` (`:220-224`). That is not a unit conversion;
   per-channel min-max normalization changes the relative entropy values and
   the comparison scale on every image.

3. **The code adds three unpublished filters.** Local entropy uniformity,
   local color consistency, and morphological close/open are implemented at
   `entropy.py:96-143` and intersected with the matching mask at `:272-278`.
   The cited source only marks the cross-channel entropy-match mask. These
   additions may be reasonable experiments, but they are not the cited method.

4. **The published qualitative group cue is replaced by total area.** The blog
   distinguishes cohesive versus scattered matching regions. This code reduces
   the post-filter mask to `np.mean(suspicious_regions)` and compares it with a
   threshold (`entropy.py:278-281`). That discards connectedness and group size,
   the only qualitative discriminator stated by the source.

5. **There is an unsigned subtraction defect.** The normalized maps are
   `uint8` at `entropy.py:210-212`; subtraction at `:215-217` therefore wraps
   modulo 256 before `np.abs` is applied. A value pair such as `0` and `255`
   produces a difference of `1`, not `255`. This can mark strongly different
   entropy values as matching and is independent of the paper-verification
   status.

6. **The size cap is added and inconsistent by entry point.** The ndarray path
   downsizes to longest side `1024` at `entropy.py:19-31` and `:201-202`; the
   adapter also supplies the shared `1600`-bounded image at
   `backend/app/analysis/adapters.py:105-115` and `base.py:77-87`. Path/bytes
   calls do not use `_analysis_image`. The cited blog does not specify either
   cap, and local entropy is scale-dependent.

7. **The catalog and runtime thresholds disagree.** The catalog describes
   `matching_threshold (0.35)` at `plan/reference/detector-catalog.yaml:128`.
   Direct `EntropyAnalyzer` construction defaults to the legacy `0.35` at
   `entropy.py:43-65` and `calibration.json:484-487`, but the normal adapter
   passes the detector calibration value `0.6979472477` at
   `adapters.py:99-110` and `calibration.json:171-181`. The adapter also ignores
   the analyzer's returned boolean and recomputes the flag from the calibrated
   score (`adapters.py:108-115`). This is an internal configuration drift, not
   a published threshold.

### Signal direction

The direction is **broadly consistent with the cited blog's premise** if the
raw scalar is interpreted as the amount of matching entropy structure: fewer
matching/cohesive regions in generated images means lower matching proportion,
and `higher_is_worse=false` (`calibration.json:171-181`) maps lower raw values
to higher suspicion. But the source only gives a qualitative connected-region
interpretation, so it does not validate this scalar direction or its threshold.

### Prioritised fixes

1. **P0:** Obtain a real paper or explicitly reclassify this as a blog-derived
   heuristic. Until then, keep the detector UNVERIFIED and do not attribute a
   paper-level result to it.
2. **P0:** If the blog heuristic is retained, fix the `uint8` subtraction and
   choose whether to preserve the blog's raw entropy comparison or document the
   normalization as a new method. The current combination changes both the
   units and the decision mask.
3. **P1:** Remove or separately validate the extra uniformity/color/morphology
   masks and connected-component replacement. They are not part of the cited
   procedure.
4. **P1:** Reconcile the catalog's `0.35` claim with the adapter's
   `0.6979472477` runtime threshold before further calibration.

