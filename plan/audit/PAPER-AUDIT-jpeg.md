# Paper-fidelity audit: JPEG family

Date: 2026-09-01
Scope: `double_jpeg.py`, `ghosts.py`, `zero.py`, and `qtable.py`. This is a
read-only audit of the implementations and catalog; no detector code was
changed.

Grades describe fidelity to the cited method, not how well a detector happens
to score on the current corpus:

- **FAITHFUL** — the published method and direction are substantially preserved.
- **MINOR-DRIFT** — a bounded, documented deviation that does not replace the
  method's core signal.
- **MAJOR-DRIFT** — a core signal, statistical test, required branch, or
  direction is replaced or omitted.
- **UNVERIFIED** — the actual paper could not be obtained. None of these four
  detectors is UNVERIFIED.

## Sources

- D. Fu, Y. Q. Shi, and W. Su, [“A generalized Benford's law for JPEG
  coefficients and its applications in image forensics”](https://doi.org/10.1117/12.704723),
  SPIE 6505 (2007). The official NJIT record is bibliographic and marks the
  full text unavailable; the paper text was consulted through this
  [full-text mirror](https://paperzz.com/doc/6987028/a-generalized-benford-s-law-for-jpeg-coefficients-and-its).
- T. Bianchi and A. Piva, [“Image Forgery Localization via Block-Grained
  Analysis of JPEG Artifacts”](https://iris.polito.it/retrieve/e384c42e-2465-d4b2-e053-9f05fe0a1d67/bian_TIFS2012_OA.pdf),
  IEEE TIFS 7(3) (2012).
- H. Farid, [“Exposing Digital Forgeries from JPEG Ghosts”](https://farid.berkeley.edu/downloads/publications/tifs09.pdf),
  IEEE TIFS 4(1) (2009). The paper's experimental details were checked against
  the [accessible text transcription](https://www.readkong.com/page/exposing-digital-forgeries-from-jpeg-ghosts-5798702).
- H. Farid, [“Digital Image Ballistics from JPEG Quantization”](https://farid.berkeley.edu/downloads/publications/tr06a.pdf),
  Dartmouth TR2006-583 (2006).
- Nikoukhah et al., [“ZERO: a Local JPEG Grid Origin Detector Based on the
  Number of DCT Zeros and its Applications in Image Forensics”](https://www.ipol.im/pub/art/2021/390/),
  IPOL 11 (2021), [paper PDF](https://www.ipol.im/pub/art/2021/390/article_lr.pdf).

## `double_jpeg.py` — **MAJOR-DRIFT**

### Paper specification

Fu et al. study the first digits of **JPEG quantized block-DCT
coefficients**. Their generalized law is fitted to the coefficient
distribution, and they state that “this law will be violated if the image is
double JPEG compressed by using different Q-factors.” Their experiments use
8-bit luminance images, quality factors 100, 90, 80, 70, 60, and 50, and
report fitted `s` and `q` values rather than a universal fixed parameter
grid. They also assume the 8x8 grid origin is known.

Bianchi and Piva's cited localization method is a different, fuller method:
it has separate aligned and non-aligned double-JPEG models, estimates the
primary quantizer with EM, estimates a spatial shift for the non-aligned case,
and produces a likelihood ratio for every 8x8 block. The paper describes the
output as “a likelihood map indicating the probability for each 8 × 8
discrete cosine transform block of being doubly compressed.”

### Our implementation

- `backend/app/analysis/double_jpeg.py:25-34` converts the decoded RGB image
  to Y, crops to complete 8x8 blocks, and recomputes an unquantized OpenCV
  DCT. It does not read the JPEG's quantized DCT coefficients.
- `backend/app/analysis/double_jpeg.py:45-61` discards zero coefficients,
  takes leading digits, fits the generalized model by a local grid, and emits
  a chi-square-like divergence.
- `backend/app/analysis/double_jpeg.py:64-68` adds a histogram FFT peak ratio.
- `backend/app/analysis/double_jpeg.py:71-78` emits an energy visualization,
  not Bianchi–Piva's per-block double-JPEG likelihood map.
- `backend/app/analysis/double_jpeg.py:90-95` requires JPEG and 256px minimum
  dimensions. `:103-125` averages the two statistics and maps the result.

### Deltas and direction

1. **Core coefficient domain is changed (P0).** Fu et al.'s target is the
   quantized coefficients in the JPEG stream; `:25-34` recomputes DCT values
   from already decoded pixels. This can retain a related artifact, but it is
   not the published JPEG-coefficient test. If the implementation is intended
   to remain a decoded-image heuristic, it should not be described as a direct
   implementation of Fu et al.
2. **The Bianchi–Piva method is not implemented (P0).** There is no EM
   estimate of Q1, Q2-conditioned likelihood model, non-aligned shift search,
   or per-block likelihood ratio. The FFT ratio at `:64-68` is a related
   periodicity heuristic, not that paper's statistical detector.
3. **The fitted-parameter search contains constants that are ours (P1).**
   `:53-54` restricts `s` to 0.0..2.0 and `q` to 0.5..2.0. Fu et al.'s table
   reports negative `s` values for QF 90 through 50 (and image/quality-specific
   fitted `q` values); the paper does not specify this non-negative 9x7 grid.
   This restriction can make a valid generalized fit impossible.
4. **The aggregate direction is inverted (P0).** The physical premise is:
   different-quality recompression creates a larger departure from the
   generalized law and stronger periodic structure, so each raw component's
   increase means more double compression. The catalog says exactly that at
   `plan/reference/detector-catalog.yaml:241`. But `:108-112` negates the
   aggregate while still passing `higher_is_worse` from calibration. Thus a
   larger paper-style indicator produces a smaller raw aggregate and a lower
   confidence score. The comment explicitly says the inversion was made from
   corpus correlation; corpus behavior cannot change the paper's signal
   direction.
5. **Preprocessing is partly compatible but not neutral (P1).** Luminance and
   8x8 blocking are consistent with the papers' luminance/block setup, and
   cropping complete blocks is a reasonable boundary choice. Recomputing the
   DCT after decode, however, loses the quantization-domain premise; the code
   also fixes the grid origin at the crop origin and has no non-aligned branch.

### Prioritized fixes

1. Correct the direction only after preserving the raw components: the current
   inversion is a silent false-negative path.
2. Decide whether this detector is a decoded-DCT heuristic or a paper-faithful
   JPEG-coefficient detector. For the latter, extract/use the quantized stream
   coefficients and preserve the generalized-fit domain.
3. If Bianchi–Piva remains a citation, implement its aligned/non-aligned
   likelihood and Q1 estimation, or remove that citation and call the FFT/map
   output a heuristic. Do not use the present aggregate as evidence that the
   cited likelihood method was reproduced.

## `ghosts.py` — **MAJOR-DRIFT**

### Paper specification

Farid's method repeatedly recompresses the image, compares the original and
recompressed pixels, spatially averages the error over `b x b` neighborhoods
(the experiments use `b=16`), normalizes the quality curve, and uses a
two-sample Kolmogorov–Smirnov statistic to compare candidate regions with the
rest of the image. A ghost is a local minimum at the earlier quality. The
paper states that the method is “only effective when the tampered region is of
lower quality than the image into which it was inserted.” For a misaligned
JPEG grid it tests all 64 horizontal/vertical alignments. Its experiments use
quality sweeps including q2=30..90 at step 1, and explicitly ignore blocks
whose average intensity variance is below 2.5 gray values.

### Our implementation

- `backend/app/analysis/ghosts.py:12-22` sweeps q=50..100 by 2 and adds a
  longest-side 1024px `INTER_AREA` resize.
- `:67-80` recompresses with OpenCV, calculates RGB mean-squared error, and
  averages it over disjoint 16x16 blocks.
- `:81-86` min-max normalizes each block's curve and takes its argmin.
- `:25-41` groups q0 values, applies a 3x3 close, connected components, a 1%
  area floor, and a quality separation of 4; `:86-98` reports the number of
  coherent modes and uses that as the raw score.
- The catalog records the same choices at
  `plan/reference/detector-catalog.yaml:255-270`.

### Deltas and direction

1. **The decision statistic is replaced (P0).** The paper's region-vs-rest
   K–S comparison and its false-positive threshold are absent. `:25-41` turns
   the q0 map into a new connected-component/mode-count heuristic. The
   min-max and argmin steps at `:81-86` are faithful building blocks, but they
   do not make the final detector Farid-faithful without the K–S test.
2. **All 64 alignment shifts are omitted (P0).** Farid explicitly tests the
   eight-by-eight spatial offsets when the tampered JPEG grid is not aligned.
   `:67-86` evaluates only the current pixel grid, so a valid misaligned ghost
   can be missed.
3. **The sweep is an implementation choice, not the paper's reported main
   sweep (P1).** `:12` uses 50..100 step 2. Farid's main experiment reports
   q2=30..90 step 1, with other illustrative sweeps such as 60..100 step 2;
   there is no single universal paper range. The current range therefore
   excludes lower qualities and odd quality values used in the experiment.
4. **The low-variance exclusion is omitted (P1).** The paper excludes regions
   below 2.5 gray-value average intensity variance; the code processes every
   block. This increases content-dependent false minima in flat areas.
5. **The 1024px resize is extra preprocessing (P1).** `:16-22` performs a
   lossy scale operation not specified by the paper. It changes the physical
   size of a 16px neighborhood and can alter the block-grid history that the
   ghost signal depends on. It is a runtime policy, not a paper step.

The direction is only conditionally consistent. Under Farid's stated premise,
a lower-quality inserted region can create a second coherent q0 minimum, so a
higher mode count is a reasonable *heuristic* risk direction and matches the
catalog at `:267`. But it is not a general “any splice” direction: the paper
also says the method is ineffective when that lower-quality condition does not
hold. The code's `higher_is_worse` mapping at `:87-89` is therefore acceptable
for the narrowed heuristic, not proof of the full paper method.

### Prioritized fixes

1. Restore alignment search and the paper's K–S/region comparison before
   treating this as a faithful ghost detector.
2. Add the paper's low-variance exclusion and make the tested quality range a
   documented scope decision; do not silently present 50..100 step 2 as the
   paper's constant.
3. Measure the effect of the 1024px cap on ghost visibility. If it remains a
   product constraint, document the detector as a resized heuristic and
   calibrate it separately from the paper claim.

## `zero.py` — **MAJOR-DRIFT**

### Catalog status

The catalog has `zero` in the variant policy at
`plan/reference/detector-catalog.yaml:41`, and a measurement at `:611`, but no
`zero` detector specification under `new:` (which begins at `:168`). This is a
catalog omission in addition to the implementation drift below.

### Paper specification

ZERO rounds the luminance image, evaluates DCT zeros in overlapping 8x8 blocks
for all 64 grid origins, lets each pixel vote for the origin with the most
zeros, and uses binomial a-contrario NFA tests for global and local evidence.
Ties and the seven-pixel border are invalid. The paper states, “A JPEG grid is
detected when NFA < 1.” It also excludes blocks constant along a horizontal or
vertical direction. Local foreign-grid detection uses region growing; missing
grids are detected by a second vote pass after JPEG-compressing the input at
QF=99. The paper does not use a coarse 32px cell or a four-sample-per-cell
approximation.

### Our implementation

- `backend/app/analysis/zero.py:35-39` computes rounded luminance, matching the
  paper's luminance-only setup.
- `:26-30` defines 8x8 blocks, 64 phases, a 9px neighborhood, but also adds
  `CELL_SIZE=32` and `SAMPLE_OFFSETS=(8,24)`.
- `:41-87` tests four sampled blocks per 32px cell for each phase, sums their
  zero counts, and selects one winning phase per cell.
- `:90-121` applies a binomial tail to those coarse-cell winners; `:124-156`
  derives foreign regions with dilation, connected components, and closing.
- `:177-196` uses `ctx.downscaled_rgb_uint8`; the shared context caps the
  longest side at 1600 in `backend/app/analysis/base.py:78-88`.
- `:208-237` maps `-log10(NFA)` through a higher-is-worse logistic score.

### Deltas and direction

1. **The vote domain is changed (P0).** The paper evaluates every valid pixel's
   overlapping 8x8 blocks and 64 phases. `:41-87` evaluates only four sampled
   blocks per 32px cell and then votes once per cell. `CELL_SIZE`, the two
   offsets, the summed count, and the clamping at `:58-64` are ours. This is a
   core algorithm reduction, not a paper constant substitution.
2. **The NFA sample model no longer matches the paper (P0).** `:101-111`
   treats coarse-cell winners as the support and uses `votes.shape` in the
   multiplicity term; `:114-121` counts those cell winners globally. The
   paper's NFA is based on the spatially sampled pixel votes (or its stated
   conservative /64 reduction), not a new winner-per-cell population. The
   comment at `:106-107` confirms that the paper's block-to-grid reduction is
   intentionally not repeated, but it does not establish that the altered
   population remains a valid NFA model.
3. **Required branches are missing (P0).** `_foreign_regions` returns no
   evidence when there is no dominant grid (`:124-156`), so the implementation
   cannot handle the paper's compressed foreign region in an otherwise
   uncompressed image. It also has no QF=99 second pass for missing-grid
   detection, which the paper presents as a separate application path.
4. **Validity rules differ (P1).** The paper invalidates the seven-pixel
   border; `:58-64` clamps origins to the image boundary. The paper excludes
   blocks constant along either horizontal or vertical direction; `:75-80`
   checks only overall block standard deviation equal to zero.
5. **The 1600px downscale is extra preprocessing (P1).** `zero.py:193-196`
   selects the shared resized image, whereas the paper's grid phase is defined
   on the input pixel lattice. Resizing can change phase evidence and spatial
   support. It may be necessary for application latency, but it is not a
   faithful paper preprocessing step and needs separate calibration.

The direction of the implemented foreign-grid evidence is correct: the paper
uses NFA < 1 as evidence, and `:201-208` makes lower log10 NFA produce larger
`-log10(NFA)` evidence with `higher_is_worse=True`. There is no direction
inversion here. The missing-grid path and altered NFA population are separate
coverage/fidelity failures.

### Prioritized fixes

1. Restore the paper's pixel-level 64-phase vote and recompute the NFA using
   the corresponding support/multiplicity model; otherwise the reported NFA is
   not the paper's significance value.
2. Implement the missing-grid QF=99 pass and the no-global-grid local foreign
   case, or explicitly narrow the detector claim to foreign grids inside a
   detected dominant grid.
3. Match the border and horizontal/vertical-constant validity rules. Treat the
   1600px resize as a separately validated runtime variant.
4. Add the missing ZERO catalog entry so method, preprocessing, direction, and
   supported branches are reviewable in future rounds.

## `qtable.py` — **MAJOR-DRIFT**

### Paper specification

Farid's JPEG-ballistics paper extracts the image's quantization tables and
compares them against a database of known camera and software tables. It
reports that different sources can share a table, so a match narrows or
supports provenance rather than proving a unique encoder. The paper's premise
is that “a signature of sorts is embedded within each JPEG image.” It does not
define “distance to the two Annex-K tables equals re-save,” require EXIF
Make/Model, or prescribe a quality 1..100 minimization.

### Our implementation

- `backend/app/analysis/qtable.py:15-27` hard-codes two Annex-K-style 64-value
  tables.
- `:30-48` reads `PIL.Image.quantization`, applies the libjpeg quality scaling
  formula, searches qualities 1..100, and chooses the minimum L1 distance.
- `:68-74` makes EXIF Make and Model a hard applicability gate.
- `:87-114` reports the summed distance and a SHA-256 fingerprint; exact zero
  distance is described at `:102-109` as a standard-table match.
- The catalog specifies this narrower heuristic at
  `plan/reference/detector-catalog.yaml:176-204`, including the explicit
  decision that exact standard-table distance plus camera EXIF is a re-save
  indicator.

### Deltas and direction

1. **The cited method's core comparison is omitted (P0).** There is no corpus of
   camera/software tables and no source-identification comparison. The code
   only asks whether the table is close to two standard baselines. That is a
   useful provenance heuristic, but it cannot reproduce Farid's database-based
   ballistics method; exact standard tables are not unique proof of generic
   software.
2. **The quality sweep and distance are ours (P1).** Farid's paper reports
   extracting and comparing table values; it does not specify the `1..100`
   search at `:44-47`, the quality scaling implementation at `:35-37`, or
   summed L1 distance as the forensic statistic. These constants must be
   labeled as an engineering heuristic, not paper parameters.
3. **The EXIF gate is extra preprocessing/policy (P1).** `:68-74` skips a
   JPEG without Make/Model. The paper's table comparison does not require that
   metadata. This gate may be a defensible safety policy because standard
   tables alone are weak evidence, but it reduces coverage and is not a
   published method step. The existing catalog correctly lists it as a
   limitation, not as a fact about JPEG tables.
4. **Signal direction is only valid for the catalog's narrow claim (P1).**
   `:99-100` uses `higher_is_worse=False`, so lower distance to the standard
   table yields higher suspicion. That is coherent with the catalog's special
   premise “camera EXIF plus exact generic table,” but Farid's paper has no
   monotone suspicious-distance rule: matching a table can identify or narrow
   a source, and shared tables are expected. Poor corpus AUC is therefore not
   evidence to invert this direction; the provenance hypothesis and table
   database need to be fixed first.
5. **Table-order handling needs an explicit check (P2).** The constants at
   `:15-27` are written in natural 8x8 layout while the comment says Pillow
   exposes zig-zag order. Pillow's JPEG parser performs an order conversion.
   The existing tests' exact-zero assertion is not enough evidence that both
   orders are aligned for every Pillow/libjpeg version. Verify the conversion
   against a real decoded DQT before relying on `libjpeg_distance`; this is a
   correctness risk, but not the grade-defining drift above.

The raw-table path is otherwise good preprocessing: `:30-33` reads the JPEG
bytes directly and does not introduce a pixel resize. The problem is the
interpretation of the table, not the absence of a required image transform.

### Prioritized fixes

1. Either add the cited camera/software table corpus and compare fingerprints
   as provenance evidence, or rename/document this detector as a standard-
   libjpeg-distance heuristic rather than a Farid implementation.
2. Keep the EXIF gate only as an explicit product safety policy, not as a
   paper requirement, and keep its score separate from a source-table match.
3. Verify Pillow DQT ordering with a real JPEG fixture before trusting exact
   zero distance; then calibrate the narrower heuristic on provenance-labeled
   data.

## Overall priority

1. **Fix the double-JPEG score inversion** (`double_jpeg.py:108-112` versus
   catalog `:241`); it silently converts stronger raw evidence into weaker
   suspicion.
2. **Do not report paper-faithful confidence for the other three until their
   core tests are named honestly:** ghost mode count is not Farid's K–S test,
   ZERO's coarse-cell NFA is not its pixel-level a-contrario test, and qtable's
   standard-table distance is not Farid's source-table database comparison.
3. **Separate runtime variants from paper methods:** the 1024px ghost resize
   and 1600px ZERO resize need their own validation/calibration because both
   can alter the spatial compression evidence.
4. **Add the missing ZERO catalog entry** so its method, branches, direction,
   and preprocessing cannot drift silently again.
