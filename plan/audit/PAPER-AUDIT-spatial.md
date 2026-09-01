# Paper-Fidelity Audit: spatial family

Round: 17, read-only audit  
Date: 2026-09-01  
Scope: `backend/app/analysis/copy_move.py`, `cfa.py`, `ela.py`, their catalog entries, and the runtime adapter/context paths they use.

All three papers were obtained. No detector is `UNVERIFIED`.

## `copy_move.py` — MAJOR-DRIFT

Paper: I. Amerini et al., “A SIFT-Based Forensic Method for Copy–Move Attack Detection and Transformation Recovery,” IEEE TIFS 6(3), 2011. [Author manuscript](https://www.lambertoballan.net/downloads/2011_tifs_preprint.pdf), [DOI](https://doi.org/10.1109/TIFS.2011.2129512).

### Paper specification

The physical premise is correct in the catalog: a copy-moved patch is from the same image, so local descriptors can match even though the patch has been geometrically transformed. The paper describes three stages: SIFT extraction/matching; spatial clustering and forgery detection; and geometric-transformation estimation (paper pp. 4–6).

The published method is more specific than “SIFT plus RANSAC”:

1. For each descriptor, find its nearest neighbours among the other `n-1` descriptors. Its generalized 2NN test iterates `d_i/d_(i+1)` until the ratio exceeds `T`; the paper sets `T = 0.5` and retains all matches before the stopping point. The paper says the image is altered when it finds “two (or more) clusters with at least three pairs of matched points.”
2. Perform agglomerative hierarchical clustering on the spatial coordinates of matched points. The paper uses an inconsistency-coefficient cut; Ward linkage with `Th = 2.2` is its selected setting after training.
3. Estimate a 3×3 affine homography using RANSAC with three point pairs per trial, `Niter = 1000`, and `β = 0.05`. Before this, each point set is normalized to centroid zero and mean distance `sqrt(2)`. The homography is then refined by maximum-likelihood estimation and decomposed for rotation, anisotropic scale, and translation.

### Our implementation

- `copy_move.py:73-76` uses grayscale SIFT, which is the right feature family. `base.py:78-88` and the catalog at `detector-catalog.yaml:284-287` add a longest-side limit of `1600` pixels; the paper does not specify this bound.
- `copy_move.py:24-37` removes the self-match from `knnMatch(..., k=3)`, then keeps only one `best`/`second` pair and applies Lowe ratio `0.75`. This is ordinary 2NN, not Amerini’s generalized 2NN with multiple retained matches and `T = 0.5`.
- `copy_move.py:33-37` hashes offset vectors into an `8`-pixel grid. This is not the paper’s spatial agglomerative hierarchy or its inconsistency coefficient. `MIN_OFFSET = 32` at `copy_move.py:14` is also ours; no such value is specified in the paper.
- `copy_move.py:95-115` applies a candidate minimum of `3`, then requires at least `8` matches/inliers in one cluster. `copy_move.py:148-151` makes the raw statistic the number of verified clusters and flags from one verified cluster. The paper’s decision is at least two clusters, each linked by at least three matched pairs.
- `copy_move.py:109-115` calls `estimateAffinePartial2D(..., ransacReprojThreshold=3.0)`. This is a partial affine/similarity model in pixel units, not the paper’s full affine homography with normalized coordinates, `β = 0.05`, `Niter = 1000`, and maximum-likelihood refinement. The code does correctly return source/destination hulls and transformation metrics (`copy_move.py:126-153`), but they are derived from the narrower model.

### Deltas and priority

1. **P0 — matching is a different algorithm.** Replace the one-match `0.75` test at `copy_move.py:24-37` with the paper’s generalized 2NN procedure (`T = 0.5`, retaining multiple matches). This is the paper’s mechanism for surviving repeated/cloned features; the current form can discard the very matches copy-move creates.
2. **P0 — clustering and the positive decision are different.** `copy_move.py:33-37, 95-115, 148-151` uses offset hashing and “one cluster with eight inliers”; the paper uses spatial hierarchical clustering, `Th = 2.2` for Ward linkage, and two or more three-pair clusters. This changes both false-alarm control and the meaning of the score.
3. **P0 — transform estimation is narrower and differently calibrated.** `copy_move.py:109-115` omits point normalization, the paper’s `β = 0.05` and `Niter = 1000`, ML refinement, and anisotropic affine scale. A rotated or anisotropically scaled copy can therefore be rejected even though it is within the published method’s scope.
4. **P1 — added size/keypoint guards are unpapered variant choices.** The `1600`-side resize (`base.py:81-86`), `MIN_OFFSET = 32`, `MIN_KEYPOINTS = 100` (`copy_move.py:13-14, 78-83`), and eight-match cutoff are ours. They may be valid cost/robustness policy, but must be labeled as a repository variant rather than paper constants. The distinct insufficient-texture outcome is honest and agrees with the paper’s stated SIFT limitation.

### Signal direction

The premise is “copy-move present → multiple geometrically consistent descriptor matches.” Our statistic is `len(verified)` (`copy_move.py:148`) and the catalog/runtime use higher-is-worse (`detector-catalog.yaml:306`, `calibration.json:1`). That direction is physically consistent: more verified geometric clusters is more copy-move evidence. There is no sign inversion here. The catalog’s `>=1 verified affine cluster` decision, however, is not the paper’s two-cluster decision and should not be mistaken for a direction finding.

## `cfa.py` — MAJOR-DRIFT

Catalog sources: A.C. Popescu and H. Farid, “Exposing Digital Forgeries in Color Filter Array Interpolated Images,” IEEE TSP 53(10), 2005, [open paper](https://farid.berkeley.edu/downloads/publications/sp05a.pdf); and P. Ferrara et al., “Image Forgery Localization via Fine-Grained Analysis of CFA Artifacts,” IEEE TIFS 7(5), 2012, [open paper](https://iris.polito.it/retrieve/handle/11583/2505936/e384c42e-24cc-d4b2-e053-9f05fe0a1d67/ferr_TIFS12_OA.pdf). The module itself names a different source, Bammey et al., [IPOL 2021 paper](https://www.ipol.im/pub/art/2021/355/revisions/2022-01-01/article.pdf).

### Paper specification

There is a citation-to-code mismatch before the numerical comparison:

- Popescu–Farid models each channel as a linear neighbourhood correlation and uses expectation/maximization to estimate both the correlation coefficients and the per-sample probability. It then exposes the periodic structure in the probability map, including its Fourier peaks (2005 paper pp. 7–10). The paper says: “To simultaneously estimate both we employ the expectation/maximization (EM) algorithm.”
- Ferrara et al. use the green channel, a predictor, locally weighted prediction-error variance, and the feature `L = log(GMA/GMI)`. They fit a two-component Gaussian mixture with EM and produce a Bayesian tampering-probability map for each `B×B` block (2012 paper pp. 4–6). Their output is not a dominant-pattern disagreement ratio.
- Bammey’s IPOL method, which is what the module actually resembles, counts intermediate values for the four Bayer phases. It first detects the global diagonal/pattern, then repeats the test in overlapping windows. Its bidirectional mask averages horizontal and vertical intermediate-value masks and masks a “2-pixels-wide border.” It normalizes count differences by `1/(2XY)`, groups inconsistent windows into connected components, uses the maximum absolute count difference per component as confidence, merges full-pattern and diagonal maps, and thresholds with hysteresis `γ` (IPOL paper pp. 3–9).

Reported Bammey settings are not universal constants, but they provide concrete comparison points: the paper’s Table 2 defaults are bidirectional masks, continuous normalization at `γ = 0.2`, and `32×32` windows; it also reports `64×64` windows and `γ = 0.1`. The paper explicitly says window size and stride are algorithm parameters.

### Our implementation

- `cfa.py:133-143` does implement the Bammey-style bidirectional intermediate-value mask and `cfa.py:146-178` correctly follows the broad green-diagonal then red/blue phase-selection structure. This is the part that is faithful to Bammey, not to either paper named in the catalog.
- `cfa.py:139-142` writes a one-pixel border and values `0`, `0.5`, or `1`. Bammey explicitly masks two pixels on each side for balance. The code therefore includes a border population the paper excludes.
- `cfa.py:157-169` divides the count differences by `2 * blocks * 255.0`. The masks just created are unit-valued masks, not 0–255 image samples, so the paper’s `1/(2XY)` normalization has an extra `/255` in this implementation. This is consistent with the repository’s tiny calibrated threshold (`calibration.json:1`, `cfa.threshold ≈ 0.000236`) versus the paper’s confidence scale, which lies in `[-1, 1]` and is typically below `0.3`.
- `cfa.py:104-126` uses `window = 128`, `stride = 64`, marks every locally different pattern, and reports the mean of all inconsistent-window values. It does not perform connected-component segmentation, per-component maxima, separate diagonal/full confidence maps, or the paper’s `γ` thresholding.
- `cfa.py:100-102` returns ratio `0.0` and an empty map when no dominant pattern is found. `cfa.py:66-80` then returns an applicable, low-suspicion result. Bammey’s algorithm makes no grid decision on an unresolved equality (`IPOL paper pp. 5–7`); the catalog itself says “no dominant pattern → no CFA conclusion” (`detector-catalog.yaml:330-335`). Returning an applicable zero is stronger than abstaining.
- `cfa.py:36-47` adds a strict JPEG plus Make/Model plus exact EXIF-dimension gate. Neither catalog paper requires EXIF metadata, and Ferrara’s experiments use original TIFF images (`2012 paper p. 6`). This is an operational provenance guard, not a published-method step.
- The round-16 `1024` bound is not applied here: `cfa.py:66` passes `ctx.rgb_uint8`, not `ctx.downscaled_rgb_uint8`. That preserves full decoded dimensions, although it does not solve the papers’ sensitivity to resampling or JPEG loss.

### Deltas and priority

1. **P0 — resolve the citation/method identity.** Either implement the Popescu–Farid/Ferrara methods named at `detector-catalog.yaml:341-345`, or change the catalog to cite Bammey and label this as the intermediate-values variant. These are related CFA cues, not interchangeable implementations.
2. **P0 — remove the extra `/255`.** At `cfa.py:159-169`, the paper normalization is `1/(2XY)` for the unit masks. The extra factor changes the statistic by 255× and makes the calibrated threshold incomparable with the published confidence scale.
3. **P1 — restore Bammey’s localization/confidence path if Bammey is the intended source.** Add the two-pixel border, connected components, component-maximum confidence, separate diagonal/full maps, and an explicit `γ`; the current global mean at `cfa.py:124-126` is not the published decision procedure.
4. **P1 — abstain on unresolved global pattern.** `cfa.py:100-102, 66-80` should not turn “no pattern detected” into an applicable clean score. The paper’s premise only supports higher suspicion when a local pattern is inconsistent with a detected global pattern.
5. **P1 — document the gate as product policy.** JPEG-only and exact EXIF dimensions can be conservative safeguards, but they narrow the published scope. Bammey reports that even JPEG quality 95 can make the algorithm unable to detect anything (IPOL paper pp. 13–14), so strict JPEG applicability puts the detector in a known weak regime.

### Signal direction

The physical premise is: demosaicing makes interpolated pixels more likely to be intermediate values, and a forged/spliced region can have a different Bayer phase. Bammey’s confidence is the absolute count difference for a locally inconsistent pattern; higher inconsistency is more suspicious. The code’s `cfa_ratio` is monotone in that same broad evidence, and `higher_is_worse = true` is therefore directionally correct (`cfa.py:67-70`, `detector-catalog.yaml:335`). The extra `/255` only rescales it. The no-dominant-pattern path is not a sign inversion, but it violates the paper/catalog requirement to make no CFA conclusion.

## `ela.py` — MAJOR-DRIFT

Paper: N. Krawetz, “A Picture’s Worth: Digital Image Analysis and Forensics.” The open whitepaper is [Black Hat USA 2007](https://www.blackhat.com/presentations/bh-usa-07/Krawetz/Whitepaper/bh-usa-07-krawetz-WP.pdf). The catalog/module cite Black Hat DC 2008 (`detector-catalog.yaml:83`, `ela.py:4-7`), while the fetched whitepaper itself is labeled “presented at Black Hat Briefings USA 2007”; the date/citation should be corrected or explicitly explained as a later presentation.

### Paper specification

Krawetz’s ELA premise is JPEG’s block-local lossy history. The method intentionally resaves the input at a known quality, “such as 95%,” and computes the pixel difference between the input and that resaved image. A large change means the pixels are not at their local minimum and are effectively “original”; recent modifications can make previously stable 8×8 cells unstable. The paper presents this as a visual forensic aid, not a fixed numeric classifier.

The cited ELA method therefore requires one comparison:

`decoded input JPEG` → `decoded copy resaved at q=95` → absolute pixel difference.

The paper also warns that repeated resaves and low-quality JPEGs can obscure the result. It does not specify Canny edge scores, texture/noise votes, a two-of-three rule, or numeric thresholds for those features.

### Our implementation

- `ela.py:53-64` converts the input and resizes its longest side to `1024` with Lanczos before analysis. This is not in the paper and changes the JPEG block grid and pixel values whose history ELA is meant to inspect.
- `ela.py:101-115` first saves the decoded/resized image at `quality=95`, then saves that synthetic q95 decode again at `quality=75`. `ela.py:121-129` compares q95 against q75. It does not compare the input JPEG against a q95 resave. The repository’s `resave_quality=75` (`ela.py:33-51`) is therefore not the paper’s ELA operation; q75 is being used as a second synthetic history.
- `ela.py:122-127` globally rescales the residual to make its maximum 255. The paper defines the difference image; this adaptive max scaling is an unpapered normalization that makes scores dependent on the single largest pixel difference.
- `ela.py:134-211` replaces the paper’s direct ELA interpretation with Canny/dilation edge-gap, local texture variance, median-filter noise, and 8×8-boundary statistics. `ela.py:213-272` adds thresholds `0.45`, `2000`, `25`, `100`, combines violations, and declares tampering at two violations. None of these steps or constants is in Krawetz’s ELA description.
- The runtime does not even use that vote as its detector statistic: `adapters.py:46-53` calls `detect_tampering` but maps only `features.edge_discontinuity` into the calibrated score. The current `uint8` arithmetic is not the catalog’s recorded defect: `ela.py:121` uses `cv2.absdiff`, while `detector-catalog.yaml:59-80` still describes the old `np.abs(high_array - low_array)` wraparound. The catalog note is stale as of this read; re-fixing that arithmetic would be incorrect.
- The JPEG-only runtime gate at `adapters.py:22-37` is consistent with ELA’s meaningful input domain. The direct legacy `ELAAnalyzer` API can still accept PNG or other inputs and manufacture a JPEG history (`ela.py:73-106`), but that path is not used by the adapter for non-JPEG uploads.

### Deltas and priority

1. **P0 — compare the correct two images.** At `ela.py:101-129`, compare the decoded input JPEG with one q95 resave. The current q95→q75 comparison measures a different compression response and cannot support paper-fidelity claims.
2. **P0 — separate raw ELA from repository heuristics.** The Canny/texture/noise/block feature stack and two-of-three vote (`ela.py:134-272`) are a custom classifier, not Krawetz ELA. Preserve them only as an explicitly named heuristic variant; the paper-faithful detector should expose the residual map and a documented spatial contrast statistic.
3. **P1 — remove or justify the 1024 Lanczos preprocessing.** `ela.py:21, 53-64` can destroy or move 8×8-local evidence. If a cost bound is retained, it needs an explicit validation showing that the chosen resize preserves the signal; it is not a paper constant.
4. **P1 — repair catalog bookkeeping.** Update the Krawetz venue/date and remove the stale ELA-1 description at `detector-catalog.yaml:59-80`, or record the defect as fixed. Do not treat the currently correct `cv2.absdiff` line as still broken.

### Signal direction

For Krawetz’s paper-defined residual, higher local difference in an otherwise stable JPEG region is consistent with recent modification: the paper says a large change means the pixels are not at their local minima. Thus “higher ELA residual → more suspicious” is directionally sound. The runtime statistic, however, is edge-gap ratio (`adapters.py:47-53`), not the paper’s residual magnitude. Krawetz does not define or validate that statistic, so `higher_is_worse = true` is plausible but not paper-grounded; there is no demonstrated sign inversion, only a statistic substitution.

## Priority summary

1. **Copy-move:** restore generalized 2NN, spatial hierarchical clustering/two-cluster decision, and normalized full-affine RANSAC. These are core algorithm substitutions.
2. **CFA:** resolve the catalog citation mismatch and remove the extra `/255`; then restore published confidence/localization semantics or explicitly label the implementation as a simplified Bammey variant.
3. **ELA:** fix the input-vs-q95 comparison before tuning any thresholds. Current calibration is attached to a different statistic, so performance cannot be interpreted as evidence for or against the cited method.
