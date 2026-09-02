# Round 19 repair report — spatial family

Date: 2026-09-01

Scope: `copy_move.py`, `cfa.py`, `ela.py`, their owned tests, and their owned
catalog entries. No calibration artifact, data file, shared script, or adapter
was changed.

## Result summary

| detector | audit grade | grade after repair | resolution |
|---|---|---|---|
| `copy_move` | MAJOR-DRIFT | MINOR-DRIFT | (a), with documented repository guards |
| `cfa` | MAJOR-DRIFT | MINOR-DRIFT against Bammey; old citation claim corrected | (a) for Bammey; (b) for citation mismatch |
| `ela` | MAJOR-DRIFT | MAJOR-DRIFT for the production detector; raw analyzer repaired | (a) for residual; (b) for production heuristic claim |

ELA remains major against Krawetz because the service adapter still scores the
legacy `edge_discontinuity`, not the paper-defined residual.

## Verification commands and real outputs

```text
$ .venv/bin/python -m pytest backend/tests/test_copy_move.py backend/tests/test_cfa.py backend/tests/test_ela.py -q
.............................                                            [100%]
29 passed in 2.07s

$ .venv/bin/python -m pytest backend/tests -q
.................................................................. [ 53%]
..........................................................           [100%]
134 passed, 1 warning in 94.61s (0:01:34)

$ .venv/bin/python -c "import yaml,sys; yaml.safe_load(open('plan/reference/detector-catalog.yaml'))"
[exit 0, no stdout]
```

The before/after measurement command was a `.venv/bin/python` heredoc probe
using deterministic seeds 10, 21, and 22. The copy probe used a 512x512 random
image and copied `[64:160,64:160]` to `[300:396,300:396]`; the CFA probe used a
256x256 random RGB array; the ELA probe used a 64x64 JPEG encoded at quality
75. The pre-edit leg loaded the `HEAD` modules with read-only `git show`; the
after leg imported the working-tree modules.

Copy-move probe, before:

```text
constructed_positive: state applicable, score 0.7310585786300049, flagged True, verified_clusters 2.0
constructed_negative: state not_applicable, score None, flagged None, verified_clusters 0.0
reference_forgery: state not_applicable, score None, flagged None, verified_clusters 0.0, surviving_matches 0.0
```

Copy-move probe, after:

```text
constructed_positive: state applicable, score 0.9990889488055994, flagged True, verified_clusters 8.0, verified_cluster_pairs 8.0
constructed_negative: state applicable, score 0.9525741268224334, flagged True, verified_clusters 4.0, verified_cluster_pairs 4.0
reference_forgery: state applicable, score 0.7310585786300049, flagged True, verified_clusters 2.0, verified_cluster_pairs 2.0, inlier_count 3.0, translation_dx -7148.747749711866
```

The supplied repository instrumentation says both pasted regions in the
reference forgery contain zero SIFT keypoints. The after result is therefore a
false positive from unrelated three-point matches, not localization evidence.

CFA probe, before and after:

```text
before {'ratio': 2.384559820711729e-05, 'phase': 3, 'map_shape': (256, 256), 'map_max': 2.8243719498277642e-05}
after  {'ratio': 0.09311224520206451, 'phase': 3, 'map_shape': (256, 256), 'map_max': 0.09311224520206451}
```

ELA probe, before and after:

```text
before {'input_shape': (64, 64, 3), 'residual_mean': 45.201985677083336, 'residual_max': 255}
after  {'input_shape': (64, 64, 3), 'residual_mean': 6.751302083333333, 'residual_max': 43, 'matches_input_to_q95': True}
```

## 1. `copy_move`

### Paper specification

Paper: I. Amerini et al., “A SIFT-Based Forensic Method for Copy–Move Attack
Detection and Transformation Recovery,” IEEE TIFS 6(3), 2011; [author
manuscript](https://www.lambertoballan.net/downloads/2011_tifs_preprint.pdf).

The paper says generalized 2NN iterates “between `d_i/d_(i+1)` until this
ratio is greater than `T`”; experiments set `T=0.5` and retain all matches
before the stop. It hierarchically clusters matched point coordinates and uses
“two (or more) clusters with at least three pairs of matched points linking a
cluster to another one.” Transform estimation uses three-point RANSAC,
`Niter=1000`, `beta=0.05`, centroid/mean-distance `sqrt(2)` normalization, and
maximum-likelihood homography estimation.

### Changes and resolutions

- Audit P0, ordinary Lowe 2NN: **(a)**. `copy_move.py:22-54` now searches all
  other descriptors, applies the generalized ratio at `0.5`, and retains the
  prefix. Mutual filtering at `:53-54` is retained as a bounded isolation guard
  and disclosed at catalog `:305-309`.
- Audit P0, offset hashing and one-cluster/eight-match decision: **(a)**.
  `copy_move.py:57-87` clusters the union of matched keypoint locations with
  Ward/inconsistency `Th=2.2` and keeps links between distinct clusters.
  `MIN_CLUSTER_MATCHES=3` at `:14-19` follows the paper's explicit rule; the
  paper also says “more than three” in the preceding sentence, so that textual
  ambiguity is recorded rather than hidden.
- Audit P0, partial affine model and pixel-unit threshold: **(a)**.
  `copy_move.py:90-133` normalizes both point sets, uses full `estimateAffine2D`,
  `beta=0.05`, `maxIters=1000`, and accepts three or more inliers. OpenCV's
  `refineIters=10` is bounded refinement, not claimed to be maximum likelihood.
- Audit P1, unpapered resize/keypoint guards: **(b) for the claim**. The shared
  1600-side bound at `copy_move.py:168` and `MIN_KEYPOINTS=100` at `:14` remain
  product bounds; the catalog no longer presents them as paper constants.
  Fewer than 100 keypoints is `NOT_APPLICABLE` at `:173-179`; enough keypoints
  with no verified link is an applicable zero-evidence result at `:212-218`.

### Signal direction

The premise is two spatially distinct regions with geometrically consistent
local matches. The raw metric is the number of distinct verified clusters at
`copy_move.py:230-243`; higher is more evidence, so `higher_is_worse` remains
physically correct. The false positive and implausible translation above are
method failures, not a sign inversion.

### Calibration impact

The `verified_clusters` key is preserved, but its meaning and range changed;
the old effective eight-inlier path is now the paper's three-pair path. Existing
calibration is invalid and was not edited. Human refitting is required.

### Tests and open items

`backend/tests/test_copy_move.py:71-102` covers generalized matching and joint
clustering; `:105-129` covers normalized full-affine RANSAC and the three-inlier
minimum. Mutual-pair filtering, the depth-4 cut, the 1600-side resize, and
OpenCV-vs-ML refinement remain repository variants. SIFT cannot see the supplied
pasted regions; dense/block matching would be a separate detector.

## 2. `cfa`

### Paper specification

The old catalog cited Popescu–Farid and Ferrara, but this module implements Q.
Bammey, R. Grompone von Gioi, and J.-M. Morel, “Image Forgeries Detection
through Mosaic Analysis: the Intermediate Values Algorithm,” IPOL 11, article
355, 2021; [open IPOL paper](https://www.ipol.im/pub/art/2021/355/revisions/2022-01-01/article.pdf).

Bammey specifies a “2-pixels-wide border,” bidirectional masks with values
`0`, `1/2`, and `1`, `1/(2XY)` count normalization, green-diagonal then
red/blue-pattern selection, connected-component confidence equal to the
maximum absolute count difference, merged diagonal/full maps, and a `gamma`
threshold. Its default table uses bidirectional masks, `gamma=0.2`, and
`32x32` windows.

### Changes and resolutions

- Audit P0, citation-to-method mismatch: **(b)**. Popescu–Farid's EM
  correlation/probability model and Ferrara's predictor/GMM are not present.
  Catalog `detector-catalog.yaml:362-366` now cites Bammey and labels the
  replacement honestly; `cfa.py:1-9` states the same.
- Audit P0, extra `/255`: **(a)**. `cfa.py:199-229` uses unit masks and divides
  by `2.0*blocks`; the extra factor is gone.
- Audit P1, missing localization/confidence path: **(a)**.
  `cfa.py:106-168` uses even 32x32 windows with stride16, computes local full
  and diagonal patterns, applies `_connected_confidence` separately, merges by
  maximum, and expands to the image lattice. `:239-251` assigns maxima.
- Audit P1, one-pixel border: **(a)**. `_intermediate_values` at
  `cfa.py:175-185` masks two pixels on every side and averages horizontal and
  vertical masks.
- Audit P1, unresolved pattern returned as applicable clean: **(a)**.
  `cfa.py:72-84` returns `NOT_APPLICABLE`, `score=None`, and `flagged=None`.
- Audit P1, strict JPEG/EXIF gate: **(b) for the claim**. The gate at
  `cfa.py:42-53` remains product policy and is documented at catalog `:355`;
  the cited paper does not require those EXIF fields.

### Signal direction

The premise is that demosaicing creates intermediate-value periodicity and a
forged region may use another Bayer grid. The absolute local count-difference
confidence is higher for greater inconsistency, so `higher_is_worse` remains
correct. An unresolved pattern abstains and is not interpreted as authentic.

### Calibration impact

`cfa_ratio` moved from `0.00002384559820711729` to `0.09311224520206451` on the
same probe; phase stayed `3`. Existing thresholds/scales require refitting. No
calibration file was edited.

### Tests and open items

`backend/tests/test_cfa.py:49-55` covers the two-pixel border and `:126-144`
covers abstention, full resolution, and unit-mask scale. No eligible real-camera
corpus row was available for end-to-end validation. The original
Popescu–Farid/Ferrara algorithms remain unimplemented because the catalog now
states the actual Bammey variant.

## 3. `ela`

### Paper specification

Paper: N. Krawetz, “A Picture's Worth: Digital Image Analysis and Forensics,”
presented at Black Hat Briefings USA 2007; [open whitepaper](https://www.blackhat.com/presentations/bh-usa-07/Krawetz/Whitepaper/bh-usa-07-krawetz-WP.pdf).

Krawetz says ELA works by “intentionally resaving the image at a known error
rate, such as 95%, and then computing the difference between the images.” A
large difference means pixels are not at their local error minima. The paper
does not define Canny thresholds, texture/noise votes, a two-of-three rule, or
the legacy numeric thresholds.

### Changes and resolutions

- Audit P0, q95-to-q75 synthetic comparison: **(a)**. `ela.py:103-115` now
  decodes the input, saves it once at default quality 95, and returns
  `cv2.absdiff(input,resaved)`. `resave_quality` at `:31-53` remains an
  explicit compatibility override.
- Audit P0, stale unsigned subtraction: **(a)**. `ela.py:113` uses
  `cv2.absdiff`; catalog ELA-1 at `detector-catalog.yaml:59-64` is fixed.
- Audit P0, heuristic classifier presented as ELA: **(b)**. Legacy features
  remain at `ela.py:120-300`, but the module header and catalog `:65-68` call
  them repository-specific heuristics. The adapter remains outside scope and
  maps only `features.edge_discontinuity` at `adapters.py:46-53`.
- Audit P1, default 1024-side Lanczos resize: **(a)**. `MAX_ANALYSIS_SIDE` is
  `None` at `ela.py:21`; resize happens only for explicit `max_image_size` at
  `:55-66`, preserving the default JPEG pixel lattice.
- Audit P1, venue/date and catalog bookkeeping: **(a)**. The module/catalog
  identify the fetched Black Hat USA 2007 whitepaper and distinguish raw
  residual from heuristic features.

### Signal direction

For the paper residual, higher local difference means a cell is less stable at
the controlled quality and is more suspicious, so higher-is-worse is consistent.
The service scalar is still edge discontinuity, not raw residual magnitude; its
direction is retained as a repository heuristic convention, not attributed to
Krawetz.

### Calibration impact

The raw probe moved from mean `45.201985677083336`, max `255` for q95-to-q75 to
mean `6.751302083333333`, max `43` for input-to-q95. The adapter's
`edge_discontinuity` distribution also changes. Existing calibration requires
refitting; no calibration file was edited.

### Tests and open items

`backend/tests/test_ela.py:22-34` covers default/override qualities and
`:116-144` covers no default resize and exact input-to-q95 residual equality.
The service still lacks a paper-defined ELA scalar/decision because the shared
adapter was outside scope. Direct analyzer use on non-JPEG input still
manufactures a JPEG residual; the runtime adapter remains JPEG-only.

## Final open items

1. Refit calibration for changed `copy_move`, `cfa_ratio`, and ELA-derived
   `edge_discontinuity`; calibration work was intentionally excluded.
2. Decide whether to expose raw ELA or retain the explicitly labeled heuristic
   adapter.
3. Validate CFA on eligible full-resolution camera material.
4. Add a separately named dense/block copy-move method if textureless pasted
   regions must be covered; changing SIFT thresholds will not solve that limit.
