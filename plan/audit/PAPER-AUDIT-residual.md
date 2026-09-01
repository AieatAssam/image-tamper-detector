# Paper-fidelity audit: residual family

Read-only audit of the working tree on 2026-09-01. The only file changed for this audit is this report. Line references are to the implementations and catalog entries inspected during the audit.

## Grade summary

| Detector | Grade | Bottom line |
|---|---|---|
| `prnu.py` / Noisesniffer | **MINOR-DRIFT** | The core Noisesniffer a-contrario method is present. One NFA region-growth constant, one unsupported block size, the analysis downsampling, and the catalog/public description drift. |
| `splicebuster.py` | **MAJOR-DRIFT** | The residual and quantization front end is close, but the published feature reduction and blind EM model are absent; stride, masking, and decision thresholds are repository substitutions. |
| `resampling.py` | **MAJOR-DRIFT** | The fixed predictor is faithful to Kirchner, but the published p-map/cumulative-periodogram detector is replaced by a new block-disagreement heuristic. |

## `prnu.py` / Noisesniffer

**Grade: MINOR-DRIFT.** The implementation is substantially the IPOL Noisesniffer method, but it is not an exact reproduction. The catalog entry is also for a different method and is stale relative to the current default code path.

### Paper specification

Paper: Gardella, Musé, Colom, and Morel, “Image Forgery Detection Based on Noise Inspection: Analysis and Refinement of the Noisesniffer Method,” IPOL 2024, article 462: [IPOL article](https://www.ipol.im/pub/art/2024/462/).

The paper's premise is that camera processing produces a coherent final noise model and that a local forgery can disrupt that coherence. The method is not PRNU camera attribution. Its published pipeline is:

- enumerate overlapping `w x w` blocks and discard blocks containing saturated pixels;
- sort valid blocks into intensity bins; compute a 2-D DCT-II; use only the low/medium frequency mask excluding DC;
- in each bin retain the lowest-noise/energy percentile `n`, then retain the lowest standard-deviation percentile `m`; reject bins with too many zero-variance blocks;
- aggregate the selected blocks into `L` and the low-standard-deviation subset `V`;
- use the binomial a-contrario model, the overlap-compensating `w^2` grid factor, square cells, 4-connectivity, seed condition `K_beta/N_beta > m`, region growth, and NFA threshold `epsilon = 1`;
- report the most meaningful region and its NFA.

The paper's tested/default values are `w=3`, `B=20000`, `n=0.1`, `m=0.5`, and `l_beta=100`. Its region NFA uses the constants `0.316915` and `4.062570`. Short paper wording relevant to the selection and decision is: “only a small percentile given by the parameter n is kept” and “threshold ε is set to 1.”

### Our implementation

The code has the main published stages:

- `prnu.py:121-160` uses overlapping blocks, per-channel saturation rejection, DCT-II, and the paper's low/medium-frequency construction.
- `prnu.py:163-185` bins by intensity, selects the low-energy `n` fraction, sorts by standard deviation, rejects flat selections, and constructs `L` and `V`.
- `prnu.py:77-109` implements the binomial tail and NFA terms, including `w^2`, `0.316915`, and `4.062570`.
- `prnu.py:188-272` implements square-cell counts, 4-connected region growth, the seed ratio, NFA testing, the `epsilon=1` decision (`log_nfa < 0`), and the significance output.
- Defaults in `prnu.py:285-289` match the paper's `w=3`, `B=20000`, `n=0.1`, `m=0.5`; the default cell size matches `l_beta=100`.

### Deltas and evidence

1. **Region growth uses the wrong published constant.** `prnu.py:218` compares candidate regions with `log(4.0)`. The paper's region-growth NFA term uses `4.062570`; the same code correctly uses `4.062570` in the final NFA at `prnu.py:107`. This can change which neighboring cells are admitted, even when the final NFA formula is otherwise correct. **Fix priority: P1.**

2. **`w=7` is an added, unsupported variant.** `prnu.py:153-159` defines a mask for `w=7`, and `prnu.py:300-301` accepts it. The paper's tested mask family is `w in {3, 5, 8}` with thresholds `T={3,5,9}`. The default `w=3` is faithful, but the public constructor claims a paper-derived option that the paper does not specify. **Fix priority: P2.**

3. **The 1024-side preprocessing is ours, not the paper's prescribed method.** `prnu.py:33` and `prnu.py:65-74` resize any larger image with `INTER_AREA` before analysis. The paper reports a separate downsampling experiment, including keeping one sample out of two, and observes degraded performance after downsampling; it does not specify this repository-wide 1024-side bound as part of the method. This changes the effective block and cell scale and can remove small local evidence. The shared `ctx.downscaled_rgb_uint8` path adds another possible resampling before this cap. **Fix priority: P2, or explicitly treat the cap as a calibrated deployment adaptation.**

4. **The catalog and adapter describe the wrong statistic.** `plan/reference/detector-catalog.yaml:92-112` says the detector extracts a wavelet/Gaussian residual, reports local variance, uses a threshold of 300, and cites the Lukas/Fridrich/Goljan PRNU camera-attribution paper. The current default path instead returns `-log10(NFA)` from Noisesniffer (`prnu.py:112-118`, `prnu.py:307-324`); its active calibration is threshold `3.2242669451552217` and scale `10.652132275660499` (`backend/app/analysis/calibration.json:291-301`). The adapter still says “noise residual variance” and emits `uniformity_score` (`backend/app/analysis/adapters.py:83-86`). The module docstring explicitly says it is not PRNU attribution (`prnu.py:1-14`). This is a metadata and product-semantics defect, not evidence that the Noisesniffer algorithm is absent. **Fix priority: P0.**

### Signal direction

The paper's physical premise is “noise-model coherence disrupted by a local forgery.” The code's raw statistic is NFA significance: `-log10(NFA)` (`prnu.py:112-118`), and it flags when `NFA < 1` (`prnu.py:264`). Thus lower NFA means more suspicious and higher returned significance means worse. The adapter uses `higher_is_worse=true` (`backend/app/analysis/calibration.json:299`), which is directionally correct for the current statistic. The catalog's “HIGHER local variance” statement (`detector-catalog.yaml:110`) is not directionally descriptive of the current Noisesniffer statistic and should not be used as its rationale.

### Prioritized fixes

1. **P0:** Rename/update the catalog, adapter reason, metric name, and user-facing concept to noise-residual inconsistency / NFA significance; remove the PRNU citation and threshold-300 claim.
2. **P1:** Change the region-growth `4.0` at `prnu.py:218` to the paper's `4.062570`.
3. **P2:** Remove or clearly label `w=7`; decide whether the 1024-side cap is a deliberate calibrated deployment adaptation, then calibrate and document it at that analysis resolution.

The remaining discrepancy is implementation drift around constants and deployment preprocessing. The central published method is present, so this is not a reason to discard the detector as a method; first correct the fidelity and metadata issues.

## `splicebuster.py`

**Grade: MAJOR-DRIFT.** The residual/co-occurrence front end follows the paper's starting point, but the published feature representation and blind inference procedure are replaced by materially different statistics.

### Paper specification

Paper: Cozzolino, Poggi, and Verdoliva, “Splicebuster: A new blind image splicing detector,” IEEE WIFS 2015, DOI 10.1109/WIFS.2015.7368565. The public full text used for this audit is [the paper copy](https://www.researchgate.net/publication/284350985_Splicebuster_A_new_blind_image_splicing_detector).

The paper assumes that host and pasted regions have different local residual-feature distributions and learns those distributions from the image in its blind mode. The published pipeline is:

- grayscale/high-pass third-order residuals;
- quantization with `T=1` and `q=2`, producing three residual symbols;
- four-sample row/column co-occurrences;
- symmetry pooling, square-root normalization to a unit-L2 feature, and PCA reduction. In the paper's experiment the feature is length 50 and is reduced to 25;
- local 128x128 features; the dense unsupervised formulation uses unit stride;
- a two-class model learned jointly with segmentation by EM, with the paper discussing Gaussian-Gaussian and Gaussian-uniform variants. In the supervised variant, statistics come from a selected training set;
- saturated and very dark areas are excluded from the usable localization map;
- the paper leaves binary heat-map conversion as a later practical step and does not prescribe the repository's numeric threshold `5.0` or logistic scale `2.0`.

The paper's wording is explicit: “Feature extraction is based on three main steps”; “the final feature has length 50, which is further reduced to 25 through PCA”; and the dense formulation operates “with unit stride.”

### Our implementation

The following front-end terms are present and numerically aligned:

- `splicebuster.py:48-57` applies the third-order `[1,-3,3,-1]` residual and `q=2`, truncation `[-1,+1]`.
- `splicebuster.py:60-68` encodes four-symbol co-occurrences into `3^4=81` bins.
- `splicebuster.py:97-120` builds horizontal and vertical block histograms over 128x128 blocks and normalizes them.

The inference actually used is different: `splicebuster.py:123-136` fits one regularized Gaussian to all blocks and returns a square-root Mahalanobis distance; `splicebuster.py:211-213` takes the maximum distance; `splicebuster.py:168-173` applies repository settings `threshold=5.0`, `scale=2.0`; and `splicebuster.py:71-73` samples block origins at 32px stride.

### Deltas and evidence

1. **The published feature vector is not implemented.** `splicebuster.py:118-120` sums horizontal and vertical histograms and performs only L1 normalization. It does not perform the paper's symmetry pooling, square-root/Hellinger normalization, or PCA reduction. The implementation exposes 81 dimensions (`splicebuster.py:29-30`, `97-120`), while the paper's experiment reports 50 before PCA and 25 after. These are not cosmetic changes: they alter the feature space and every downstream covariance/distance. **Fix priority: P0.**

2. **The published blind model is omitted.** `splicebuster.py:123-136` fits one Gaussian to the same image and scores the maximum self-distance. The blind paper method learns the genuine/forged classes jointly with EM; the paper's two-class posterior is not present. The catalog openly records this omission (`plan/reference/detector-catalog.yaml:458-470`), so it is known drift, but it is central enough to make the detector a major variant. If the code intends the paper's supervised one-class alternative instead, it still has no separate host training population: it estimates mean and covariance from the query image itself. **Fix priority: P0.**

3. **Stride differs by an order of magnitude from the dense method.** `splicebuster.py:21-22` and `71-73` use 32px stride; the paper's dense formulation says “with unit stride.” The coarser stride reduces localization coverage and changes the sample population used for the model. **Fix priority: P1.**

4. **The decision statistic and constants are repository inventions.** `splicebuster.py:26-27` sets `threshold=5.0` and `scale=2.0`, and `splicebuster.py:211-221` uses a maximum square-root Mahalanobis distance. The paper's model uses class posteriors or a model-distance decision and says binary conversion remains to be done; it does not supply these two values. The calibration is therefore an engineering layer, not a paper constant, and must be recalibrated after restoring the feature/model. **Fix priority: P1.**

5. **The paper's usable-area masking is absent.** The paper excludes saturated and very dark areas from the localization. `splicebuster.py:48-120` has no intensity validity mask; the only applicability checks are JPEG format, qtable proxy, quality, and size (`splicebuster.py:175-190`). Saturated/dark blocks can therefore enter the fitted distribution and the map. **Fix priority: P1.**

6. **Preprocessing and applicability are added adaptations.** `splicebuster.py:33-45` adds a second 1024-side `INTER_AREA` resize. The paper's experiment crops camera images to 768x1024 for speed, but that is not the detector's general 1024-side preprocessing rule. `splicebuster.py:156` restricts the class to JPEG, and `175-182` adds a hard estimated-quality gate of 80. The paper discusses JPEG robustness as an issue to explore, but does not specify this qtable-derived `quality >= 80` abstention rule. `splicebuster.py:50-52` also chooses `BORDER_REFLECT101`; the paper's interior residual equation does not make that border policy a published constant. **Fix priority: P2**, unless these are explicitly retained as deployment gates and excluded from claims of paper fidelity.

### Signal direction

The paper's premise is that a pasted region comes from a different processing population, so its feature should depart from the genuine-region model. A larger distance or stronger forged-class posterior is more suspicious. The current raw statistic is maximum block Mahalanobis distance (`splicebuster.py:211-223`), and `higher_is_worse=True` (`splicebuster.py:166-171`), so the sign is correct for the implemented approximation. This is not a direction inversion; it is a model/feature drift. The catalog's direction at `detector-catalog.yaml:471` is therefore reasonable, but it does not make the current statistic the published Splicebuster detector.

### Prioritized fixes

1. **P0:** Implement the paper's symmetry-pooled, square-root-normalized, PCA-reduced feature and the two-class EM inference, or rename/reclassify this as a separate one-Gaussian residual-cooccurrence heuristic.
2. **P1:** Restore the paper's dense sampling if localization fidelity is required; add saturated/very-dark masking; replace max-distance thresholding with the model decision used by the selected paper variant.
3. **P2:** Re-evaluate the 1024 cap, JPEG-quality gate, border rule, and all thresholds as deployment adaptations; recalibrate only after the method choice is fixed.

The front end is a useful related feature extractor, but current performance should be attributed to this approximation, not used as evidence that the published Splicebuster method performs poorly.

## `resampling.py`

**Grade: MAJOR-DRIFT.** The 3x3 predictor is faithful to Kirchner's accelerated detector. The signal extraction, aggregation, preprocessing, and intended behavior are otherwise materially different from both cited papers.

### Paper specification

The catalog cites both:

- Popescu and Farid, “Exposing Digital Forgeries by Detecting Traces of Resampling,” IEEE TSP 2005: [open paper PDF](https://farid.berkeley.edu/downloads/publications/sp05.pdf).
- Kirchner, “Fast and Reliable Resampling Detection by Spectral Analysis of Fixed Linear Predictor Residue,” ACM MM&Sec 2008: [open paper PDF](https://ws.binghamton.edu/kirchner/papers/2008_MMSec.pdf).

Popescu and Farid's method models resampling correlations, estimates a probability map, transforms that map spectrally with a radial window/high-pass operation and gamma correction, and compares the result against synthetic resampling maps. Their candidate search uses 160 scale cases for upsampling, 160 for downsampling, and 45 rotations. Their experiments also show sensitivity to JPEG quality and emphasize uncompressed or minimally compressed inputs.

Kirchner's accelerated method keeps the fixed 3x3 predictor, maps the prediction error to a p-map, takes its spectral representation, and detects a characteristic periodic anomaly with a cumulative periodogram. Its fixed controls are `lambda=1`, `sigma=1`, and `tau=2`; the published fast decision uses the maximum gradient of the cumulative periodogram and a Sobel edge detector. The paper's own short descriptions include “fixed filter coefficients,” “p can be seen as a contrast function,” and “we employ a Sobel edge detector.”

Kirchner's reported experiment uses never-compressed 8-bit grayscale images, downsamples by a factor of two with nearest-neighbor sampling to avoid CFA interference, and crops 256x256 regions. Its original exhaustive search, described separately from the accelerated detector, evaluates 692 synthetic maps: 601 scaling maps at step 0.0025 and 91 rotation maps at step 0.5.

### Our implementation

The exact predictor is present in `resampling.py:24-31`, and `_absolute_residual` applies it in `resampling.py:60-65`. The rest of the implementation is:

- grayscale conversion and a second 1024-side `INTER_AREA` resize (`resampling.py:39-49`);
- 128x128 blocks at 64px stride (`resampling.py:20-23`, `52-57`, `90-100`);
- a Hann-windowed, centered DFT of the absolute residual and a 99.5th-percentile-to-median radial spectral ratio (`resampling.py:68-80`);
- a raw score equal to the 75th-percentile absolute deviation of block ratios from their median (`resampling.py:101-113`);
- provisional threshold `0.115`, scale `0.04`, and `higher_is_worse=True` (`resampling.py:33-36`, `140-146`).

### Deltas and evidence

1. **The p-map is omitted.** `resampling.py:60-65` passes `abs(I-P(I))` directly to the DFT. Kirchner maps the prediction error to `p = lambda * exp(-(abs(e)/sigma)^tau)` before spectral analysis, with `lambda=1`, `sigma=1`, `tau=2`. The current statistic therefore has a different dynamic range and physical interpretation. **Fix priority: P0.**

2. **The published spectral decision is replaced.** `resampling.py:68-80` uses a per-block Hann-windowed FFT, a radial annulus, and an invented 99.5th-percentile/median ratio. `resampling.py:101-105` then measures inter-block disagreement. There is no cumulative periodogram, first-quadrant construction, maximum periodogram gradient, Sobel edge detector, radial synthetic-map similarity, gamma correction, or candidate parameter sweep. Those omissions cover the core detector, not an optional optimization. **Fix priority: P0.**

3. **The local-block score changes the physical premise.** The papers detect resampling from a coherent periodic spectral artifact anywhere in the image. The current code deliberately suppresses a uniformly resized image and only scores disagreement among local blocks (`resampling.py:103-105`, `127-137`). That can be a useful local-splice heuristic, but it will miss the global-resampling case the cited methods target. **Fix priority: P0**, or change the detector's citation/family to a new local-resampling method.

4. **The block and size constants are ours.** `resampling.py:20-23` sets max side 1024, block 128, stride 64, and minimum side 512. Neither cited paper specifies this block-disagreement geometry. Kirchner's 256x256 crop is an experiment detail, not a 512-side applicability rule. The `_measure` error text also says “at least 256x256” at `resampling.py:87-88`, while the actual gate is 512px at `resampling.py:156-157`; this is an internal message defect, not a paper match. **Fix priority: P1.**

5. **The input preprocessing does not reproduce the cited experiment.** `resampling.py:39-49` converts RGB to grayscale and uses `INTER_AREA` for the second resize. Kirchner's experiment used never-compressed 8-bit grayscale and nearest-neighbor factor-two downsampling to avoid CFA interference. The detector also advertises JPEG, PNG, WEBP, and TIFF with no quality gate (`resampling.py:127-132`), while Popescu and Farid report degraded detection as JPEG compression becomes stronger. This is a broad deployment adaptation with no paper-derived threshold. **Fix priority: P2**, unless supported formats and resampling policy are explicitly treated as a new calibrated operating point.

### Signal direction

For the cited papers, a stronger periodic spectral anomaly means more evidence of resampling, so `higher_is_worse=True` is the correct direction. For the current code, the raw statistic is not that published anomaly: it is local block-to-block disagreement (`resampling.py:101-113`). Higher disagreement is indeed worse for the code's stated local-splice premise, so the sign is internally consistent but not sufficient to claim paper fidelity. The intentional low score for global resize is a behavioral divergence, not an inversion.

### Prioritized fixes

1. **P0:** Choose the intended method. For Kirchner fidelity, add the p-map and cumulative-periodogram/Sobel decision. For Popescu/Farid fidelity, implement the probability-map spectral comparison and its candidate map generation. Do not continue citing both while using neither decision statistic.
2. **P1:** Remove or separately name the local block-disagreement aggregation; align block/size rules and correct the 256-vs-512 message if the heuristic remains.
3. **P2:** Reproduce the paper's compression and grayscale/downsampling assumptions or document a new calibration for each supported format and preprocessing path. The provisional `0.115` and `0.04` values are not paper constants.

The fixed predictor is a faithful reusable component. The detector built around it is a major method substitution, so weak results from this implementation cannot be attributed to Kirchner or Popescu/Farid.
