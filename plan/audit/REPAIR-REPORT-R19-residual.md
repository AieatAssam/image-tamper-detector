# Round 19 residual-family repair report

Date: 2026-09-01

Scope: `prnu.py` / Noisesniffer, `splicebuster.py`, and `resampling.py`. The
catalog edits are limited to these three entries. No calibration artifact,
corpus file, benchmark script, or other detector was changed.

## Status

| detector | audit grade | resolution | claimed grade after repair | result |
|---|---|---|---|---|
| `prnu.py` / Noisesniffer | MINOR-DRIFT | (a) repair the region constant; (b) correct the stale catalog claim | MINOR-DRIFT | the active method and claim now identify Noisesniffer; adapter/docs remain outside this family's ownership |
| `splicebuster.py` | MAJOR-DRIFT | (a) implement the feature, GG-EM model, dense sampling, mask, and ratio decision | MINOR-DRIFT | paper-derived GG branch is present; bounded fitting, gates, and repository calibration remain |
| `resampling.py` | MAJOR-DRIFT | (a) implement Kirchner's fast p-map/cumulative-periodogram detector | MINOR-DRIFT | the cited fast detector is present; service preprocessing and the Popescu exhaustive variant remain adaptations |

## Paper verification

The cited primary sources were fetched and reread:

- Noisesniffer: [Gardella et al., IPOL 2024, article 462](https://www.ipol.im/pub/art/2024/462/)
- Splicebuster: [Cozzolino, Poggi, and Verdoliva, IEEE WIFS 2015](https://www.researchgate.net/publication/284350985_Splicebuster_A_new_blind_image_splicing_detector)
- Resampling: [Kirchner, ACM MM&Sec 2008](https://ws.binghamton.edu/kirchner/papers/2008_MMSec.pdf) and [Popescu and Farid, IEEE TSP 2005](https://farid.berkeley.edu/downloads/publications/sp05.pdf)

## 1. `prnu.py` / Noisesniffer

### Audit items and decisions

1. **P0 stale method/citation/statistic claim — resolution (b), catalog
   correction.** The code already followed Noisesniffer, while the catalog
   described a wavelet/Gaussian local-variance PRNU variant and cited Lukas,
   Fridrich, and Goljan. The catalog now identifies Gardella et al.'s
   Noisesniffer method, its NFA significance score, paper defaults, and the
   absence of a reference-fingerprint PRNU path (`plan/reference/detector-catalog.yaml:76-105`).
   The adapter's old `uniformity_score` wording and the stale
   `docs/detection-principles.md` section could not be edited under the
   family-file restriction; they remain open.

2. **P1 region-growth constant — resolution (a).** The comparison now uses
   the paper's `4.062570` at `backend/app/analysis/prnu.py:218`, matching the
   final NFA term at `backend/app/analysis/prnu.py:107`. The paper describes
   retaining “only a small percentile given by the parameter n” and sets the
   decision threshold epsilon to 1; the implementation retains those stages
   and the `log_nfa < 0` decision (`prnu.py:163-185`, `197-272`).

3. **Unsupported `w=7` option — P2, also resolved.** The constructor and
   frequency-mask table now accept only the paper's tested `w=3,5,8` values,
   with the corresponding `T=3,5,9` map (`backend/app/analysis/prnu.py:153-159`,
   `285-305`).

### Paper specification and implementation

The published method selects overlapping blocks, rejects saturated blocks,
uses DCT-domain low-frequency energy and standard-deviation ordering, keeps
the low-energy fraction, accumulates four-connected cells, and evaluates a
binomial-tail NFA. The implementation contains those stages at
`prnu.py:121-185` and `prnu.py:197-272`. The paper constants are represented
at `prnu.py:77-109` and the defaults at `prnu.py:285-289`:
`w=3`, `B=20000`, `n=0.1`, `m=0.5`, `l_beta=100`, `0.316915`, and `4.062570`.

The 1024-side `INTER_AREA` cap at `prnu.py:33` and `65-74` is a deployment
adaptation. The paper reports downsampling as an experiment and observes
degraded detection; it does not prescribe this service-wide cap. It remains
P2 and is explicitly retained rather than presented as a paper constant.

### Signal direction

The physical premise is local noise-model coherence: a forged region can have
a different noise population. The raw statistic is `-log10(NFA)`
(`prnu.py:112-118`), and the detector accepts a region when `NFA < 1`
(`prnu.py:264`). Therefore lower NFA means more suspicious and higher
significance means worse. `higher_is_worse` is directionally correct; no sign
was changed.

### Measurement

The read-only comparator loaded the old source with `git show HEAD:<path>` and
ran the current source on the same inputs. Its exact output was:

```text
prnu_region_members old/new 2 1
prnu old/new True 15.78352788430835 True 15.78352788430835
```

The first line is the marginal region-growth regression fixture: using the
published constant prevents one candidate neighbor from being admitted. The
reference `data/samples/tampered/landscape_copy_paste.jpg` score did not move
in this run, and no corpus AUC was claimed.

### Calibration impact

The raw score key and direction did not change. The corrected region-growth
constant can change borderline NFA values, so the human should include this
detector in the post-round recalibration pass. No calibration file was edited.

## 2. `splicebuster.py`

### Audit items and decisions

1. **P0 feature representation — resolution (a).** The paper says “the final
   feature has length 50, which is further reduced to 25 through PCA.” The
   implementation now builds 81-bin horizontal and vertical histograms,
   pools reversal/sign symmetry to 25 bins per orientation, concatenates the
   50 values, applies square-root normalization, and performs PCA to 25
   dimensions (`backend/app/analysis/splicebuster.py:50-67`, `158-208`).

2. **P0 blind model — resolution (a), GG branch.** The paper's blind method
   learns two processing populations with EM; its Gaussian-Gaussian decision
   is a ratio of class Mahalanobis distances. The implementation now has two
   Gaussian covariance/mean updates, multiple starts, and a genuine/forged
   distance ratio at `splicebuster.py:228-305`. It deliberately selects the
   paper's GG branch; the paper's Gaussian-uniform alternative is not selected.

3. **P1 dense sampling — resolution (a).** The paper's dense formulation says
   “with unit stride.” `BLOCK_STRIDE` is now `1` and block starts use it at
   `splicebuster.py:24-25`, `108-110`.

4. **P1 decision statistic and constants — resolution (a) for the statistic;
   calibration retained.** The detector now uses the maximum
   genuine-to-forged Mahalanobis ratio (`splicebuster.py:397-410`) instead of
   the old one-class maximum distance. The existing `5.0` threshold and `2.0`
   logistic scale are repository settings, not paper constants, and were not
   tuned (`splicebuster.py:29-30`).

5. **P1 saturated/very-dark masking — resolution (a).** Block means and exact
   0/255 saturation fractions are computed and invalid blocks are excluded at
   `splicebuster.py:185-208`. The paper specifies the excluded areas but not
   these numeric service cutoffs: mean `<=16`/`>=255` or saturation fraction
   `>1%` are repository adaptations.

6. **P2 deployment gates/preprocessing — retained honestly.** The JPEG
   qtable-quality `>=80` self-gate remains at `splicebuster.py:349-364`; it was
   not removed. The second 1024-side cap and reflected border rule also remain
   as service adaptations. The catalog now states those boundaries
   (`plan/reference/detector-catalog.yaml:478-506`).

### Paper specification and implementation

The paper's front end is third-order residuals, `T=1`, `q=2`, four-symbol
co-occurrences, symmetry pooling, square-root/unit-L2 normalization, and PCA.
Those values and stages are now at `splicebuster.py:85-105`, `112-183`.
The model uses two classes with EM and a GG distance-ratio decision; the code
uses 30 starts (`splicebuster.py:35`) and a deterministic seed. The service
fits PCA and EM on at most 4096 deterministic rows while scoring all dense
blocks (`splicebuster.py:141-155`, `228-305`) to keep the bounded service
runnable. That cap is ours, not a paper constant.

### Signal direction

The premise is that pasted and genuine regions arise from different local
processing-chain populations. A suspicious block should be closer to the
forged class than to the genuine class, so the ratio
`genuine_distance / forged_distance` rises. The maximum ratio at
`splicebuster.py:397-410` and `higher_is_worse=True` at
`splicebuster.py:342-345` are directionally consistent. No inversion was used
to improve an AUC.

### Measurement

The same read-only HEAD/current comparator produced:

```text
splicebuster_forged raw old/new 5.615908145904541 6.566456900231344
splicebuster_forged dims old/new 81.0 50.0 25.0
splicebuster state old/new applicable applicable score old/new 0.9043729926870624 0.8082545443840142 raw old/new 9.493573188781738 7.877416597293427
```

The synthetic processing-fingerprint raw statistic rose, while the concrete
copy-paste reference case's raw ratio and score fell. Both directions are
recorded; the lower reference score is not treated as a reason to loosen the
gate or invert the sign. The qtable gate regression test still passes.

### Calibration impact

The decision raw statistic changed from `mahalanobis_max` of a one-class model
to `mahalanobis_ratio_max` of the GG model. The old `mahalanobis_max` key is
retained as the genuine-class distance for compatibility, but the decision
uses the new ratio. Existing Splicebuster calibration is therefore not
portable; the human must refit it after all round repairs. No calibration
artifact was edited.

## 3. `resampling.py`

### Audit items and decisions

1. **P0 p-map omission — resolution (a).** Kirchner's Eq. 21 uses
   `p = lambda exp(-(abs(e)/sigma)^tau)` with `lambda=1`, `sigma=1`, and
   `tau=2`. The implementation now maps the uint8 service residual to
   `exp(-(abs(e/255))^2)` at `backend/app/analysis/resampling.py:49-62`.
   The `/255` is a service intensity normalization; the paper's equation does
   not prescribe the repository's uint8 representation.

2. **P0 spectral decision — resolution (a).** The old local Hann/block ratio
   is gone. The implementation now applies Popescu/Farid's radial spatial
   window before the FFT, the rotationally invariant high-pass filter, and
   gamma 4, then computes Kirchner's first-quadrant cumulative periodogram and
   maximum gradient (`resampling.py:65-101`). The paper defines `C` in
   `[0,1]` from the first quadrant and uses the maximum absolute gradient;
   it also says “we employ a Sobel edge detector.” The code uses Sobel at
   `resampling.py:97-100`.

3. **P0 local-block premise — resolution (a).** The score is now one global
   cumulative-periodogram gradient (`resampling.py:104-118`, `177-190`), not
   block disagreement. A global resize is no longer deliberately suppressed.

4. **P1 block/size geometry and inconsistent error — resolution (a).** The
   obsolete 128/64 block aggregation and 512 minimum are removed. The
   applicability gate and measurement error now consistently use 256px
   (`resampling.py:18-19`, `104-109`, `149-159`). This is aligned with the
   paper's 256x256 experimental crop, not claimed as a universal paper gate.

5. **P2 experiment preprocessing — retained as an explicit adaptation.** The
   service converts RGB to grayscale, uses the shared input and a 1024-side
   `INTER_AREA` cap (`resampling.py:36-46`), and accepts several decoded
   formats. Kirchner's experiment used never-compressed grayscale images and
   nearest-neighbor factor-two downsampling to avoid CFA interference. The
   Popescu/Farid exhaustive synthetic-map search is not implemented because
   this repair selected Kirchner's accelerated cumulative-periodogram branch;
   the catalog now says so (`plan/reference/detector-catalog.yaml:508-532`).

### Signal direction

The cited physical premise is that interpolation creates periodic artifacts in
prediction residuals. A stronger sharp spectral/cumulative-periodogram
gradient means more resampling evidence, so the current `higher_is_worse=True`
at `resampling.py:141-145` is correct. The new score is not the old local
disagreement score, but its direction was not changed to chase results.

### Measurement

The same read-only HEAD/current comparator produced:

```text
resampling_raw old/new 0.052046071738004684 0.14850585162639618
resampling_metric old_local_inconsistency/new_periodogram_delta 0.052046071738004684 0.14850585162639618
resampling state old/new applicable applicable score old/new 0.9997280030853758 0.9996880992499959 raw old/new 0.44337791204452515 0.4379005432128906
```

The first pair is deterministic synthetic noise; the second is the concrete
`landscape_copy_paste.jpg` reference forgery. The new raw statistic moves up
on the synthetic case and down on the reference case. No local-resampling AUC
was claimed because the catalog and audit record that the corpus lacks a
labeled positive resampling family.

### Calibration impact

The raw statistic changed from `local_inconsistency` to
`periodogram_delta`; the threshold `0.115` and scale `0.04` remain only as
the existing calibration interface (`resampling.py:29-33`). They are not
paper constants and are not valid post-repair calibration. The human must
refit or invalidate this detector's old calibration before fusion.

## Verification commands and real output

Focused owned tests:

```text
$ .venv/bin/python -m pytest backend/tests/test_prnu.py backend/tests/test_splicebuster.py backend/tests/test_resampling.py -q
22 passed in 19.81s
```

Required full suite:

```text
$ .venv/bin/python -m pytest backend/tests -q
132 passed, 1 warning in 131.12s (0:02:11)
```

The warning was the existing Starlette/httpx deprecation warning from
`fastapi/testclient.py`; it was not a test failure.

Catalog parse:

```text
$ .venv/bin/python -c "import yaml,sys; yaml.safe_load(open('plan/reference/detector-catalog.yaml'))"
# no output; exit 0
```

## Still open

- Refit residual-family calibration after all five repair agents land. No
  calibration JSON was touched.
- Update the residual-family prose in `docs/detection-principles.md` and the
  adapter's Noisesniffer naming/metric wording; both are outside the owned
  file list for this repair.
- Splicebuster still uses the paper's GG branch only, fits PCA/EM on a bounded
  4096-row sample, retains the JPEG quality gate and 1024 cap, and does not
  implement GU. These are disclosed adaptations, not silently claimed paper
  constants.
- Resampling does not implement Popescu/Farid's 692-map/160-160-45 exhaustive
  similarity search; the selected Kirchner fast branch is the implemented
  method. The corpus still has no labeled positive resampling family.
