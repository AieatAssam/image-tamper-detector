# Open items after three repair rounds — 2026-08-29

Three codex rounds ran. The corpus confound is gone, so the numbers below are
the first ones in this project that mean anything. What remains needs decisions,
not another automated pass.

## O1 — BLOCKING PRODUCT DEFECT — copy_move misses the repo's own reference forgery

`data/samples/tampered/landscape_copy_paste.jpg` is a copy-move forgery this
repository generated itself, with known source and destination regions.

```
FUSED: likely_authentic 0.283
  copy_move     flagged=False  score=0.000  threshold=1.000
  reason: "no_forgery_found: keypoints were sufficient but no verified affine cluster was found"
```

The detector reports sufficient keypoints and then finds no cluster. It is not
the `insufficient_keypoints` path — the detector ran and concluded clean on a
forgery it was written specifically to catch.

Consequence: `copy_move`'s 0.606 within-source AUC is separating something other
than copy-move forgery, and it currently carries a 0.099 fusion weight on that
basis. Also note `threshold=1.000` against `score=0.000` — a degenerate
operating point.

Reproduce:
```
.venv/bin/python - <<'PY'
from fastapi.testclient import TestClient
from backend.app.main import app
c=TestClient(app)
r=c.post('/api/v1/analyze',files={'file':('x.jpg',open('data/samples/tampered/landscape_copy_paste.jpg','rb').read(),'image/jpeg')})
d=r.json(); print(d['verdict'], d['score'])
print([x for x in d['detectors'] if x['id']=='copy_move'])
PY
```

Likely candidates, in order: the `min_offset=32px` self-match filter combined
with the 1600px downscale may be discarding the true matches (the paste offset
is 40% of image width at full resolution but the region is feathered with
sigma=11); the 8px offset-cluster grid may be splitting a genuine cluster below
the 8-member floor; or `estimateAffinePartial2D` inliers may be falling under 8.
Instrument the pipeline stage by stage on this one image before changing
parameters.

## O2 — DECISION NEEDED — prnu holds the largest weight at chance-level skill

`prnu` weight 0.190 (largest in the ensemble), `within_source_auc` 0.5019.
The zero-weight guard requires `<= 0.5`, so 0.5019 passes by 0.0019.

A guard on the point estimate cannot distinguish 0.5019 from 0.5. Fix is a
confidence interval — drop a detector unless its within-source AUC is above 0.5
by more than its standard error — but where that bar sits is a judgment call
about how much a chance-level detector may steer a verdict. Not an agent task.

## O3 — EXPECTED, NOT A BUG — fusion loses to its best member

Fused held-out AUC 0.8267 versus `double_jpeg` alone at 0.8933. S10 is correctly
marked `failed`. With four usable detectors over 120 images and 12 source
groups, fusion underperforming its best member is the expected outcome. Either
accept it and reword S10's gate, or accept that the gate stays red until the
corpus is substantially larger.

## O4 — BLOCKED ON DATA — cfa cannot be validated at all

11 real_camera entries, all graded `relaxed`; 0 `strict`. Per the S05 amendment,
`cfa` consumes only `strict` entries, so it reports `not_applicable` everywhere
and holds weight 0.000. That enforcement is correct — CFA structure does not
survive resizing — but it means the detector is dead weight until strict-grade
cameras exist. Wikimedia returned HTTP 429 with `Retry-After: 600` during the
fetch. Also 0/2 signed-C2PA entries, so the C2PA validation path is untested.

## O5 — MEASUREMENT NOTE — real detector skill is modest

On the balanced corpus, with source held constant:

| detector | within-source AUC |
|---|---:|
| double_jpeg | 0.689 |
| copy_move | 0.606 (but see O1 — likely spurious) |
| spectral | 0.547 |
| exif / entropy / qtable / ela | ~0.50 |
| jpeg_ghosts | 0.487 |
| prnu | 0.437 |
| cfa / c2pa / learned | no applicable rows |

`spectral` fell from an apparent 0.779 to 0.547 once the confound was removed.
The earlier headline numbers (0.855, then 0.9077) were measured on a corpus with
a source-identity shortcut and should not be quoted. 0.8267 fused, n=35 across
12 groups, is the first defensible figure — and per O3 it is still below its own
best member.

## Corrections to the plan itself

Two of the five root causes were defects in `plan/`, not in the implementation.
Both are now amended in `plan/stages/S05-corpus-and-benchmark.yaml`:

1. `real_camera.criteria` demanded EXIF `PixelXDimension` match the decoded
   width. That tag is absent from a large share of genuine camera JPEGs; the
   criterion rejected 10 of 10 valid candidates and produced zero camera
   entries. Now two labelled grades, `strict` and `relaxed`, with `cfa`
   restricted to `strict` in code.
2. No rule required source balance. A corpus whose authentic class came from one
   source and manipulated class from four let every detector separate source
   identity instead of manipulation. `source_balance_rule` now requires both
   classes from every contributing source, a 40% per-class cap, and a
   `within_source_auc` that S10 fits and gates on instead of the pooled AUC.

---

# After round 4 — the conclusion

Four rounds ran. Each one removed a measurement artifact, and the honest number
got worse every time. That is the finding.

| round | fused held-out AUC | why the previous number was wrong |
|---|---:|---|
| baseline | 0.855 | 11 of 12 weights at the L2 floor; one source image, so the group split leaked entirely |
| R1 | 0.9077 | 30 of 40 authentic rows were recompressions of a known forgery or of AI images |
| R3 | 0.8267 | source identity predicted the label; detectors were ranked on which picture it was |
| **R4** | **0.5375** | weights were anti-correlated with skill; corpus now has 12 strict cameras and 2 C2PA fixtures |

**0.5375 with n=44 across 12 balanced source groups is chance.** Best single
detector is 0.600. The ensemble, measured without a shortcut, does not work.

This is not a repair failure — it is the first honest measurement of a system
whose thresholds were originally hand-tuned against four images. The earlier
numbers were never real.

## C1 — What further agent rounds cannot fix

The detectors need algorithm work, not orchestration. `double_jpeg` is the only
one that has ever shown real skill (0.689 within-source at R3, 0.500 at R4 after
the corpus changed). `ela`, `entropy`, `qtable` and `exif` sit at chance on
every honest corpus they have been measured against. No amount of refitting
creates signal that the statistics do not contain.

The realistic paths, in order of expected value:
1. A substantially larger and more diverse corpus. 123 rows from 12 sources
   cannot separate 12 detectors; every held-out estimate here has an SE of
   roughly ±0.08, so most of the ranking is noise. This is the single highest
   value action and it is a data-acquisition task, not a coding one.
2. Re-derive the weak detectors against their literature specifications rather
   than tuning them. `qtable` returning a constant across the whole corpus
   (round 2 established the corpus uses one libjpeg table) means the corpus must
   vary encoders before the detector can be judged at all.
3. Accept that a stateless single-image forensics tool built from classical
   statistics has a low ceiling, and reposition the product around the detectors
   that do work plus C2PA provenance, which is declarative and reliable when
   present.

## C2 — `copy_move` can no longer produce a true negative

Round 4's I1 fix is correct in substance: instrumentation proved both pasted
regions contain **zero SIFT keypoints**, so the detector genuinely cannot see
the reference forgery, and no parameter was loosened to hide that. Reporting
NOT_APPLICABLE rather than "clean" follows the plan's three-valued rule.

But the detector now returns NOT_APPLICABLE for every image where it finds no
verified cluster — 30 of 32 `authentic_recompress` rows, 14 of 14 `splice` rows.
It is therefore structurally incapable of saying "I looked and found nothing."
Consequences: its AUC is computed only on images where it already found
something, which biases it upward; and it can never contribute negative evidence
to fusion.

The distinction the code now collapses is between *no keypoints in any candidate
region* (genuinely cannot assess) and *keypoints present, no matching cluster*
(a real negative). Round 4's instrumentation already computes what separates
them. Splitting those two cases restores the true-negative path.

## C3 — Corpus is the binding constraint on everything else

`real_ai` is 9 of 12 and is the last unmet axis; S05 and S08 stay failed on it.
More importantly, 123 rows is too few for 12 detectors regardless of which axes
are complete. Everything in C1 depends on this being addressed first.
