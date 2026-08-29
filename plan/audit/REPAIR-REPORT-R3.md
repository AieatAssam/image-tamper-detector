# Round 3 repair report - 2026-08-29

Round 3 removes the measured source-identity confound from the synthetic
corpus and makes calibration source-local. The real corpus is improved but
still incomplete, so S05, S08, and S10 remain failed in `plan/STATUS.yaml`.

## H1 - fixed

Changed `scripts/make_corpus.py`, `scripts/benchmark.py`,
`scripts/calibrate.py`, `data/corpus/MANIFEST.yaml`, the synthetic index and
sidecars, `backend/tests/test_corpus.py`, `docs/corpus.md`, and
`docs/calibration.md`.

The before command was:

```sh
.venv/bin/python scripts/benchmark.py --out /tmp/r3-before.json --corpus all
```

The supplied reproduce block then reported these pooled/landscape-only AUCs:

```text
double_jpeg 0.3070833333333333 0.7808823529411765
cfa         0.6458333333333334 0.38529411764705884
prnu        0.6516666666666666 0.36470588235294116
spectral    0.77875             0.7
```

The after command was:

```sh
.venv/bin/python scripts/benchmark.py --out /tmp/post-r3-final.json --corpus all
```

Its source-balanced synthetic pooled / source-held-constant results are:

```text
double_jpeg 0.5508333333333333 0.6628352490421456
cfa         unavailable        unavailable
prnu        0.5316666666666666 0.5019157088122606
spectral    0.4979166666666667 0.5593869731800766
```

The benchmark now emits `within_source_auc` for every detector. The fused
source-local held-out AUC is `0.8266666666666667`. CFA is unavailable rather
than falsely scored because the currently downloaded camera evidence is
relaxed; this is the intended H3/H4 interaction.

## H2 - fixed

Changed `scripts/make_corpus.py`, `data/corpus/synthetic/index.json` and its
sidecars, `backend/tests/test_corpus.py`, and `docs/corpus.md`.

Before, 40 authentic rows were concentrated on `landscape_original.jpg`,
while manipulated rows came from four sources. After generation, the command
reported:

```text
generated 100 entries: 40 authentic, 60 manipulated
```

The source-balance assertion then reported 12 contributing sources, class
totals of 40/60, and every source supplied both classes. The largest source
contribution was 4 authentic rows and 7 manipulated rows, below the 40 percent
limit for either class. Authentic output rows are restricted to genuinely
authentic camera/sample sources.

## H3 - partially-fixed; blocked on corpus shortfall

Changed `data/corpus/MANIFEST.yaml`, `scripts/fetch_corpus.py`,
`scripts/make_corpus.py`, `docs/corpus.md`, and the S05 camera acceptance
check. The amended strict/relaxed criterion is enforced by the fetch verifier,
and every camera entry records `unresized_evidence`.

Before: `real_camera` was `0/12`; ten otherwise qualifying candidates had
been rejected for missing `PixelXDimension`. After:

```text
11 real_camera entries, all unresized_evidence=relaxed
9 real_ai entries
0 real_c2pa_signed entries
20 manifest entries verified
```

The exact missing fact is one more downloaded, checksum-verified JPEG with
EXIF Make/Model, plausible dimensions for that Model, no editor Software tag,
and absent `PixelXDimension` or strict matching `PixelXDimension`. Three more
AI entries and two signed C2PA entries are also missing from the amended S05
counts.

Commands tried included the Wikimedia Commons API discovery query, direct
upload-host downloads, cached candidate inspection with Pillow, and:

```sh
.venv/bin/python scripts/fetch_corpus.py --check
```

The upload host returned HTTP 429 with `Retry-After: 600` during discovery.
Existing verified bytes were cached and reused. No URL, checksum, license, or
unverified C2PA entry was invented.

## H4 - fixed, with strict-camera coverage still blocked by H3

Changed `scripts/make_corpus.py`, `backend/app/analysis/exif.py`,
`backend/app/analysis/cfa.py`, `scripts/calibrate.py`,
`backend/tests/test_cfa.py`, `docs/calibration.md`, and the generated
calibration output. `backend/app/analysis/qtable.py` did not need a detector
change: the diagnosis was a corpus fault.

Before, EXIF and qtable both reported pooled and within-source AUC `0.500`.
After the generator varies Software, datetime, thumbnails, and JPEG tables;
the final benchmark found 3 EXIF score values and 3 qtable score values over
their applicable rows. Their source-local AUCs are `0.5076628352490421` and
`0.4731800766283525`, respectively. Qtable is therefore explicitly assigned
weight `0.0` with reason
`within_source_auc=0.473180 <= 0.5`.

CFA now consumes only strict real-camera evidence. All 11 current camera
entries are relaxed, so CFA returns `NOT_APPLICABLE` with score `None` for
them; no relaxed row is treated as a zero score.

## H5 - fixed

Changed `scripts/calibrate.py`, `backend/app/analysis/calibration.json`, and
`docs/calibration.md`.

Before, the H1 control showed positive weights for below-chance CFA (`0.186`)
and PRNU (`0.147`), while double JPEG had weight `0.000` despite its controlled
signal. After fitting on the source-balanced corpus, the explicit guard
excludes every detector with `within_source_auc <= 0.5` from the fit. The
post-R3 check found no below-chance detector with positive weight; qtable is
the concrete case at `0.4731800766283525 -> 0.0` with a recorded reason.

The final calibration records `n_images: 120`, `heldout.split_by:
source_image`, and fused held-out AUC `0.8266666666666667`. Double JPEG now
has within-source AUC `0.6628352490421456` and weight
`0.1402731766507949`.

The existing relative S10 gate still fails honestly: fused held-out AUC is
`0.8266666666666667` versus the best single-detector held-out AUC
`0.8933333333333333`. This is recorded in `plan/STATUS.yaml`; no absolute AUC
floor or weakened acceptance check was introduced.

## H6 - fixed

Changed `scripts/benchmark.py` and `plan/reference/api-contract.yaml`.

Before, benchmark JSON omitted `generated_at` while the contract described it.
After, the benchmark still intentionally omits the field and the contract
explicitly declares that omission. The exact check against
`/tmp/post-r3-final.json` reported:

```text
generated_at_in_benchmark False
```

`calibration.json` continues to carry a real UTC `generated_at` timestamp.

## Verification

- `.venv/bin/python scripts/fetch_corpus.py --check`: 20 manifest entries verified.
- `.venv/bin/python -m pytest backend/tests -q`: `55 passed, 1 warning`.
- `.venv/bin/python scripts/benchmark.py --out /tmp/post-r3-final.json --corpus all`: exit 0; 120 images.
- `.venv/bin/python plan/validate.py`: passed; 15 stages and 219 shell snippets checked.
- `git diff --check`: passed.

No git commit was created.
