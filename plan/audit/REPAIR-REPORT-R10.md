# Round 10 repair report

Date: 2026-08-30

## Result

Round 10 makes AI-generation detection measurable on two open-access Zenodo
axes. S1 was fetched, extracted, sampled, verified, and benchmarked before S2
was touched. No image or archive bytes are tracked, and no commit was created.

The benchmark reports rank AUC. `±` is the Hanley-McNeil AUC standard error,
not a confidence interval. The negative scope is recorded as
`real_camera` because neither downloaded archive includes the genuine camera
counterpart bytes. These are cross-source comparisons, not image-level paired
comparisons.

## Per-item status

| Item | Status | Evidence / limitation |
|---|---|---|
| S1 SD3.5 + Flux | COMPLETE | Zenodo DOI `10.5281/zenodo.22166280`; 60 fixed-seed prompt stems from each of the two dataset directories, 120 extracted images. |
| S2 Synthbuster | COMPLETE WITH LIMITATION | Zenodo DOI `10.5281/zenodo.10066460`; 30 fixed-seed common stems from each of nine generator directories, 270 extracted images. `prompts.csv` supplies a RAISE-1k image-name key, but the archive does not include the RAISE camera bytes. |
| K3 NPR | IMPLEMENTED / MEASURED | Training-free 2x2 relative-difference statistic, self-gated from the image, no learned weights. Overall AUC is `0.3417` on its combined Round 10 sample, so fusion weight remains zero. |
| K4 AEROBLADE | BLOCKED | Missing `models/taesd/encoder.onnx` and `models/taesd/decoder.onnx`. No permitted local weights or export toolchain was available. The detector returns NOT_APPLICABLE. |
| K5 MLEP | BLOCKED | The primary paper defines multi-scale shuffled LEP feature maps followed by a trained CNN, but this repo has no verified compatible ONNX checkpoint or output mapping. A handcrafted LEP scalar would not be MLEP, so it was not fabricated. |
| K6 CLIP probe | BLOCKED / UNCHANGED | No verified permissive local CLIP ONNX encoder plus trained probe is available; no torch was added. |

## Corpus and provenance

The manifest has 816 image rows: 400 IMD2020, 120 `sd35_flux`, 270
`synthbuster`, 12 `real_camera`, 12 `real_ai`, and 2 `real_c2pa_signed`.

S1 archive verification:

- size: `3,575,134,662` bytes
- MD5: `6a6125f7483e93108fa859c0cb8ebb20`
- licence: CC BY 4.0
- generator directory names: `FLUX.1-schnell`,
  `stable-diffusion-3.5-medium`
- manifest sample seed: `20260830`, 60 common stems per generator

S2 archive verification:

- size: `12,372,557,226` bytes
- MD5: `0695bd328e16ea21c5c9cc2ae1d994ff`
- licence: CC BY-NC-SA 4.0
- licence constraint: non-commercial use and share-alike apply; the whole
  corpus must not be described as CC BY
- generator directory names: `dalle2`, `dalle3`, `firefly`, `glide`,
  `midjourney-v5`, `stable-diffusion-1-3`, `stable-diffusion-1-4`,
  `stable-diffusion-2`, `stable-diffusion-xl`
- manifest sample seed: `20260830`, 30 common stems per generator

The generator field is copied from each archive directory. The S2 manifest
records `source_key` from the archive's `prompts.csv` and a `raise-1k/<stem>`
source group, but does not claim a paired real image. The download directory
passed `git check-ignore -v` before either archive was fetched, and
`git ls-files` contains no extracted corpus image or archive.

## Per-generator measurements

The S1 spectral/entropy/CFA/learned/AEROBLADE run used
`--sample 120 --seed 20260830`: 55 Flux images, 54 SD3.5 images, and 11 camera
negatives. The separate NPR run retained all 60 images per S1 generator and
12 camera negatives. S2 used all 30 preselected images per generator and 12
camera negatives because the manifest sample was already at the requested
30-per-generator floor. Every displayed standard error uses the displayed
positive and negative population.

### S1: `sd35_flux`

| Generator | spectral | entropy | cfa | learned | aeroblade | npr |
|---|---:|---:|---:|---:|---:|---:|
| `FLUX.1-schnell` | 0.5339 ± 0.0944 | 0.5884 ± 0.0905 | N/A | 0.4314 ± 0.0978 | N/A | 0.5347 ± 0.0902 |
| `stable-diffusion-3.5-medium` | 0.5286 ± 0.0948 | 0.4747 ± 0.0971 | N/A | 0.4024 ± 0.0979 | N/A | 0.2819 ± 0.0889 |

### S2: `synthbuster`

| Generator | spectral | entropy | cfa | learned | aeroblade | npr |
|---|---:|---:|---:|---:|---:|---:|
| `dalle2` | 0.5583 ± 0.0973 | 0.6611 ± 0.0890 | N/A | 0.4333 ± 0.1005 | N/A | 0.1222 ± 0.0685 |
| `dalle3` | 0.5722 ± 0.0965 | 0.2944 ± 0.0946 | N/A | 0.5028 ± 0.0997 | N/A | 0.3083 ± 0.0957 |
| `firefly` | 0.6056 ± 0.0941 | 0.5361 ± 0.0984 | N/A | 0.3306 ± 0.0972 | N/A | 0.5139 ± 0.0993 |
| `glide` | 0.9639 ± 0.0268 | 0.8111 ± 0.0670 | N/A | 0.3944 ± 0.0999 | N/A | 0.6889 ± 0.0858 |
| `midjourney-v5` | 0.6083 ± 0.0939 | 0.3694 ± 0.0991 | N/A | 0.3167 ± 0.0963 | N/A | 0.5222 ± 0.0990 |
| `stable-diffusion-1-3` | 0.5694 ± 0.0967 | 0.6806 ± 0.0868 | N/A | 0.2778 ± 0.0931 | N/A | 0.1417 ± 0.0730 |
| `stable-diffusion-1-4` | 0.5361 ± 0.0984 | 0.5861 ± 0.0956 | N/A | 0.3611 ± 0.0988 | N/A | 0.1389 ± 0.0724 |
| `stable-diffusion-2` | 0.5028 ± 0.0997 | 0.5528 ± 0.0976 | N/A | 0.2944 ± 0.0946 | N/A | 0.1056 ± 0.0643 |
| `stable-diffusion-xl` | 0.7000 ± 0.0844 | 0.5083 ± 0.0995 | N/A | 0.3833 ± 0.0996 | N/A | 0.2667 ± 0.0920 |

`cfa` is N/A because the downloaded AI images are PNG and the detector's
camera-EXIF applicability gate correctly abstains. `aeroblade` is N/A because
the external TAESD pair is absent, not because its score was zero.

The measured pattern is generator-specific. Spectral and entropy separate
`glide` strongly in this sample, while the learned face model is below chance
on both axes overall and NPR is below chance overall. No AUC floor was used and
no fusion parameter was tuned to improve these results.

## Files changed

- `data/corpus/MANIFEST.yaml`: Zenodo source metadata, licences, hashes,
  fixed-seed selections, generator names, and 390 sampled image rows.
- `scripts/fetch_corpus.py`: accepts the recorded CC BY-NC-SA 4.0 licence.
- `scripts/benchmark.py`: `--axes`, per-generator AUC/SE, source-negative
  scope, and generator-stratified sampling metadata.
- `backend/app/analysis/npr.py`, `backend/app/analysis/registry.py`, and
  `backend/app/analysis/calibration.json`: image-only NPR statistic with zero
  fusion weight.
- `backend/tests/test_corpus.py` and `backend/tests/test_npr.py`: benchmark
  SE and NPR contract checks.
- `docs/corpus.md`, `docs/detection-principles.md`, and
  `plan/reference/detector-catalog.yaml`: provenance, licence, scope, and
  limitation documentation.
- `plan/STATUS.yaml`: Round 10 status.

## Verification

Passed:

```text
.venv/bin/python scripts/fetch_corpus.py --check
816 manifest entries verified
.venv/bin/python -m pytest backend/tests/test_npr.py backend/tests/test_corpus.py backend/tests/test_contract.py -q
12 passed
```

The full repository test gate and vault validation completed cleanly:

```text
.venv/bin/python -m pytest -q
79 passed, 1 warning
vault Scripts/validate.py
notes: 163
problems: 0
vault Scripts/verify-quotes.py
quotes checked: 460 | too short to test: 12 | not found: 0
```
