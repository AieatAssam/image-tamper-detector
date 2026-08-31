# Round 15C — byte-budget parity and controlled-encoding dataset survey

Date: 2026-08-31
Scope: 402 AI rows (`real_ai`, `sd35_flux`, `synthbuster`) plus 12 strict
`real_camera` negatives.  No corpus bytes were added to the repository.

## Result

`scripts/parity_encode.py` creates a deterministic 1024×1024 RGB JPEG variant
and binary-searches the highest integer JPEG quality whose encoded bytes do not
exceed the target.  The encoder uses optimized, non-progressive JPEG,
4:2:0 subsampling, and no EXIF.  A zero-byte tail after JPEG EOI fills the
remaining integer gap, so the file length is exactly the budget; Pillow and
OpenCV both decoded the output sample.

The reproducibility seed is `20260831`.  It controls deterministic sidecar
ordering; Pillow encoding and the canvas transform are deterministic.  The
trial used a 120,000-byte budget and 20,000-byte quality-search tolerance.  All
414 outputs were exactly 120,000 bytes, JPEG, 1024×1024, and EXIF-free.

The tail padding is deliberate but non-canonical.  Consumers must treat the
variant as the parity corpus representation, not as a byte-for-byte native
JPEG export.

Commands run:

```sh
.venv/bin/python scripts/parity_encode.py \
  --manifest data/corpus/MANIFEST.yaml \
  --out /tmp/r15c-parity-exact2 \
  --axes real_ai,sd35_flux,synthbuster,real_camera \
  --target-bytes 120000 --tolerance-bytes 20000 \
  --seed 20260831 --canvas-size 1024

for feature in all format dimensions file_size exif; do
  .venv/bin/python scripts/check_format_shortcut.py \
    --manifest /tmp/r15c-parity-exact2/manifest.jsonl \
    --features "$feature" --seed 20260831 --check \
    --out "/tmp/r15c-parity-exact2/gate-$feature.json"
done
```

## Metadata shortcut gate

Every full-corpus gate passed.  The checker’s 70/30 split and pooled result
were both exactly chance for every feature group:

| features | train AUC | held-out AUC | pooled AUC | result |
|---|---:|---:|---:|---|
| all | 0.500 | 0.500 | 0.500 | pass |
| format | 0.500 | 0.500 | 0.500 | pass |
| dimensions | 0.500 | 0.500 | 0.500 | pass |
| file size | 0.500 | 0.500 | 0.500 | pass |
| EXIF | 0.500 | 0.500 | 0.500 | pass |

The `all` pooled sample was 402 positive / 12 negative.  Its per-axis pooled
AUCs were also 0.500: `real_ai` (12/12), `sd35_flux` (120/12), and
`synthbuster` (270/12).  The file-size selector therefore chooses the constant
120,000-byte threshold rather than a label-separating threshold.  This removes
R14’s `file_size >= 218220` result (held-out AUC 0.6571).

## Quality distribution: the new forensic signal

Quality is the highest integer quality at or below the budget.  The exact
120,000-byte outputs produced this distribution:

| class | n | min | Q1 | median | Q3 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| AI-generated | 402 | 18 | 47 | 66 | 80 | 98 | 63.24 |
| authentic camera | 12 | 22 | 70 | 81.5 | 84 | 90 | 74.58 |

Counts by quality bucket (lower bound inclusive, upper bound exclusive except
for the final bucket):

| class | 10–19 | 20–29 | 30–39 | 40–49 | 50–59 | 60–69 | 70–79 | 80–89 | 90–99 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AI-generated | 2 | 19 | 43 | 52 | 47 | 62 | 72 | 59 | 46 | 0 |
| authentic camera | 0 | 1 | 0 | 0 | 0 | 1 | 3 | 6 | 1 | 0 |

The AI-axis means were `real_ai` 70.92 (n=12), `sd35_flux` 55.04 (n=120),
and `synthbuster` 66.54 (n=270).  Thus byte parity removes the size shortcut
but creates a class-correlated quality distribution: generated images need
lower quality to reach the same budget.  Quality is itself forensic evidence.

## Detector cost and native-vs-parity decision

Native values below are the R14 native baseline, including its Hanley–McNeil
standard error.  New parity values are pooled AI-vs-camera AUCs from the exact
120,000-byte files, with the same standard-error calculation.  `n` is the
number of applicable scored images.  `blocked` means the command was actually
run but stopped before serialization because the 1024² detector computation did
not complete; no value is fabricated.

| detector | R14 native AUC ± SE (n) | R15C exact parity AUC ± SE (n) | cost / observation |
|---|---:|---:|---|
| aeroblade | 0.540 ± 0.082 (402) | blocked: LPIPS load / run did not complete | optional LPIPS-backed path |
| c2pa | N/A (30) | N/A (0) | parity strips provenance |
| cfa | N/A (12) | N/A (0) | not applicable on parity |
| clip_probe | 1.000 ± 0.000 (402) | blocked: model path did not complete | optional model path |
| copy_move | 0.386 ± 0.127 (82) | 0.561 ± 0.117 (88) | +0.175 AUC |
| double_jpeg | 0.192 ± 0.082 (42) | 0.373 ± 0.088 (414) | +0.181; compression history changes |
| ela | 0.303 ± 0.095 (42) | blocked: 1024² run did not complete | compression-sensitive |
| entropy | 0.554 ± 0.081 (402) | blocked: local entropy run did not complete | slow rank-entropy path |
| exif | 0.083 ± 0.058 (42) | N/A (0) | parity intentionally removes EXIF |
| jpeg_ghosts | 0.417 ± 0.100 (42) | blocked: quality sweep did not complete | JPEG-history-sensitive |
| learned | 0.424 ± 0.137 (114) | blocked: model path did not complete | optional ONNX path |
| npr | 0.342 ± 0.087 (402) | blocked: 1024² run did not complete | 1024² statistic was too slow in the time window |
| prnu | 0.588 ± 0.078 (402) | blocked: 1024² run did not complete | noise cue needs a timed follow-up |
| qtable | N/A (12) | N/A (0) | qtable/provenance applicability disappears |
| resampling | 0.298 ± 0.094 (360) | 0.240 ± 0.082 (414) | −0.059 AUC |
| spectral | 0.602 ± 0.077 (402) | 0.508 ± 0.084 (414) | −0.094; near chance after parity |
| splicebuster | 0.720 ± 0.110 (35) | blocked: model path did not complete | manipulation-sensitive |
| zero | 0.275 ± 0.084 (402) | blocked: 64-grid run did not complete | compression/grid-sensitive |

Completed parity measurements came from isolated benchmark commands for
`copy_move,cfa,spectral,resampling,c2pa`, `double_jpeg`, `qtable`, `exif`,
and the other individual paths that completed.  The full all-detector command
was also attempted.  The reproducible blocking trace ended in
`ThreadPoolExecutor` waiting inside `run_all`; the launched batches included
LPIPS-backed AEROBLADE, optional model paths, ELA/entropy/JPEG-history work,
and ZERO’s 64-grid analysis.  This is a runtime limitation, not an assertion
that those detectors have a particular parity AUC.

Recommendation: add a separate `parity` corpus variant for the AI detectors
that are explicitly validated on AI axes, starting with spectral (its measured
AUC moves from 0.602 to 0.508).  Keep `native` for compression, provenance,
and manipulation-history detectors: ELA, double-JPEG, JPEG ghosts, qtable,
EXIF, ZERO, splicebuster, and C2PA.  Do not silently mix the two variants in a
single detector metric.  Resampling and copy-move also show material score
changes, so they should remain native until a detector-specific parity gate is
added.  The same per-detector scoping principle used by `VALIDATED_BY` and the
round-8c self-gates applies here.

### Later-round manifest proposal (not merged)

Do not write these rows into `data/corpus/MANIFEST.yaml` in R15C.  A later
round can merge one parity row for each of the 414 sidecar entries under a
parity path, retaining the native provenance fields:

```yaml
- id: ai_xai_aurora_001__parity
  path: data/corpus/parity/images/ai_xai_aurora_001.jpg
  corpus: parity
  variant: parity
  axis: real_ai
  label: ai_generated
  generator: xAI Aurora
  source_image: ai_xai_aurora_001
  native_path: data/corpus/real/ai_xai_aurora_001.jpg
  parity_quality: 91
  parity_file_size: 120000
  target_bytes: 120000

- id: cam_010__parity
  path: data/corpus/parity/images/cam_010.jpg
  corpus: parity
  variant: parity
  axis: real_camera
  label: authentic
  source_image: cam_010
  native_path: data/corpus/real/cam_010.jpg
  parity_quality: 73
  parity_file_size: 120000
  target_bytes: 120000
```

The proposed merge should be generated from the sidecar rather than hand
edited; the examples above are copied from measured rows, not invented values.

## Controlled-encoding dataset survey

The decisive gate is the existing `check_format_shortcut.py` metadata gate on
a small sample before bulk download.  AUC N/A means no small, both-class image
sample was publicly available through the inspected interface; the candidate
was not bulk downloaded.

| candidate | availability / auth | licence | published size | per-image generator labels | small-sample metadata gate |
|---|---|---|---:|---|---|
| [CommunityForensics-Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small) | HF API: `private=false`, `gated=false`, `disabled=false`; 200 image rows fetched through the public rows API | CC BY-NC-SA 4.0 | API `usedStorage=265,429,612,036` bytes | `label`, generated `model_name`, real `real_source`; observed generated model names and FFHQ real rows | **FAIL**: all pooled 1.000; format 0.500; dimensions 1.000; file size 1.000; EXIF 0.500 (100 AI + 100 real) |
| [DDA Training Set](https://huggingface.co/datasets/Junwei-Xi/DDA-Training-Set) | HF API public, ungated; only metadata/README inspected | Apache-2.0 | API/tree total 112,965,528,387 bytes: ten 10 GiB `.z01`–`.z10` parts plus 5,591,345,987-byte `.zip` | class/source construction is MS-COCO real plus DDA-aligned synthetic; no per-image generator field exposed in the inspected README | N/A: no small public image sample; no bulk bytes fetched |
| [AncesTree / QuAD](https://github.com/grip-unina/QuAD/tree/main/datasets/AncesTree) | GitHub metadata/CSV public without auth; image archives are direct download links | all rights reserved; informational/nonprofit only | published archives ~5 GB real, ~21 GB SD, ~18 GB commercial, ~7 GB other; 136,400 degraded images | CSV exposes `type`, `label`, `format`, `w`, `h`, `last_QF`, `current_QF`; generators include the documented source folders | N/A: no small public image sample; no archives fetched |
| [B-Free](https://grip-unina.github.io/B-Free/) | GitHub docs public; data URL connection failed twice (`curl` could not connect to `www.grip.unina.it:443`) | all rights reserved; informational/nonprofit only | not published in the inspected training README | folders identify COCO real and SD2.1 variants; no richer per-image generator field exposed | N/A: data host unavailable; no bytes fetched |
| [VIPCup 2022](https://grip-unina.github.io/vipcup2022/) | competition page requires a free account/class access code; evaluation sets explicitly not provided | not stated on the inspected official page | not stated | documented test categories include StyleGAN2/3, Gated Conv, GLIDE, Taming Transformers, but no downloadable per-image labels | N/A: no public both-class sample; no bytes fetched |
| [X-AIGD](https://huggingface.co/datasets/Coxy7/X-AIGD) | HF API: `private=false`, `gated=false`, `disabled=false`; fake images public; real set is metadata/reconstruction and exact set requires a non-commercial research request | CC BY 4.0 for released dataset; real-image access has separate restrictions | API `usedStorage=77,996,657,375` bytes | fake rows expose `generator`, `uid`; metadata exposes corresponding real `jpeg_quality` and `chroma_subsampling`, but real image bytes are not hosted | N/A: no small both-class image sample; no bulk bytes fetched |

Commands actually run for availability, size, and gates:

```sh
# CommunityForensics-Small: public metadata and 200-row sample
curl -fsSL 'https://huggingface.co/api/datasets/OwensLab/CommunityForensics-Small'
curl -fsSL 'https://datasets-server.huggingface.co/first-rows?dataset=OwensLab%2FCommunityForensics-Small&config=default&split=train'
for offset in 0 1000 5000 10000; do
  curl -fsSL "https://datasets-server.huggingface.co/rows?dataset=OwensLab%2FCommunityForensics-Small&config=default&split=train&offset=$offset&length=100"
done

for feature in all format dimensions file_size exif; do
  .venv/bin/python scripts/check_format_shortcut.py \
    --manifest /tmp/r15c-survey/community-sample/manifest.jsonl \
    --features "$feature" --seed 20260831 \
    --out "/tmp/r15c-survey/community-sample/gate-$feature.json"
done

# DDA: no bulk download; inspect API and file sizes first
curl -fsSL 'https://huggingface.co/api/datasets/Junwei-Xi/DDA-Training-Set'
curl -fsSL 'https://huggingface.co/api/datasets/Junwei-Xi/DDA-Training-Set/tree/main?recursive=true'

# AncesTree: public metadata and source-documented archive sizes/licence
curl -fsSL 'https://raw.githubusercontent.com/grip-unina/QuAD/main/datasets/AncesTree/README.md' -o /tmp/r15c-survey/ancestree-readme.md
curl -fsSL 'https://api.github.com/repos/grip-unina/QuAD/git/trees/main?recursive=1' -o /tmp/r15c-survey/quad-tree.json
curl -fsSL 'https://raw.githubusercontent.com/grip-unina/QuAD/main/datasets/AncesTree/AncesTree_test.csv' | head -3

# B-Free: public docs, then the advertised data host (both failed)
curl -fsSL https://grip-unina.github.io/B-Free/ -o /tmp/r15c-survey/bfree.html
curl -fsSL https://raw.githubusercontent.com/grip-unina/B-Free/main/training_data/README.md -o /tmp/r15c-survey/bfree-training-readme.md
curl --http1.1 -sSIL 'https://www.grip.unina.it/download/prog/B-Free/training_data/'
curl --http1.1 -sSIL 'https://www.grip.unina.it/download/prog/B-Free/training_data/COCO_real_512.zip'

# X-AIGD: public status and tree, metadata only
curl -fsSL 'https://huggingface.co/api/datasets/Coxy7/X-AIGD'
curl -fsSL 'https://huggingface.co/api/datasets/Coxy7/X-AIGD/tree/main?recursive=true'
```

### Survey conclusion

CommunityForensics-Small is publicly accessible and labelled, but its available
sample fails the decisive gate because the sampled AI rows are 512×512 and the
sampled FFHQ real rows are 1024×1024; it is not compression-matched by
construction.  DDA is the most promising licensing/format lead: its README
describes paired DDA-aligned real/synthetic data and lossless PNG, but the
released split archives are 112.97 GB and no small gate sample or per-image
generator labels were available.

AncesTree is the strongest encoding-control lead because its metadata records
the per-image degradation path and quality factors across real and AI trees,
but its all-rights-reserved, nonprofit-only terms block adoption without an
explicit licensing decision.  B-Free, VIPCup, and X-AIGD are useful leads but
were not admissible in this round: inaccessible/no sample, no downloadable
evaluation data, or real-image reconstruction restrictions respectively.

No candidate passed a decisive public small-sample gate in R15C.  Do not bulk
download any candidate until licensing and the gate are cleared.

## Validation

```sh
.venv/bin/python -m pytest backend/tests/test_parity_encode.py -q
# 3 passed

git diff --check
```
