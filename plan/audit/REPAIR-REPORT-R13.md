# Repair report R13: corpus parity investigation

Date: 2026-08-31  
Status: blocked for external replacement; the existing corpus also fails the parity gate.

## Result

No investigated external candidate is admissible as a replacement corpus in this
round. NTIRE has the strongest stated augmentation design, but its public
metadata has no licence and its unauthenticated sample service failed. The other
candidates are either licence-restricted, require an email or interactive
download, lack a two-class image sample, or are too large to probe safely.

All downloaded bytes and extracted probes were kept under `/tmp/r13-candidates`
or another `/tmp/r13-*` directory; no repository corpus directory was used and
no image bytes are untracked in the repository. The requested ignore check was
run against the actual temporary directory before closeout:
`git check-ignore -v /tmp/r13-candidates` returned Git's exact error that the
path is outside the repository. This is why no repository ignore rule was
needed or relied upon for these probes.

The reusable metadata gate is now [scripts/check_format_shortcut.py](../../scripts/check_format_shortcut.py).
It fits one deterministic threshold stump using only decoded container format,
width, height, file size, and EXIF presence. It uses a grouped 70/30 split and
reports Hanley-McNeil AUC standard error. `--check` exits non-zero when held-out
AUC exceeds 0.55. A future corpus must run this gate before bulk acquisition.

## Per-item status and files

| Item | Status |
|---|---|
| Metadata-only adversarial check | **Passed**: reusable script and regression test added; current corpus fails the gate |
| B: NTIRE 2026 | **Blocked**: no explicit licence, sample service HTTP 500, no generator field evidenced |
| C: WildFake | **Blocked**: licensed and labelled, but no safe two-class image sample below the size constraint |
| C: Chameleon | **Blocked**: email request and academic-only data terms |
| C: Fake2M | **Blocked**: accessible sample has one class only; commercial use needs written permission |
| C: ForenSynths | **Blocked**: restrictive licence and incomplete archive transfer |
| C: ELSA_D3 | **Blocked**: no explicit licence and no local two-class image sample |
| C: GenImage | **Blocked**: non-commercial terms and unauthenticated Drive file list unavailable |
| A: uniform re-encode fallback | **Measured, rejected**: fixed-square JPEG variant still fails metadata parity |

Repository files changed:

- `scripts/check_format_shortcut.py`
- `backend/tests/test_format_shortcut.py`
- `plan/audit/REPAIR-REPORT-R13.md`
- `plan/STATUS.yaml`

No detector implementation, calibration value, corpus manifest, or
`data/samples/` file was changed. The vault received one linked discovery note
outside the repository: `Discoveries/External corpus acceptance needs a metadata shortcut gate.md`.

## 1. Metadata shortcut gate

Command:

```text
.venv/bin/python scripts/check_format_shortcut.py --out /tmp/r13-format-native.json
```

The current corpus contains 390 generated rows and 12 strict camera rows. The
gate found:

| Population | AUC +/- SE | Positive | Negative |
|---|---:|---:|---:|
| train | 1.0000 +/- 0.0000 | 273 | 8 |
| held-out | 0.8750 +/- 0.0598 | 117 | 4 |
| pooled | 0.9583 +/- 0.0137 | 390 | 12 |

The selected feature was `width`, with threshold 3008 and generated images on
the lower-width side. The candidate feature set included format one-hot values,
width, height, file size, and EXIF presence. This is a measured shortcut, not an
inference from file suffixes.

The decoded-file audit also corrects an over-broad statement in the R12 report:
the generated set is not literally all PNG bytes. Its observed distribution is
330 generated PNG/no-EXIF, 30 generated PNG/EXIF, 30 generated JPEG/no-EXIF,
and 12 camera JPEG/EXIF. The gate uses the image header, so a `.png` suffix
cannot hide this distinction.

The regression test is:

```text
.venv/bin/python -m pytest backend/tests/test_format_shortcut.py -q
# 1 passed
```

## 2. External-corpus acceptance gates

`N/A (blocked)` means that an image-level two-class sample could not be obtained
under the licence, authentication, size, or transfer constraints. It is not an
AUC estimate.

| Candidate | Availability and size evidence | Licence/auth evidence | Per-image label evidence | Sample metadata AUC +/- SE | Parity result |
|---|---|---|---|---:|---|
| NTIRE 2026 | HF API: public, `gated:false`; 6 train shards totaling 114,410,515,725 bytes; individual shards 11.37-20.83 GB | HF `card_license:null`; no licence in official README; no auth wall | Official README documents `labels.csv` with binary 0/1 labels; no generator field was evidenced | N/A (blocked) | JPG and transformation claims are documented, but empirical format/resolution/encoding parity and generator labels were not verified |
| WildFake | ModelScope API HTTP 200; storage 1,291,478,056,101 bytes; smallest generated archive `DDIM.zip` 6,054,264,809 bytes | ModelScope reports Apache License 2.0; public, no auth wall | Downloaded `label_csv_files/dalle2.csv` (5,581,113 bytes), with `Generator`, `IsFake`, and `Image_path` columns | N/A (blocked) | Generator labels pass inspection; no two-class image sample was safely available, so parity is unverified |
| Chameleon | Official AIDE repository contains no dataset bytes; no public size | README says academic research only, commercial use prohibited; requests an email for the data | No image-level data accessible to inspect | N/A (blocked) | Not admissible under D7/D7b and not inspectable |
| Fake2M | HF API: public, `gated:false`; storage 717,209,124,328 bytes; complete Midjourney validation archive 1,214,683,777 bytes | Card says Apache 2.0 but commercial use requires official written permission | `Midjourneyv5-5K.csv` has 5,449 rows, all label `0`; generator is encoded by the archive/path | N/A (blocked) | Accessible shard contains generated positives only; no two-class parity result |
| ForenSynths | HF storage 96,641,839,586 bytes; `progan_val.zip` 830,792,545 bytes; main test zip 20,052,866,587 bytes | HF `card_license:cc-by-nc-sa-4.0`; public endpoint but not compatible with this repository's use | Official README documents generator folders and `0_real`/`1_fake` paths | N/A (blocked) | No valid image sample after transfer failure; licence independently blocks use |
| ELSA_D3 | HF API: public, `gated:false`; 2,631,650,634,984 bytes storage, about 2.568 TB download; first train parquet 479,879,983 bytes | HF `card_license:null`; no explicit licence in the card | Dataset-server rows expose `url` plus four `model_gen0..3` fields and generated image assets | N/A (blocked) | Pairing and model fields are evidenced, but no local two-class sample or licence was established |
| GenImage | Official page exposes Google Drive and Baidu downloads; unauthenticated file size was not exposed | Official `License` says CC BY-NC-SA 4.0/non-commercial terms; Drive page showed sign-in markers | README documents generator directories and `ai` versus `nature` paths | N/A (blocked) | No unauthenticated image sample; licence/auth constraints block use |

### NTIRE commands and results

```text
curl -fsSL https://huggingface.co/api/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train
# private:false, gated:false, disabled:false, usedStorage:114410515725, card_license:null

curl -fsSL 'https://huggingface.co/api/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train/tree/main?recursive=true' \
  | jq -r '.[] | [.path,.size] | @tsv'
# shard_0.zip 20589999903
# shard_1.zip 20833978590
# shard_2.zip 20447618786
# shard_3.zip 20615488046
# shard_4.zip 20500683037
# shard_5.zip 11370161676

curl -LsS -w '\nHTTP %{http_code}\n' \
  'https://datasets-server.huggingface.co/first-rows?dataset=deepfakesMSU%2FNTIRE-RobustAIGenDetection-train&config=default&split=train'
# HTTP 500
# Cannot load dataset split ... UnidentifiedImageError ... cannot identify image file ...
```

The first-rows service therefore did not produce usable image bytes. No NTIRE
shard was downloaded. The official sources are the [NTIRE paper](https://arxiv.org/abs/2604.11487),
[official repository](https://github.com/msu-video-group/NTIRE-2026-DeepFake-Detection),
and [Hugging Face dataset page](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train).

### WildFake commands and results

```text
curl -LsS 'https://www.modelscope.cn/api/v1/datasets/hy2628982280/WildFake'
# HTTP 200; License: Apache License 2.0; StorageSize: 1291478056101

curl -fsSL 'https://www.modelscope.cn/api/v1/datasets/hy2628982280/WildFake/repo/tree?Revision=master&Recursive=true&PageNumber=1&PageSize=3000' \
  | jq -r '.Data[] | [.Path,.Size] | @tsv' | rg 'DDIM.zip|DDPM.zip'
# Images/Diffusion_based/DDIM.zip 6054264809
# Images/Diffusion_based/DDPM.zip 8141353209

curl -fsSL 'https://www.modelscope.cn/api/v1/datasets/hy2628982280/WildFake/repo?Revision=master&FilePath=label_csv_files/dalle2.csv' \
  -o /tmp/r13-candidates/wildfake-dalle2.csv
head -2 /tmp/r13-candidates/wildfake-dalle2.csv
# Generator,Architecture,Weight,Category,IsAdvanced,IsFake,Image_path,Num
# Diffusion_based,DALLE,DALLE,DALLE,0,1,./Diffusion_based/DALLE/Typical/DALLE2/dalle/0.png,1
```

A direct request for `DDIM.zip` was stopped after a partial response because the
advertised file is 6,054,264,809 bytes. The partial transfer was not treated as
a sample and was removed. No archive integrity check passed. The official
sources are the [dataset repository](https://github.com/hy-zpg/AIGC-Image-Detection-Dataset)
and [ModelScope page](https://modelscope.cn/datasets/hy2628982280/WildFake/summary).

### Chameleon commands and results

```text
curl -fsSL https://raw.githubusercontent.com/shilinyan99/AIDE/main/README.md \
  | rg -n -C 2 'Chameleon|email|academic|commercial'
# Chameleon dataset ... only used for academic research. Commercial use ... prohibited.
# If you need Chameleon dataset, please send an email to tattoo.ysl@gmail.com.

curl -fsSL https://api.github.com/repos/shilinyan99/AIDE/git/trees/main?recursive=1 \
  | jq -r '.tree[].path' | rg -i 'chameleon|dataset|zip|tar'
# no Chameleon dataset archive or image shard
```

The [AIDE repository](https://github.com/shilinyan99/AIDE) has an MIT code
licence, but that does not change the separate academic-only data terms.

### Fake2M commands and results

```text
curl -fsSL https://huggingface.co/api/datasets/InfImagine/FakeImageDataset \
  | jq '{private,gated,disabled,usedStorage,card_license}'
# private:false, gated:false, disabled:false, usedStorage:717209124328, card_license:apache-2.0

curl --http1.1 -fsSL \
  'https://huggingface.co/datasets/InfImagine/FakeImageDataset/resolve/main/MetaData/val/Midjourneyv5-5K.csv' \
  -o /tmp/r13-candidates/fake2m-midjourney.csv
cut -d' ' -f2 /tmp/r13-candidates/fake2m-midjourney.csv | sort | uniq -c
# 5449 0

# Stream the first 200 members of the 1,214,683,777-byte archive, then build
# a JSONL manifest with label=true and run the reusable gate:
.venv/bin/python scripts/check_format_shortcut.py \
  --manifest /tmp/r13-candidates/fake2m-midjourney/sample.jsonl
# ValueError: need at least one authentic and one generated image
```

The accessible sample is 200 generated PNGs and has no authentic class. Its
failure is a useful blocked result, not an AUC. The [Sentry repository](https://github.com/Inf-imagine/Sentry)
and [dataset card](https://huggingface.co/datasets/InfImagine/FakeImageDataset)
provide the source and terms.

### ForenSynths commands and results

```text
curl -fsSL https://huggingface.co/api/datasets/sywang/CNNDetection \
  | jq '{private,gated,disabled,usedStorage,card_license}'
# private:false, gated:false, disabled:false, usedStorage:96641839586,
# card_license:cc-by-nc-sa-4.0

curl --http1.1 -fsSL \
  'https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip' \
  -o /tmp/r13-candidates/progan_val.zip
# curl: (92) HTTP/2 stream 1 was not closed cleanly: CANCEL (err 8)
unzip -t /tmp/r13-candidates/progan_val.zip
# End-of-central-directory signature not found
```

The fresh HTTP/1.1 retry stalled after about 77 MB and was terminated; no valid
image was extracted and the partial was removed. The [official repository](https://github.com/peterwang512/CNNDetection)
and [Hugging Face mirror](https://huggingface.co/datasets/sywang/CNNDetection)
document the folder labels and archive sizes.

### ELSA_D3 commands and results

```text
curl -fsSL https://huggingface.co/api/datasets/elsaEU/ELSA_D3 \
  | jq '{private,gated,disabled,usedStorage,card_license}'
# private:false, gated:false, disabled:false, usedStorage:2631650634984,
# card_license:null

curl -LsS -w '\nHTTP %{http_code}\n' \
  'https://datasets-server.huggingface.co/first-rows?dataset=elsaEU%2FELSA_D3&config=default&split=train'
# HTTP 200; rows include url, model_gen0..3, image_gen0..3 and dimensions
```

This proves row-level model fields and a real-image URL, but not that the real
and generated bytes arrive in matching containers or quality histories. The
[ELSA_D3 card](https://huggingface.co/datasets/elsaEU/ELSA_D3) has no explicit
licence, so D7b blocks download and use.

### GenImage commands and results

```text
curl -fsSL https://raw.githubusercontent.com/GenImage-Dataset/GenImage/main/License
# Unless specifically labeled otherwise ... CC BY-NC-SA 4.0 ... non-commercial purposes

curl -LsS -o /tmp/r13-genimage-drive.html -w '%{http_code} %{size_download}\n' \
  'https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS?usp=sharing'
# 200 361332
rg -n 'Sign in|Google Drive' /tmp/r13-genimage-drive.html
# sign-in markers present; no unauthenticated file list or file size
```

The official README provides `ai`/`nature` and generator-directory labels, but
the unauthenticated Drive response did not provide an image endpoint. No
GenImage bytes were downloaded. See the [official repository](https://github.com/GenImage-Dataset/GenImage)
and [homepage](https://genimage-dataset.github.io/).

Kaggle remains Tier B and was not probed because no credentials are present.
CIFAKE remains rejected because its 32x32 images cannot support the required
JPEG-block, CFA, SIFT, or spectral measurements.

## 3. Fallback Option A: measured, but rejected as a parity fix

Because every B/C path failed the admissible-sample gate, I measured the
fallback. This is a temporary variant under `/tmp/r13-parity-square`, not a
corpus change and not a write to `data/samples/`.

Pipeline: decode each of the same 402 rows to RGB, pad to a fixed 1024x1024
canvas with the same centered pipeline, save JPEG quality 90 with EXIF removed.
The output is 102,584,402 bytes and every file is JPEG with 1024x1024 decoded
dimensions.

An initial fixed-longest-side save still left held-out metadata AUC 0.9402
because aspect-ratio-derived dimensions differed. The fixed square save reduced
the selected shortcut to file size, but did not remove it:

| Population | AUC +/- SE | Selected feature |
|---|---:|---|
| train | 0.7617 +/- 0.0685 | file_size >= 218220 |
| held-out | 0.6571 +/- 0.1238 | file_size |
| pooled | 0.7282 +/- 0.0613 | file_size |

Therefore the variant fails the reusable 0.55 held-out gate. The file-size
shortcut is a residual content/encoding confound, not evidence that a parity
corpus has been achieved.

### Detector trade-off

These results compare the native files with the fixed-square parity attempt.
SE is the Hanley-McNeil estimate computed from the applicable scores. `n` is
the number of applicable rows; N/A means fewer than two labels among applicable
rows. This includes compression and provenance detectors, not only AI methods.

| Detector | Native AUC +/- SE | Square parity AUC +/- SE | Native n | Square n |
|---|---:|---:|---:|---:|
| aeroblade | 0.540 +/- 0.082 | 0.438 +/- 0.091 | 402 | 401 |
| c2pa | N/A | N/A | 30 | 0 |
| cfa | N/A | N/A | 12 | 0 |
| clip_probe | 1.000 +/- 0.000 | 1.000 +/- 0.001 | 402 | 402 |
| copy_move | 0.386 +/- 0.127 | 0.487 +/- 0.124 | 82 | 76 |
| double_jpeg | 0.192 +/- 0.082 | 0.219 +/- 0.080 | 42 | 402 |
| ela | 0.303 +/- 0.095 | 0.312 +/- 0.086 | 42 | 402 |
| entropy | 0.554 +/- 0.081 | 0.821 +/- 0.045 | 402 | 402 |
| exif | 0.083 +/- 0.058 | N/A | 42 | 0 |
| jpeg_ghosts | 0.417 +/- 0.100 | 0.302 +/- 0.086 | 42 | 402 |
| learned | 0.424 +/- 0.137 | 0.253 +/- 0.166 | 114 | 136 |
| npr | 0.342 +/- 0.087 | 0.270 +/- 0.084 | 402 | 402 |
| prnu | 0.588 +/- 0.078 | 0.637 +/- 0.073 | 402 | 402 |
| qtable | N/A | N/A | 12 | 0 |
| resampling | 0.298 +/- 0.094 | 0.211 +/- 0.079 | 360 | 402 |
| spectral | 0.602 +/- 0.077 | 0.501 +/- 0.085 | 402 | 402 |
| splicebuster | 0.720 +/- 0.110 | 0.797 +/- 0.050 | 35 | 402 |
| zero | 0.275 +/- 0.084 | 0.429 +/- 0.087 | 402 | 402 |

The uniform save removes EXIF and makes qtable N/A, while it activates the
JPEG-history detectors on generated rows. It also changes the scores of every
pixel/statistical detector. This is not a safe substitute for a corpus whose
native acquisition is already format-balanced.

### AI per-generator AUC

Each cell is `native -> square parity`, with AUC +/- SE. The negative scope is
the 12 strict camera rows, matching the existing benchmark contract. These are
measurements of the confounded corpus variants, not trustworthy AI-generation
claims.

#### SD3.5 and Flux

| Generator | AEROBLADE | CLIP | learned | NPR | entropy | PRNU | spectral | resampling | Splicebuster | ZERO |
|---|---|---|---|---|---|---|---|---|---|---|
| FLUX.1-schnell | 0.668 +/- 0.079 -> 0.500 +/- 0.095 | 1.000 +/- 0.000 -> 1.000 +/- 0.000 | 0.471 +/- 0.151 -> 0.246 +/- 0.172 | 0.535 +/- 0.090 -> 0.261 +/- 0.087 | 0.599 +/- 0.086 -> 0.968 +/- 0.019 | 0.642 +/- 0.082 -> 0.612 +/- 0.085 | 0.558 +/- 0.089 -> 0.454 +/- 0.093 | 0.500 +/- 0.099 -> 0.147 +/- 0.072 | N/A -> 0.828 +/- 0.054 | 0.315 +/- 0.091 -> 0.415 +/- 0.094 |
| stable-diffusion-3.5-medium | 0.536 +/- 0.090 -> 0.356 +/- 0.096 | 1.000 +/- 0.000 -> 1.000 +/- 0.000 | 0.410 +/- 0.149 -> 0.216 +/- 0.167 | 0.282 +/- 0.089 -> 0.103 +/- 0.062 | 0.506 +/- 0.092 -> 0.915 +/- 0.034 | 0.747 +/- 0.068 -> 0.785 +/- 0.062 | 0.546 +/- 0.090 -> 0.443 +/- 0.093 | 0.242 +/- 0.093 -> 0.115 +/- 0.065 | N/A -> 0.838 +/- 0.052 | 0.149 +/- 0.072 -> 0.383 +/- 0.093 |

#### Synthbuster

| Generator | AEROBLADE | CLIP | learned | NPR | entropy | PRNU | spectral | resampling | Splicebuster | ZERO |
|---|---|---|---|---|---|---|---|---|---|---|
| dalle2 | 0.603 +/- 0.094 -> 0.433 +/- 0.104 | 1.000 +/- 0.000 -> 1.000 +/- 0.000 | 0.533 +/- 0.165 -> 0.278 +/- 0.184 | 0.122 +/- 0.069 -> 0.150 +/- 0.075 | 0.661 +/- 0.089 -> 0.947 +/- 0.033 | 0.489 +/- 0.100 -> 0.654 +/- 0.090 | 0.558 +/- 0.097 -> 0.456 +/- 0.101 | 0.123 +/- 0.075 -> 0.044 +/- 0.043 | N/A -> 0.836 +/- 0.062 | 0.075 +/- 0.055 -> 0.381 +/- 0.100 |
| dalle3 | 0.500 +/- 0.100 -> 0.358 +/- 0.102 | 1.000 +/- 0.000 -> 1.000 +/- 0.000 | 0.467 +/- 0.159 -> 0.273 +/- 0.184 | 0.308 +/- 0.096 -> 0.156 +/- 0.076 | 0.294 +/- 0.095 -> 0.925 +/- 0.040 | 0.694 +/- 0.085 -> 0.828 +/- 0.064 | 0.572 +/- 0.096 -> 0.461 +/- 0.100 | 0.417 +/- 0.108 -> 0.214 +/- 0.086 | N/A -> 0.947 +/- 0.033 | 0.883 +/- 0.051 -> 0.314 +/- 0.096 |
| firefly | 0.547 +/- 0.098 -> 0.497 +/- 0.103 | 1.000 +/- 0.000 -> 1.000 +/- 0.000 | 0.433 +/- 0.159 -> 0.222 +/- 0.176 | 0.514 +/- 0.099 -> 0.342 +/- 0.098 | 0.536 +/- 0.098 -> 0.689 +/- 0.086 | 0.697 +/- 0.085 -> 0.483 +/- 0.100 | 0.606 +/- 0.094 -> 0.497 +/- 0.100 | 0.427 +/- 0.108 -> 0.297 +/- 0.095 | 0.720 +/- 0.110 -> 0.614 +/- 0.093 | 0.358 +/- 0.099 -> 0.461 +/- 0.100 |
| glide | 0.803 +/- 0.069 -> 0.755 +/- 0.079 | 1.000 +/- 0.000 -> 1.000 +/- 0.000 | N/A -> 0.267 +/- 0.185 | 0.689 +/- 0.086 -> 0.764 +/- 0.075 | 0.811 +/- 0.067 -> 0.978 +/- 0.021 | 0.111 +/- 0.066 -> 0.290 +/- 0.094 | 0.964 +/- 0.027 -> 0.908 +/- 0.045 | N/A -> 0.117 +/- 0.067 | N/A -> 0.878 +/- 0.053 | 0.281 +/- 0.093 -> 0.506 +/- 0.100 |
| midjourney-v5 | 0.636 +/- 0.091 -> 0.697 +/- 0.087 | 1.000 +/- 0.000 -> 0.997 +/- 0.007 | 0.383 +/- 0.158 -> 0.375 +/- 0.203 | 0.522 +/- 0.099 -> 0.536 +/- 0.098 | 0.369 +/- 0.099 -> 0.472 +/- 0.100 | 0.706 +/- 0.084 -> 0.619 +/- 0.093 | 0.608 +/- 0.094 -> 0.489 +/- 0.100 | 0.337 +/- 0.105 -> 0.511 +/- 0.099 | N/A -> 0.703 +/- 0.084 | 0.325 +/- 0.097 -> 0.592 +/- 0.095 |
| stable-diffusion-1-3 | 0.386 +/- 0.100 -> 0.206 +/- 0.088 | 1.000 +/- 0.000 -> 1.000 +/- 0.000 | N/A -> 0.111 +/- 0.132 | 0.142 +/- 0.073 -> 0.081 +/- 0.057 | 0.681 +/- 0.087 -> 0.997 +/- 0.007 | 0.350 +/- 0.098 -> 0.644 +/- 0.091 | 0.569 +/- 0.097 -> 0.500 +/- 0.100 | 0.190 +/- 0.089 -> 0.008 +/- 0.019 | N/A -> 0.892 +/- 0.049 | 0.164 +/- 0.077 -> 0.300 +/- 0.095 |
| stable-diffusion-1-4 | 0.397 +/- 0.100 -> 0.215 +/- 0.089 | 1.000 +/- 0.000 -> 1.000 +/- 0.000 | N/A -> 0.194 +/- 0.165 | 0.139 +/- 0.072 -> 0.083 +/- 0.058 | 0.586 +/- 0.096 -> 0.994 +/- 0.010 | 0.350 +/- 0.098 -> 0.613 +/- 0.094 | 0.536 +/- 0.098 -> 0.433 +/- 0.101 | 0.143 +/- 0.080 -> 0.006 +/- 0.015 | N/A -> 0.908 +/- 0.045 | 0.169 +/- 0.078 -> 0.342 +/- 0.098 |
| stable-diffusion-2 | 0.219 +/- 0.086 -> 0.242 +/- 0.093 | 1.000 +/- 0.000 -> 1.000 +/- 0.000 | 0.257 +/- 0.142 -> 0.242 +/- 0.178 | 0.106 +/- 0.064 -> 0.206 +/- 0.084 | 0.553 +/- 0.098 -> 0.442 +/- 0.101 | 0.675 +/- 0.087 -> 0.703 +/- 0.084 | 0.503 +/- 0.100 -> 0.372 +/- 0.099 | 0.144 +/- 0.081 -> 0.514 +/- 0.099 | N/A -> 0.508 +/- 0.100 | 0.158 +/- 0.076 -> 0.569 +/- 0.097 |
| stable-diffusion-xl | 0.519 +/- 0.099 -> 0.579 +/- 0.099 | 1.000 +/- 0.000 -> 1.000 +/- 0.000 | 0.556 +/- 0.164 -> 0.417 +/- 0.195 | 0.267 +/- 0.092 -> 0.464 +/- 0.100 | 0.508 +/- 0.100 -> 0.461 +/- 0.100 | 0.792 +/- 0.071 -> 0.656 +/- 0.090 | 0.700 +/- 0.084 -> 0.603 +/- 0.094 | 0.288 +/- 0.103 -> 0.511 +/- 0.099 | N/A -> 0.742 +/- 0.079 | 0.233 +/- 0.088 -> 0.519 +/- 0.099 |

The per-generator results confirm that the save pipeline itself is a material
intervention. For example, entropy rises from near chance to 0.915-0.997 on
the SD3.5/SD1.x rows, while AEROBLADE falls below chance on SD3.5/SD1.x. This
is why the fallback is recorded and rejected rather than used to rehabilitate
the AI measurement.

## 4. Fused metric and recommendation

The benchmark's fused held-out AUC is **0.5785 before and 0.5785 after**. The
calibration file was not refit for a temporary parity variant, so this equality
is expected and is not evidence that the variant preserves fusion behaviour.
No detector weight was tuned.

Recommendation: do not acquire or add any candidate yet. Keep the current
corpus as a clearly confounded native baseline, and next pursue NTIRE only after
the maintainers expose an explicit licence, a usable unauthenticated sample,
and generator labels or a documented generator mapping. Run this gate on that
sample before any archive download:

```text
.venv/bin/python scripts/check_format_shortcut.py --manifest sample.jsonl --check
```

Accept only a sample whose metadata-only held-out AUC is near chance and whose
container, resolution, and encoding distributions have been directly inspected.
