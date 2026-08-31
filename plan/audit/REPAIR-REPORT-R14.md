# Repair report R14: WildFake acceptance gate

Date: 2026-08-31  
Status: WildFake rejected by the metadata shortcut gate; no new corpus axis was ingested.

## Outcome

WildFake is available and licensed, but it does not fix the measurement
problem. A fixed-seed, two-class sample of 400 actual images gives metadata-only
held-out AUC **1.0000 +/- 0.0000** and pooled AUC **1.0000 +/- 0.0000**. The
single selected feature is decoded `format=JPEG`, with JPEG authentic
CelebA-HQ rows and PNG generated DDIM rows.

Per the acceptance rule, acquisition stopped after the gate. No WildFake row
was added to `data/corpus/MANIFEST.yaml`, no calibration was run, and no image
bytes are committed. The downloaded archive and sample remain in the
gitignored `data/corpus/real/r14-wildfake-download/` directory for local audit.

## Per-item status and files

| Item | Status |
|---|---|
| WildFake availability/licence/auth | **Passed**: public ModelScope API, Apache License 2.0, no auth wall |
| WildFake label evidence | **Passed**: CSV supplies `Generator`, `IsFake`, and `Image_path` |
| WildFake two-class sample | **Passed**: 200 DDIM generated plus 200 CelebA-HQ authentic images extracted and integrity-checked |
| WildFake metadata gate | **Failed**: held-out and pooled AUC are both 1.0000 +/- 0.0000 |
| Residual leak ablation | **Measured**: file size is the only residual signal in the fixed-square fallback |
| Standing corpus acceptance check | **Passed**: feature groups and per-axis reporting added to the reusable script; docs now make it mandatory |

Repository files changed:

- `scripts/check_format_shortcut.py`
- `backend/tests/test_format_shortcut.py`
- `docs/corpus.md`
- `docs/detection-principles.md`
- `plan/audit/REPAIR-REPORT-R14.md`
- `plan/STATUS.yaml`

No detector implementation, calibration file, corpus manifest, or
`data/samples/` file changed.

## 1. Ignored download preflight and source evidence

Before creating the download directory or fetching bytes:

```text
git check-ignore -v data/corpus/real/r14-wildfake-download
# .gitignore:8:data/corpus/real/* data/corpus/real/r14-wildfake-download
```

The source tree and label CSV were inspected before the archive fetch:

```text
curl --http1.1 -sSIL \
  'https://www.modelscope.cn/api/v1/datasets/hy2628982280/WildFake/repo?Revision=master&FilePath=Images/Diffusion_based/DDIM.zip'
# final Content-Length: 6054264809

curl --http1.1 -sSIL \
  'https://www.modelscope.cn/api/v1/datasets/hy2628982280/WildFake/repo?Revision=master&FilePath=Images/Real/celebahq.zip'
# final Content-Length: 350991722

curl --http1.1 -fsSL \
  'https://www.modelscope.cn/api/v1/datasets/hy2628982280/WildFake/repo?Revision=master&FilePath=label_csv_files/ddim.csv' \
  -o data/corpus/real/r14-wildfake-download/ddim.csv
curl --http1.1 -fsSL \
  'https://www.modelscope.cn/api/v1/datasets/hy2628982280/WildFake/repo?Revision=master&FilePath=label_csv_files/real_celebahq.csv' \
  -o data/corpus/real/r14-wildfake-download/real_celebahq.csv
head -4 data/corpus/real/r14-wildfake-download/ddim.csv
# Generator,Architecture,Weight,Category,IsAdvanced,IsFake,Image_path,Num
# Diffusion_based,DDIM,DDIM,DDIM,0,1,./Diffusion_based/DDIM/imgs_CC9K/00071f9a7d2572ac58a3b0529c695718.png,1
head -4 data/corpus/real/r14-wildfake-download/real_celebahq.csv
# Generator,Architecture,Weight,Category,IsAdvanced,IsFake,Image_path,Num
# Real,celebahq,celebahq,celebahq,0,0,./Real/celebahq/data1024x1024/img000000.jpg,1
```

The authorized downloads completed at exactly the advertised sizes:

| File | Bytes | SHA-256 |
|---|---:|---|
| `DDIM.zip` | 6,054,264,809 | `fa509e0ae546d91b2edd6dad91a1efe0ae3bd5c50d31609cb1db56a31d9f6e9c` |
| `celebahq.zip` | 350,991,722 | `bfc71b04c16786267781110c52b36c515558c8889ed657cf8d89b629b531` |

Both archives passed `unzip -t` with `No errors detected in compressed data`.
The [WildFake repository](https://github.com/hy-zpg/AIGC-Image-Detection-Dataset)
and [ModelScope dataset page](https://modelscope.cn/datasets/hy2628982280/WildFake/summary)
are the recorded source and licence evidence.

## 2. WildFake sample gate

The selected sample is reproducible from the existing CSVs and archive
structure:

- seed `20260831`;
- 200 rows sampled from the `ddim.csv` rows, with generator copied as
  `Diffusion_based/DDIM` and `IsFake=1`;
- 200 rows sampled from `real_celebahq.csv`, with generator copied as
  `Real/celebahq` and `IsFake=0`;
- the actual archive members were extracted by basename and recorded in the
  ignored `sample-axis.jsonl` sidecar.

The gate was run before any manifest edit:

```text
.venv/bin/python scripts/check_format_shortcut.py \
  --manifest data/corpus/real/r14-wildfake-download/sample-axis.jsonl \
  --out /tmp/r14-wildfake-format.json
```

Result:

| Population | AUC +/- SE | Positive | Negative | Selected feature |
|---|---:|---:|---:|---|
| train | 1.0000 +/- 0.0000 | 140 | 140 | `format=JPEG`, threshold 1, JPEG is predicted authentic |
| held-out | 1.0000 +/- 0.0000 | 60 | 60 | same |
| pooled | 1.0000 +/- 0.0000 | 200 | 200 | same |

The script also reports `per_axis.wildfake` with the same 1.0000 held-out and
pooled AUC. This is a decisive format shortcut, so the required action is to
stop. WildFake has not been represented as a new axis and no WildFake detector
AUC is reported.

## 3. Residual leak ablation

The ablation targets the R13 fixed-square fallback: all 402 existing rows were
decoded, centered-padded to 1024x1024, saved as JPEG quality 90 with EXIF
removed, and evaluated without refitting any detector. The gate now accepts
`--features all|format|dimensions|file_size|exif`.

```text
for feature in all format dimensions file_size exif; do
  .venv/bin/python scripts/check_format_shortcut.py \
    --manifest /tmp/r13-parity-square/manifest.jsonl \
    --features "$feature" \
    --out "/tmp/r14-r13-${feature}.json"
done
```

| Feature group | Held-out AUC +/- SE | Pooled AUC +/- SE | Selected feature | Gate |
|---|---:|---:|---|---|
| all | 0.6571 +/- 0.1238 | 0.7282 +/- 0.0613 | file size | fail |
| format only | 0.5000 +/- 0.1474 | 0.5000 +/- 0.0847 | format is constant JPEG | pass |
| dimensions only | 0.5000 +/- 0.1474 | 0.5000 +/- 0.0847 | dimensions are constant 1024x1024 | pass |
| file size only | 0.6571 +/- 0.1238 | 0.7282 +/- 0.0613 | `file_size >= 218220` | fail |
| EXIF presence only | 0.5000 +/- 0.1474 | 0.5000 +/- 0.0847 | EXIF is absent everywhere | pass |

This isolates the residual leak in the tested fallback: file size is still
label-predictive after format, dimensions, and EXIF are equalized. It does not
prove that file size is the only possible leak in another corpus; it proves that
this save pipeline did not equalize it. A later parity pipeline must measure
file size as well as container, dimensions, and EXIF.

For comparison, the same ablation on the WildFake sample is:

| Feature group | Held-out AUC +/- SE | Pooled AUC +/- SE |
|---|---:|---:|
| all | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| format only | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| dimensions only | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| file size only | 1.0000 +/- 0.0000 | 0.9975 +/- 0.0025 |
| EXIF presence only | 0.5000 +/- 0.0529 | 0.5000 +/- 0.0289 |

WildFake therefore fails even if the format feature is the only feature
allowed. Its dimensions and file sizes also separate the two classes.

## 4. Standing acceptance wiring and caveat

`scripts/check_format_shortcut.py` now has two standing behaviours:

1. the default current-corpus check remains the reproducible 390 generated
   `sd35_flux`/`synthbuster` rows against 12 strict camera negatives;
2. `--manifest sample.jsonl --check` accepts a future candidate sample before
   manifest ingestion, reports `per_axis`, and fails if the overall or any
   axis held-out AUC exceeds 0.55.

The feature-group switch makes the check diagnostic, not merely pass/fail.
The regression test is:

```text
.venv/bin/python -m pytest backend/tests/test_format_shortcut.py -q
# 2 passed
```

`docs/corpus.md` and `docs/detection-principles.md` now state prominently that:

- Round 10's per-generator AI table is metadata-confounded and exploratory;
- Round 12's CLIP `1.0000 +/- 0.0000` seen and unseen result is not generation
  evidence;
- Round 14 WildFake `1.0000 +/- 0.0000` is an independent failed corpus gate;
- no AI-generation AUC should be presented as generalization evidence until its
  candidate axis passes the metadata-only gate.

The existing R13 fused held-out AUC remains `0.5784615384615385`; no calibration
or fusion weight was changed in this round.
