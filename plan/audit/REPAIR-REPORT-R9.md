# Round 9 repair report - 2026-08-30

Status: blocked at the corpus gate. No detector implementation, model weight,
image byte, Parquet byte, or ZIP byte was added. The existing twelve `real_ai`
manifest entries remain unchanged and do not form generated/real source pairs.

## K1 - ELSA1M corpus

Status: BLOCKED.

Files changed: none.

Evidence and commands:

```text
git check-ignore -v --no-index data/corpus/real/elsa1m
.gitignore:8:data/corpus/real/* data/corpus/real/elsa1m

bounded ELSA ranged model-column probes:
0   rows=1113 row_groups=1 models=['stabilityai/stable-diffusion-2-1-base']
445 rows=1113 row_groups=1 models=['stabilityai/stable-diffusion-2-1-base']
891 rows=1112 row_groups=1 models=['stabilityai/stable-diffusion-2-1-base']
```

The source card is `https://huggingface.co/datasets/elsaEU/ELSA1M_track1`.
Its recorded `model` field did not provide the required six distinct
generators in the bounded probes. No image or Parquet bytes were fetched.

Measurements: classical corpus AUC/SE: `null` / `null`, no K1 detector was
run. AI corpus within-source AUC/SE: `null` / `null`, paired positives 0 and
paired negatives 0.

## K2 - GenImage fallback

Status: BLOCKED.

Files changed: none.

Evidence and commands:

```text
git check-ignore -v --no-index data/corpus/real/genimage
.gitignore:8:data/corpus/real/* data/corpus/real/genimage

curl --fail --silent --show-error --max-time 30 \
  https://huggingface.co/api/datasets/ENSTA-U2IS/GenImage/tree/main?recursive=true&expand=false
curl --fail --silent --show-error --max-time 30 \
  https://raw.githubusercontent.com/GenImage-Dataset/GenImage/main/Readme.md
```

The source card is Apache-2.0 and not gated. The API lists eight generator
directories and multi-gigabyte split archive parts. The official README says
the benchmark contains fake/real pairs, but the published archive layout is
separate `train/ai`, `train/nature`, `val/ai`, and `val/nature` trees. A ranged
central-directory read of the ADM archive found generated names such as
`0_adm_0.PNG` and real names such as `n01440764_10183.JPEG`; no metadata-like
member or common image-level key was present. Matching them by order, class,
or filename would invent source groups, so no GenImage bytes or manifest rows
were added.

Measurements: classical corpus AUC/SE: `null` / `null`, no K2 detector was
run. AI corpus within-source AUC/SE: `null` / `null`, paired positives 0 and
paired negatives 0.

## K3 - NPR

Status: BLOCKED by K1/K2. The requested training-free statistic was not added
because the required AI validation population could not be constructed.

Files changed: none.

Measurements: classical corpus AUC/SE: `null` / `null`, no NPR implementation
exists to benchmark. AI corpus within-source AUC/SE: `null` / `null` because
there are no valid generated/real source pairs.

The paper-only source was reviewed at
`https://arxiv.org/abs/2312.10461`; no unlicensed upstream code or classifier
weights were used.

## K4 - AEROBLADE

Status: BLOCKED. The existing adapter remains unchanged and its TAESD weights
are absent. The exact missing facts are the external encoder and decoder at
`models/taesd/encoder.onnx` and `models/taesd/decoder.onnx`; no export was
produced because the source-balanced AI corpus was already blocked and the
repository does not contain a permissible export toolchain.

Commands tried:

```text
if test -f models/taesd/encoder.onnx && test -f models/taesd/decoder.onnx; then echo present; else echo absent; fi
absent
find models -type f -name '*.onnx'
models/onnx/model_quantized.onnx
```

Measurements: classical corpus AUC/SE: `null` / `null`; AI corpus
within-source AUC/SE: `null` / `null`.

## K5 - MLEP versus entropy

Status: BLOCKED by K1/K2. MLEP was not added and `entropy` was not deleted:
without a valid paired AI corpus neither the replacement decision nor a fair
within-source comparison can be made. The existing entropy result remains a
negative historical finding, not Round 9 MLEP evidence.

Files changed: none.

Measurements: classical corpus AUC/SE: `null` / `null` for MLEP; AI corpus
within-source AUC/SE: `null` / `null` for both MLEP and a new entropy
comparison.

The paper-only source reviewed was
`https://arxiv.org/abs/2504.13726`. No classifier source or weights were copied.

## K6 - CLIP probe

Status: BLOCKED by K1/K2 and missing local encoder/probe artifacts. No
`clip_probe.py` was added. The exact missing facts are a permissively licensed
ONNX CLIP image encoder and a paired K1 training sample; the installed
optional requirements contain ONNX Runtime but no CLIP encoder or fitted probe.

Commands tried:

```text
rg --files backend/app/analysis models | rg 'clip|onnx'
models/onnx/model_quantized.onnx
.venv/bin/python -c 'import onnxruntime; print(onnxruntime.__version__)'
1.29.0
```

No torch was installed or added.

Measurements: classical corpus AUC/SE: `null` / `null`, no K6 probe exists.
AI corpus within-source AUC/SE: `null` / `null` because K1/K2 supplied no
pairs.

## Verification

The required commands were run after the report and status updates:

```text
.venv/bin/python -m pytest backend/tests -q
.venv/bin/python scripts/benchmark.py --out /tmp/post-r9.json --corpus all
.venv/bin/python scripts/calibrate.py --corpus all --out backend/app/analysis/calibration.json --seed 20260828
.venv/bin/python plan/validate.py
git status --porcelain | grep -E '\\.(jpg|jpeg|png|zip|parquet)$'
```

Results:

```text
pytest: 77 passed, 1 warning in 329.60s
benchmark: interrupted with exit 130 after the all-detector loop reached
  backend/app/analysis/registry.py:71; /tmp/post-r9.json was not written.
  The process repeatedly emitted macOS Cache.db SQLite result=8 warnings.
calibrate: interrupted with exit 130 at scripts/calibrate.py:358 after the
  same detector loop emitted the same Cache.db warnings; calibration.json was
  not rewritten.
plan/validate.py: All structural and shell-syntax checks passed.
image-byte grep: no matching output; exit 1 is grep's empty-result status.
```

The benchmark and calibration commands were attempted exactly as requested,
but the runtime stall prevented new AUC/SE artifacts. The last command had no
image/archive output; corpus bytes are never committed.
