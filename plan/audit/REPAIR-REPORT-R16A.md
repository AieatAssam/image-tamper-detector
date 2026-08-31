# Round 16A repair report: matched native/parity model evaluation

Date: 2026-09-01

## Status

| item | status | result |
|---|---|---|
| AEROBLADE model path | COMPLETE | TAESD and LPIPS loaded; 414/414 native and 414/414 parity rows scored; zero errors |
| CLIP probe model path | COMPLETE | local ViT-L/14 checkpoint and linear probe loaded; 414/414 native and 414/414 parity rows scored; zero errors |
| learned model path | COMPLETE | ONNX path scored all face-applicable rows with zero errors; session construction is now cached |
| parity measurement | COMPLETE | 402 AI and 12 authentic rows, same IDs in native and parity variants |

The R15C phrase “model path did not complete” was a runtime completion result,
not evidence that the weights were absent. A one-image probe loaded AEROBLADE
and CLIP; that particular image correctly returned `learned` as N/A because
the image-derived face gate found no face. The full isolated learned run then
loaded ONNX for every applicable face row and completed. The local model files
used were:

```
models/taesd/config.json
models/taesd/diffusion_pytorch_model.safetensors
models/lpips/checkpoints/alexnet-owt-7be5be79.pth
models/clip/open_clip_pytorch_model.safetensors
models/clip/linear_probe.npz
models/onnx/model_quantized.onnx
```

No model bytes were added to git.

## Measurement

The source was `/tmp/r15c-parity-exact2/manifest.jsonl`. Each row supplies the
native path and the parity path for the same source image. The parity copy is a
1024x1024 RGB JPEG at exactly 120,000 bytes, with no EXIF. The AUC calculation
is the existing Hanley-McNeil implementation in `scripts/benchmark.py`. Values
below are shown to three decimals; the JSON artifacts retain full precision.

For AEROBLADE and CLIP, every row is applicable. For `learned`, the face
precondition is evaluated independently on each image. Its AUC uses only rows
where the model is applicable, and `nAI`/`nR` are the applicable positive and
authentic counts. `N/A` means the generator did not have enough applicable
positive rows for the existing AUC contract.

| generator | AEROBLADE native | AEROBLADE parity | CLIP native | CLIP parity | learned native | learned parity |
|---|---:|---:|---:|---:|---:|---:|
| FLUX.1-schnell | 0.668 +/- 0.079 (nAI=60, nR=12) | 0.456 +/- 0.093 (nAI=60, nR=12) | 1.000 +/- 0.000 (nAI=60, nR=12) | 1.000 +/- 0.000 (nAI=60, nR=12) | 0.471 +/- 0.151 (nAI=17, nR=5) | 0.125 +/- 0.119 (nAI=18, nR=4) |
| Midjourney v4 | 0.333 +/- 0.264 (nAI=1, nR=12) | 0.167 +/- 0.173 (nAI=1, nR=12) | 1.000 +/- 0.000 (nAI=1, nR=12) | 1.000 +/- 0.000 (nAI=1, nR=12) | N/A | N/A |
| Stable Diffusion 3.5 Large | 1.000 +/- 0.000 (nAI=1, nR=12) | 0.583 +/- 0.320 (nAI=1, nR=12) | 1.000 +/- 0.000 (nAI=1, nR=12) | 1.000 +/- 0.000 (nAI=1, nR=12) | N/A | N/A |
| dalle2 | 0.603 +/- 0.094 (nAI=30, nR=12) | 0.389 +/- 0.100 (nAI=30, nR=12) | 1.000 +/- 0.000 (nAI=30, nR=12) | 1.000 +/- 0.000 (nAI=30, nR=12) | 0.533 +/- 0.165 (nAI=9, nR=5) | 0.175 +/- 0.141 (nAI=10, nR=4) |
| dalle3 | 0.500 +/- 0.100 (nAI=30, nR=12) | 0.325 +/- 0.097 (nAI=30, nR=12) | 1.000 +/- 0.000 (nAI=30, nR=12) | 1.000 +/- 0.000 (nAI=30, nR=12) | 0.467 +/- 0.159 (nAI=12, nR=5) | 0.208 +/- 0.148 (nAI=12, nR=4) |
| firefly | 0.547 +/- 0.098 (nAI=30, nR=12) | 0.481 +/- 0.100 (nAI=30, nR=12) | 1.000 +/- 0.000 (nAI=30, nR=12) | 1.000 +/- 0.000 (nAI=30, nR=12) | 0.433 +/- 0.159 (nAI=12, nR=5) | 0.205 +/- 0.148 (nAI=11, nR=4) |
| glide | 0.803 +/- 0.069 (nAI=30, nR=12) | 0.756 +/- 0.077 (nAI=30, nR=12) | 1.000 +/- 0.000 (nAI=30, nR=12) | 1.000 +/- 0.000 (nAI=30, nR=12) | N/A | 0.200 +/- 0.148 (nAI=10, nR=4) |
| midjourney-v5 | 0.636 +/- 0.091 (nAI=30, nR=12) | 0.647 +/- 0.090 (nAI=30, nR=12) | 1.000 +/- 0.000 (nAI=30, nR=12) | 1.000 +/- 0.000 (nAI=30, nR=12) | 0.383 +/- 0.158 (nAI=12, nR=5) | 0.250 +/- 0.162 (nAI=9, nR=4) |
| stable-diffusion-1-3 | 0.386 +/- 0.100 (nAI=30, nR=12) | 0.189 +/- 0.082 (nAI=30, nR=12) | 1.000 +/- 0.000 (nAI=30, nR=12) | 1.000 +/- 0.000 (nAI=30, nR=12) | N/A | 0.091 +/- 0.106 (nAI=11, nR=4) |
| stable-diffusion-1-4 | 0.397 +/- 0.100 (nAI=30, nR=12) | 0.181 +/- 0.080 (nAI=30, nR=12) | 1.000 +/- 0.000 (nAI=30, nR=12) | 1.000 +/- 0.000 (nAI=30, nR=12) | N/A | 0.150 +/- 0.133 (nAI=10, nR=4) |
| stable-diffusion-2 | 0.219 +/- 0.086 (nAI=30, nR=12) | 0.264 +/- 0.092 (nAI=30, nR=12) | 1.000 +/- 0.000 (nAI=30, nR=12) | 1.000 +/- 0.000 (nAI=30, nR=12) | 0.257 +/- 0.142 (nAI=14, nR=5) | 0.227 +/- 0.154 (nAI=11, nR=4) |
| stable-diffusion-3.5-medium | 0.536 +/- 0.090 (nAI=60, nR=12) | 0.307 +/- 0.091 (nAI=60, nR=12) | 1.000 +/- 0.000 (nAI=60, nR=12) | 1.000 +/- 0.000 (nAI=60, nR=12) | 0.410 +/- 0.149 (nAI=20, nR=5) | 0.131 +/- 0.120 (nAI=21, nR=4) |
| stable-diffusion-xl | 0.519 +/- 0.099 (nAI=30, nR=12) | 0.586 +/- 0.096 (nAI=30, nR=12) | 1.000 +/- 0.000 (nAI=30, nR=12) | 1.000 +/- 0.000 (nAI=30, nR=12) | 0.556 +/- 0.164 (nAI=9, nR=5) | 0.364 +/- 0.173 (nAI=11, nR=4) |
| xAI Aurora | 1.000 +/- 0.000 (nAI=1, nR=12) | 0.667 +/- 0.316 (nAI=1, nR=12) | 1.000 +/- 0.000 (nAI=1, nR=12) | 1.000 +/- 0.000 (nAI=1, nR=12) | N/A | N/A |

### Pooled native versus parity

| detector | native AUC +/- SE | parity AUC +/- SE | native applicable | parity applicable |
|---|---:|---:|---:|---:|
| AEROBLADE | 0.547 +/- 0.082 | 0.416 +/- 0.088 | 414 | 414 |
| CLIP probe | 1.000 +/- 0.001 | 1.000 +/- 0.001 | 414 | 414 |
| learned | 0.423 +/- 0.137 | 0.184 +/- 0.131 | 116 | 140 |

The unrounded CLIP values are 0.999793 +/- 0.000532 native and 0.999585
 +/- 0.000757 parity. It did not collapse on the specified parity corpus.

## Native/parity interpretation

The files are matched by source image, so this comparison does not confuse
generator mix with encoding variant.

### AEROBLADE

The raw LPIPS reconstruction distance changed as follows:

| label | native mean distance | parity mean distance | parity - native |
|---|---:|---:|---:|
| AI | 0.10653 | 0.09892 | -0.00760 |
| authentic | 0.12623 | 0.09366 | -0.03257 |

AEROBLADE maps lower distance to a higher suspicious score. Re-saving therefore
made authentic images look more like the TAESD reconstruction manifold than AI
images. The same-image score correlations were 0.880 for AI and 0.911 for
authentic images, but the authentic shift was larger. The AUC fall from 0.547
to 0.416 is consequently consistent with JPEG re-save damage to the cue, not
just disappearance of a format shortcut. Since the native result was already
near chance, this does not establish useful generation signal in either
variant.

### CLIP probe

Mean final scores changed from 0.8162 to 0.8093 for AI and from 0.0271 to
0.0393 for authentic images. The mean paired score delta was -0.0069 for AI
and +0.0121 for authentic images, while every generator remained at the
rounded 1.000 AUC in both variants.

This is not sufficient to call the CLIP result generation signal. The R15C
parity metadata gate covers format, dimensions, file size and EXIF, but the
encoder's recorded JPEG quality remains label-correlated: mean quality was
63.24 for AI versus 74.58 for authentic. A quality-only score, with lower
quality treated as AI evidence, has AUC 0.672 (the opposite direction is
0.328). Thus a JPEG quantization/compression cue that is visible to a frozen
image model remains plausible even though the four-feature gate is 0.500.
The evidence supports “CLIP survives the re-save,” not “CLIP has proven
generation signal.” Equalizing JPEG quality/history or using a common final
pipeline with matched quality is still required before accepting this number.

### learned

The face precondition yielded 116 applicable native rows (111 AI, 5 authentic)
and 140 applicable parity rows (136 AI, 4 authentic), with zero errors. There
were 91 rows applicable in both copies; 49 changed from not-applicable to
applicable and 25 changed in the other direction after re-encoding. This is an
image-derived Haar-cascade applicability change, not corpus membership.

The learned detector is below chance in both pooled measurements and is not a
general AI-generation detector. Its per-generator values are conditional on
the face gate; N/A is the honest result when a generator has no usable positive
sample count.

## Model-path fix

`backend/app/analysis/learned.py` now caches `onnxruntime.InferenceSession`
by model path with `lru_cache(maxsize=2)`. This does not alter preprocessing,
model outputs, calibration, ranking, or applicability. The new test proves a
session is constructed once for repeated calls. AEROBLADE and CLIP did not
need module changes: their local model paths loaded successfully in the
isolated full measurements.

## Commands and artifacts

The parity/native manifests used for the runs were temporary absolute-path
JSONL copies at:

```
/tmp/r16a-measurements/native.jsonl
/tmp/r16a-measurements/parity.jsonl
```

Measurement artifacts:

```
/tmp/r16a-measurements/aeroblade-native.json
/tmp/r16a-measurements/aeroblade-parity.json
/tmp/r16a-measurements/clip_probe-batched.json
/tmp/r16a-measurements/learned-direct-cached.json
```

The isolated native AEROBLADE command was:

```
/usr/bin/time -p .venv/bin/python scripts/benchmark.py \
  --out /tmp/r16a-measurements/aeroblade-native.json \
  --corpus matched \
  --matched-manifest /tmp/r16a-measurements/native.jsonl \
  --detectors aeroblade --profile
```

It completed in 269.67 seconds. The parity AEROBLADE run used the same
`scripts.benchmark.run` implementation from an inline Python invocation with
the temporary parity manifest and completed in 1006.99 seconds. The CLIP
measurement used batch size 4 but the same local `_load_backbone`, probe,
preprocessing, frozen `encode_image`, sigmoid and calibrated score mapping;
both variants completed in 245.41 seconds. The learned direct measurement
called `LearnedDetector.run` for every row in both variants and completed in
75.31 seconds.

The current sibling-owned benchmark variant plumbing also has an independent
CLI error when asked to run these matched rows:

```
TypeError: unsupported operand type(s) for -: 'set' and 'dict'
  at scripts/benchmark.py:381
```

This round did not modify `scripts/benchmark.py`; direct detector/module
measurement was used for the native-versus-parity comparison instead.

## Files changed

- `backend/app/analysis/learned.py`: cache the ONNX session.
- `backend/tests/test_learned.py`: cache regression test.
- `plan/audit/REPAIR-REPORT-R16A.md`: this report.

`aeroblade.py`, `clip_probe.py`, and their tests were unchanged because their
model paths completed without a code change.

## Verification

Focused detector tests passed:

```
10 passed in 3.65s
```

The full backend suite ran to completion with 93 passed and two unrelated
failures. `test_corpus.py::test_per_generator_auc_reports_standard_error`
expects rows without the new sibling-owned `variant` field. The other failure,
`test_entropy.py::test_detect_ai_generated_original_images`, observed
`matching proportion 0.11897476665446559` against threshold `0.35`. Neither
failure is in this round's allowed files, and no attempt was made to weaken
either check.

No commit was made.
