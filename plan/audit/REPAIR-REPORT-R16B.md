# Round 16B — slow classical detectors

Run context: the R15C exact-parity corpus, reconstructed from the existing
manifest and R15C command under `/tmp/r15c-parity-exact2`. No corpus bytes or
measurement JSON were added to the repository.

## Result

All five previously incomplete detectors completed all 414 R15C parity rows
with zero detector errors. Hanley–McNeil standard errors are the
`scripts/benchmark.py::_auc_stats` calculation, called from a temporary
measurement wrapper because the matched benchmark path currently fails before
execution with `TypeError: unsupported operand type(s) for -: 'set' and 'dict'`
at its `set(selected_ids) - VARIANT_SCOPES` scope check. `scripts/benchmark.py`
was not changed.

The inherited R15C metadata shortcut gate was exactly chance: pooled AUC
0.500 for `all`, `format`, `dimensions`, `file_size`, and `EXIF` (train and
held-out AUC were also 0.500). R16B did not create a new corpus axis or rerun
that gate.

| detector | parity rows | applicable | pooled AUC ± Hanley–McNeil SE | stable mean-duration bucket |
|---|---:|---:|---:|---:|
| ELA | 414 | 414 | 0.3601 ± 0.0875 | 500 ms |
| NPR | 414 | 414 | 0.2803 ± 0.0846 | 500 ms |
| PRNU / Noisesniffer | 414 | 414 | 0.6490 ± 0.0719 | 3500 ms |
| entropy | 414 | 414 | 0.7616 ± 0.0559 | 2500 ms |
| JPEG ghosts | 414 | 414 | 0.5220 ± 0.0834 | 1000 ms |

The parity corpus contains 402 AI-generated rows and 12 authentic camera rows:
`real_ai` 12, `sd35_flux` 120, `synthbuster` 270, and `real_camera` 12.
Per-generator negatives are the 12 `real_camera` rows when no
source-matched authentic row exists. One-image generators necessarily have
large standard errors.

| axis | generator | positive | negative | ELA | NPR | PRNU | entropy | JPEG ghosts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| real_ai | Midjourney v4 | 1 | 12 | 0.5833 ± 0.3199 | 0.1667 ± 0.1725 | 0.0833 ± 0.1084 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| real_ai | Stable Diffusion 3.5 Large | 1 | 12 | 0.5833 ± 0.3199 | 0.1667 ± 0.1725 | 0.1667 ± 0.1725 | 0.9167 ± 0.1996 | 0.7500 ± 0.2988 |
| real_ai | xAI Aurora | 1 | 12 | 0.2500 ± 0.2236 | 1.0000 ± 0.0000 | 0.2500 ± 0.2236 | 1.0000 ± 0.0000 | 0.1667 ± 0.1725 |
| sd35_flux | FLUX.1-schnell | 60 | 12 | 0.3181 ± 0.0911 | 0.2583 ± 0.0869 | 0.5986 ± 0.0857 | 0.9653 ± 0.0198 | 0.5854 ± 0.0868 |
| sd35_flux | stable-diffusion-3.5-medium | 60 | 12 | 0.3722 ± 0.0932 | 0.1000 ± 0.0612 | 0.7819 ± 0.0623 | 0.7639 ± 0.0653 | 0.6306 ± 0.0828 |
| synthbuster | dalle2 | 30 | 12 | 0.3056 ± 0.0955 | 0.2000 ± 0.0836 | 0.6500 ± 0.0901 | 0.9250 ± 0.0400 | 0.4181 ± 0.1004 |
| synthbuster | dalle3 | 30 | 12 | 0.3278 ± 0.0970 | 0.1500 ± 0.0747 | 0.8111 ± 0.0670 | 0.9000 ± 0.0470 | 0.5181 ± 0.0992 |
| synthbuster | firefly | 30 | 12 | 0.3778 ± 0.0994 | 0.3583 ± 0.0986 | 0.5611 ± 0.0972 | 0.7139 ± 0.0826 | 0.5236 ± 0.0990 |
| synthbuster | glide | 30 | 12 | 0.3278 ± 0.0970 | 0.7500 ± 0.0774 | 0.3000 ± 0.0950 | 0.9972 ± 0.0071 | 0.2153 ± 0.0858 |
| synthbuster | midjourney-v5 | 30 | 12 | 0.2694 ± 0.0923 | 0.5528 ± 0.0976 | 0.6528 ± 0.0898 | 0.4139 ± 0.1003 | 0.5472 ± 0.0979 |
| synthbuster | stable-diffusion-1-3 | 30 | 12 | 0.5667 ± 0.0968 | 0.0694 ± 0.0531 | 0.7611 ± 0.0757 | 0.9111 ± 0.0440 | 0.5931 ± 0.0951 |
| synthbuster | stable-diffusion-1-4 | 30 | 12 | 0.4389 ± 0.1005 | 0.0667 ± 0.0521 | 0.7139 ± 0.0826 | 0.9333 ± 0.0375 | 0.5389 ± 0.0983 |
| synthbuster | stable-diffusion-2 | 30 | 12 | 0.3667 ± 0.0990 | 0.2472 ± 0.0899 | 0.6639 ± 0.0887 | 0.2722 ± 0.0926 | 0.4736 ± 0.1003 |
| synthbuster | stable-diffusion-xl | 30 | 12 | 0.2694 ± 0.0923 | 0.4750 ± 0.1003 | 0.6611 ± 0.0890 | 0.3611 ± 0.0988 | 0.5208 ± 0.0991 |

## Profile-first diagnosis

The profiles below were captured before edits with `cProfile` on one 1024²
JPEG and one 4608×3456 native camera JPEG. Times are wall seconds for the
detector call; the indented figure is the measured cumulative hotspot.

| detector | 1024² profile | native profile at the effective old bound | measured bottleneck |
|---|---:|---:|---|
| ELA | 0.312 s; compression-artifact loop 0.172 s | 0.803 s at 2000 px; loop 0.497 s | Python loop over 32,260 block-boundary means |
| NPR | 0.133 s; `measure` 0.132 s | 0.249 s at shared 1600 px | four-way overlapping-patch `stack` and variance reductions |
| PRNU | 1.821 s; `_channel_candidates` 1.740 s | 3.665 s at shared 1600 px; candidates 3.496 s | three sliding-window candidate passes; repeated `argsort` was 0.736 s at 1600 |
| entropy | 1.119 s; rank entropy 1.061 s | 1.933 s at shared 1600 px; rank entropy 1.788 s | three `skimage.filters.rank.entropy` calls |
| JPEG ghosts | 0.323 s | 0.387 s after resizing native input to existing 1024 cap | 26 JPEG encode/decode sweep points and per-pixel reductions |

## Fixes and resolution cost

The shared detector path now caps the longest analyzed side at 1024 for ELA,
NPR, PRNU, and entropy. JPEG ghosts already had this 1024 cap; its conversion
and float cast were moved outside the 26-point quality loop. ELA's default cap
changed from 2000 to 1024, NPR no longer materializes the fourth zero-valued
patch tensor and computes the same variance statistic analytically, and the
other two caps bound the existing expensive algorithms. Entropy applies the
new cap to the ndarray path used by the shared adapter; its legacy file-path
API retains original-resolution behavior for compatibility. No calibration
value, threshold, quality sweep point, or direction of evidence changed.

On the profiled native camera image, the corresponding post-fix calls were:
ELA 0.227 s, NPR 0.144 s, PRNU 2.468 s, entropy 1.081 s, and JPEG ghosts
0.547 s. The last value is a noisy single profile and is not claimed as a
speedup; ghosts' intrinsic cost remains the 26-point sweep. The native corpus
comparison below quantifies the observable AUC impact of the bounded path.
It is a comparison to the R14 baseline, not a causal estimate, because the
current manifest has 414 rows while R14 reported 402 AI rows and 12 cameras,
and ELA/ghosts have different applicable JPEG counts.

| detector | R14 native AUC ± SE (n) | R16B native bounded AUC ± SE (applicable) | comparison |
|---|---:|---:|---:|
| ELA | 0.303 ± 0.095 (42) | 0.4190 ± 0.0985 (47) | +0.1160; row/applicability sets differ |
| NPR | 0.342 ± 0.087 (402) | 0.3696 ± 0.0876 (414) | +0.0276 |
| PRNU | 0.588 ± 0.078 (402) | 0.6282 ± 0.0743 (414) | +0.0402 |
| entropy | 0.554 ± 0.081 (402) | 0.5305 ± 0.0828 (414) | −0.0235 |
| JPEG ghosts | 0.417 ± 0.100 (42) | 0.4667 ± 0.0984 (47) | +0.0497; row/applicability sets differ |

These numbers show the resolution bound did not universally improve native
skill. The full parity result is therefore reported as measured, with no
threshold tuning or AUC optimization.

## Interpretation and provenance

R15C's parity encoder produced 1024×1024, 120,000-byte, optimized
non-progressive 4:2:0 JPEGs with EXIF removed. ELA and JPEG ghosts are
compression-history detectors, so their parity AUCs (0.3601 and 0.5220) are
evidence of the expected damage from the uniform JPEG re-save, not a reason to
retune them. They remain more scientifically appropriate for the native
variant. NPR, PRNU, and entropy now have complete parity measurements, but
their parity values also belong to the controlled re-encoded variant.

No source, licence, or citation was added in R16B. The parity bytes and labels
inherit the R15C/MANIFEST records and remain temporary under `/tmp`; no image
bytes are committed. This report does not alter those records.

The permitted detector test files passed, with the one entropy test that
writes a temporary file under `data/samples/` excluded by the task rule:

```text
test_ela.py:       9 passed
test_npr.py:       2 passed
test_prnu.py:      8 passed
test_ghosts.py:    2 passed
test_entropy.py:   7 passed, 1 deselected
```

The measurements are in `/tmp/r16b-parity-{ela,npr,prnu,entropy,jpeg_ghosts}.json`
and `/tmp/r16b-native-{ela,npr,prnu,entropy,jpeg_ghosts}.json`.
