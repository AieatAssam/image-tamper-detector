# Round 19 meta-family repair report

Date: 2026-09-01

Scope: `exif.py`, `c2pa.py`, `clip_probe.py`, and `learned.py`. No calibration
artifact, corpus file, benchmark script, or other detector was changed. The
catalog edits are limited to the four owned entries.

## Status

| detector | audit grade | resolution | claimed grade after repair | result |
|---|---|---|---|---|
| `exif.py` | UNVERIFIED | (a) implement format/editor guards; (b) correct the catalog claim | UNVERIFIED | code repaired; no single primary paper exists |
| `c2pa.py` | MAJOR-DRIFT | (a) implement state-aware validation, failure typing, MIME dispatch | MINOR-DRIFT | P0 defects repaired; numeric scores and generator fallback remain project policy |
| `clip_probe.py` | MAJOR-DRIFT | (b) correct the catalog claim | MAJOR-DRIFT, explicitly claimed as a variant | no paper-specific retraining was fabricated |
| `learned.py` | UNVERIFIED | (b) correct the catalog claim | UNVERIFIED, explicitly artifact-verified | no training paper was found or invented |

The full test command is green after the final owned changes: `131 passed`.

## 1. EXIF consistency

### Audit items and decisions

1. **P1 source-verification gap — resolution (b).** The audit found no single
   paper and the cited ExifTool EXIF page was unavailable. The catalog now says
   this is a repository heuristic, not a paper reimplementation
   (`plan/reference/detector-catalog.yaml:436`). This is still UNVERIFIED; no
   source threshold was invented.

2. **P1 broad software evidence — resolution (a).** The old code treated every
   non-empty `Software`/`ProcessingSoftware` tag as strong editor evidence. It
   now records the tag but gives the strong `0.85` evidence only to the
   documented markers Photoshop, GIMP, Lightroom, and Snapseed
   (`backend/app/analysis/exif.py:25`, `71-77`). The negative regression test
   uses `NIKON CORPORATION` and asserts `raw_score == 0.0`
   (`backend/tests/test_exif.py:94-102`).

3. **P1 non-JPEG thumbnail coverage — resolution (a).** The detector now
   resolves the IFD1 offset/length pair against the container EXIF payload and
   falls back to the JPEG APP1 parser. It covers PNG/WEBP EXIF payloads and
   TIFF IFD1 bytes without adding `piexif` (`backend/app/analysis/exif.py:143-180`).
   A constructed test covers PNG, WEBP, and TIFF
   (`backend/tests/test_exif.py:105-111`).

### Direction and preprocessing

No signal direction was changed. Thumbnail mismatch, documented editor
software, timestamp disagreement, and dimension disagreement remain higher-is-
worse. EXIF absence remains `NOT_APPLICABLE` (`backend/app/analysis/exif.py:39-42`).
The detector still reads the original bytes/full decoded image; it does not
inherit the shared 1600-side bound for the thumbnail comparison.

### Measurement

The constructed non-editor case moved from the old unconditional raw evidence
`0.85` to `0.0`; the after command printed:

```text
non-editor-software 0.0 0.0 EXIF fields are internally consistent
```

The old implementation returned no usable thumbnail metric for the tested
non-JPEG containers. The repaired test now passes all three containers with
`thumbnail_similarity > 0.95`. No corpus AUC was rerun or claimed.

### Calibration impact

The existing `raw_score` key is retained, but the distribution changes for
non-editor software tags. `backend/app/analysis/calibration.json` was not
edited because recalibration is reserved for the human after all family
repairs. The human should refit native EXIF calibration before promotion.

## 2. C2PA provenance

### Audit items and decisions

1. **P0 validation/trust collapse — resolution (a).** The reader now uses an
   explicit SDK `Context`, obtains `validation_state`, and preserves the
   distinction between `Well-Formed`, `Valid`, `Trusted`, `Invalid`, `Error`,
   and unknown (`backend/app/analysis/c2pa.py:65-76`, `170-177`). Only Valid or
   Trusted manifests can produce a validated generative assertion. Trusted is
   separately exposed in metadata and metrics (`backend/app/analysis/c2pa.py:105-112`,
   `148-152`). A Well-Formed or unknown state is not scored as clean evidence.

2. **P0 validation failures mislabeled as signatures — resolution (a).**
   Content/hash mismatches become `post_signing_mismatch`, signature failures
   become `signature_invalid`, credential failures become `credential_invalid`,
   and other failures become `validation_failed`
   (`backend/app/analysis/c2pa.py:184-196`). The generative issue is emitted
   only when the active manifest is Valid or Trusted
   (`backend/app/analysis/c2pa.py:100-127`). Tests cover both a data-hash
   mismatch and a generic missing-claim failure
   (`backend/tests/test_c2pa.py:59-84`).

3. **P0 non-JPEG byte MIME bug — resolution (a).** Byte input now uses
   `_mime_for_bytes`; JPEG, PNG, TIFF, and RIFF/WEBP are dispatched with their
   actual MIME, and unknown bytes no longer default to JPEG
   (`backend/app/analysis/c2pa.py:56-57`, `231-245`). The PNG dispatch regression
   is tested at `backend/tests/test_c2pa.py:134-145`.

4. **P1 active-manifest scope — resolution (a).** Generative action scanning
   is limited to the active manifest's `assertions`, not nested ingredient
   manifests (`backend/app/analysis/c2pa.py:212-224`). The regression is at
   `backend/tests/test_c2pa.py:113-130`.

5. **P1 generator fallback and numeric values — resolution (b).** The
   project-only generator-name fallback remains, but only after a Valid or
   Trusted state. The catalog now states that the detector is state-aware and
   that the positive assertion requires Valid or Trusted
   (`plan/reference/detector-catalog.yaml:140-156`). The `0.95`, `1.0`, `0.05`,
   and `0.5` values remain adapter policy, not C2PA constants.

6. **P1 stale catalog status — resolution (b).** The catalog's obsolete
   “DEAD CODE” and removed-API claims were replaced with the current
   `c2pa-python 0.37.8` Reader behavior and test coverage
   (`plan/reference/detector-catalog.yaml:140-144`).

### Direction and absence semantics

The direction remains higher suspicion for a validated generative assertion or
an explicit validation failure. A valid non-generative manifest remains low
project suspicion. A missing manifest still returns `NOT_APPLICABLE`, null
score, null flag, and no issues (`backend/app/analysis/c2pa.py:160-162`). A
manifest with no validation state is not turned into either a clean score or a
tampering flag; it is applicable with null score and an informational
`validation_unknown` issue.

### Before and after fixture measurement

The pinned SDK fixtures were run before the repair with the existing analyzer.
The relevant pre-repair output was:

```text
C.jpg applicable 0.05 valid C2PA manifest contains no generative creation assertion [...signingCredential.untrusted...]
XCA.jpg applicable 0.95 C2PA manifest is present but validation failed [...signingCredential.untrusted..., assertion.dataHash.mismatch...]
```

The exact after command was:

```sh
.venv/bin/python - <<'PY'
from pathlib import Path
from backend.app.analysis.c2pa import C2PAAnalyzer
for name in ('C.jpg', 'XCA.jpg'):
    result = C2PAAnalyzer().analyze_image(Path('/tmp/itd-c2pa-cache') / name)
    print(name, result['state'].value, result['score'], result['flagged'], result['reason'])
    print('  issue=', result['issues'][0]['type'] if result['issues'] else None)
    print('  validation_state=', result['metadata'].get('validation_state'), 'trusted=', result['metadata'].get('trusted'))
PY
```

Its real output was:

```text
C.jpg applicable 0.05 False valid C2PA manifest contains no generative creation assertion
  issue= None
  validation_state= valid trusted= False
XCA.jpg applicable 0.95 True C2PA manifest is present but validation failed
  issue= post_signing_mismatch
  validation_state= invalid trusted= False
```

The numeric scores did not move. The XCA classification moved from the
overbroad `signature_invalid` label to `post_signing_mismatch`; C2PA's
untrusted-but-Valid C fixture remains low and is now explicitly reported as
`trusted=False`.

### Calibration impact

The existing `generative_assertion` metric key is retained, but it now means a
structured generative action in the active manifest, while
`validated_generative_assertion` records whether the state is Valid or Trusted.
`validation_failed` also becomes state-aware. The current calibration artifact
was not edited. The human must refit or explicitly invalidate the old C2PA
calibration before using this detector's updated metrics in fusion.

## 3. CLIP probe

### Audit items and decision

1. **P1 paper augmentation, single-source protocol, and checkpoint — resolution
   (b).** The cited Ojha et al. paper specifies blur/JPEG training
   augmentation, a one-source training setup for the universal-generalization
   claim, and a CLIP ViT-L/14 feature space. This repository has no paper-matched
   augmentation/checkpoint/training pipeline and the user explicitly prohibits
   adding weights or data. The catalog now says the current LAION/multi-source
   implementation is a repository variant and not a paper reproduction
   (`plan/reference/detector-catalog.yaml:556-569`).

2. **P1 local optimizer and threshold constants — resolution (b).** No code
   change was made. The catalog claim now distinguishes the repository probe's
   local standardization, direction guard, and calibration from the paper's
   unconstrained linear probe. The runtime's `higher_is_worse=True` remains
   correct (`backend/app/analysis/clip_probe.py:26-40`).

### Grade and measurements

The fidelity grade remains MAJOR-DRIFT, but the claim is now honest. No AUC,
threshold, weight, or model artifact was changed. The prior recorded parity
measurement `0.999585` and the expanded-negative measurement `0.801464` remain
historical corpus results, not a new paper-fidelity result. No paper-matched
measurement was possible without the missing training recipe/checkpoint.

### Calibration impact

None from this repair: no CLIP runtime metric or calibration key changed. The
catalog's variant note is the required resolution-(b) correction.

## 4. Learned ONNX detector

### Audit items and decision

1. **P1 missing paper citation — resolution (b).** The catalog has no training
   paper citation. The Hugging Face model card and pinned local configs verify
   an artifact, not a research method. The catalog status now explicitly says
   artifact-verified repository variant with no training-paper citation
   (`plan/reference/detector-catalog.yaml:447-453`). The grade remains
   UNVERIFIED.

2. **P1 face gate and model preprocessing — resolution (b).** The runtime's
   224x224 RGB/rescale/normalization values already match the pinned
   `config.json` and `preprocessor_config.json`; the Haar gate is an intentional
   repository applicability restriction. The catalog retains the face-only
   scope and does not claim a general or paper-derived model
   (`plan/reference/detector-catalog.yaml:454-470`). No model weights, external
   data, or inference-time training augmentation was added.

### Direction and measurements

The model-label direction remains higher `Deepfake` probability -> higher
suspicion. No code, metric, threshold, or calibration value changed. No
paper-level AUC or paper-fidelity measurement was possible without a cited
paper and the prohibited external training artifacts.

### Calibration impact

None from this repair: no learned metric key changed. The model's calibrated
threshold and scale remain repository values and were not edited.

## Verification

### Required focused checks

```text
.venv/bin/python -m pytest backend/tests/test_exif.py backend/tests/test_c2pa.py -q
..................                                                       [100%]
18 passed in 0.58s
```

```text
.venv/bin/python -c "import yaml,sys; yaml.safe_load(open('plan/reference/detector-catalog.yaml')); print('catalog yaml: valid')"
catalog yaml: valid
```

The final required full-suite command was:

```text
.venv/bin/python -m pytest backend/tests -q
............................              [100%]
131 passed, 1 warning in 127.85s (0:02:07)
```

The warning was the existing Starlette deprecation warning about `httpx`.

### Remaining open items

- EXIF remains UNVERIFIED because there is no single primary paper and the
  cited ExifTool page was unavailable during the audit.
- CLIP remains a repository variant: no paper-matched augmentation,
  checkpoint, or single-source training was added.
- Learned remains artifact-verified but paper-unverified because no training
  paper is cited.
- C2PA's calibration artifact needs human refitting after the metric-semantics
  change. No calibration file was touched.
- No benchmark/AUC rerun was performed for this repair; changing corpus or
  calibration would violate this round's scope.

No commit was created.
