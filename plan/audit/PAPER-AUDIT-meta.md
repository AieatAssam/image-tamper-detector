# Paper-fidelity audit: meta detectors

Read-only audit of `exif.py`, `c2pa.py`, `clip_probe.py`, and `learned.py`, completed 2026-09-01. The only repository file written in this round is this report. Grades are about fidelity to the cited source, not observed AUC.

| Detector | Grade | Short finding |
|---|---|---|
| `exif.py` | **UNVERIFIED** | The catalog cites no paper; the cited ExifTool page was not fetchable, so the tag semantics and format behavior cannot be paper-verified under R1. Several project thresholds are ours. |
| `c2pa.py` | **MAJOR-DRIFT** | The manifest/action path is present, but the implementation collapses C2PA validation/trust states, mislabels every validation failure as a signature failure, and has a non-JPEG byte-path MIME bug. |
| `clip_probe.py` | **MAJOR-DRIFT** | The frozen-CLIP plus linear-probe shape is present, but the paper's augmentation, single-source training protocol, and encoder checkpoint are not reproduced. |
| `learned.py` | **UNVERIFIED** | The catalog has no paper citation. The implementation is faithful to the downloaded model artifact's config, with an intentional face gate added outside model inference. |

## `exif.py` — UNVERIFIED

### Source and published specification

The catalog explicitly says “no single paper” and cites [ExifTool EXIF tag names](https://exiftool.org/TagNames/EXIF.html) for semantics (`plan/reference/detector-catalog.yaml:414-419`). The direct cited page returned HTTP 403 during this audit. The accessible [ExifTool tag-name index](https://www.exiftool.org/TagNames/index.html) explains that its tables list recognized tags and their file identifiers; search results for the cited table identify `Software` (`0x0131`), `ProcessingSoftware` (`0x000b`), and `PixelXDimension`/`PixelYDimension` (`0xA002`/`0xA003`). This is documentation, not a paper, and is insufficient to invent a paper-level threshold or claim a complete algorithm.

The repository catalog specifies five checks (`plan/reference/detector-catalog.yaml:391-413`):

1. Extract and decode an embedded JPEG thumbnail, resize the full image to that size, then compute global similarity and a regional difference map.
2. Treat a `Software` or `ProcessingSoftware` value naming an editor as direct evidence, reporting the literal value.
3. Give low-weight evidence when all four camera tags (`Make`, `Model`, `DateTimeOriginal`, `ExposureTime`) are absent from a JPEG with camera-like dimensions.
4. Compare original, digitized, and modified timestamps; a later modified time indicates re-save.
5. Compare EXIF pixel dimensions with decoded dimensions; mismatch indicates post-capture resizing and gates the camera heuristic.

No source value is available for the detector's `3.0` thumbnail mismatch multiplier, `0.5` mismatch-evidence gate, evidence weights `0.85`, `0.15`, `0.7`, `0.9`, or the final calibrated threshold/scale. Those are project policy, not ExifTool constants (`backend/app/analysis/exif.py:56-102`).

### Our implementation

- `ExifConsistencyDetector` is applicable to JPEG, PNG, WEBP, and TIFF, and abstains only when both metadata and a raw thumbnail are absent (`backend/app/analysis/exif.py:25-49`).
- It compares a decoded embedded thumbnail with the resized main image and builds a difference map (`backend/app/analysis/exif.py:56-66`, `216-231`).
- It reads `0x0131`/`0x000B`, the camera fields, three timestamp fields, and `0xA002`/`0xA003` (`backend/app/analysis/exif.py:68-89`, `105-119`).
- It uses the maximum evidence value, then applies calibration; higher mismatch/editor/timestamp/dimension evidence is worse (`backend/app/analysis/exif.py:91-102`; `backend/app/analysis/calibration.json:195-205`).

### Deltas and direction

1. **Verification gap, blocking grade.** There is no paper to re-read and the primary cited tag table was unavailable. The findings below are implementation-vs-catalog observations, not claims that ExifTool specifies this detector.

2. **Software evidence is broader than the catalog rule.** The catalog says an editor-naming string is evidence. The code records any non-empty value returned by `_first(metadata, 0x0131, 0x000B)` (`backend/app/analysis/exif.py:68-72`); it does not test whether the literal identifies an editor or use a documented allowlist. A camera vendor, scanner, or other capture pipeline that writes `Software` can therefore receive the same strong evidence. This is a semantic broadening, not a source constant.

3. **Thumbnail extraction does not cover the catalog's whole format claim.** The catalog allows a Pillow IFD1 or `piexif.load(path)['thumbnail']` route. The fallback parser in this implementation only recognizes a raw JPEG APP1 Exif segment (`backend/app/analysis/exif.py:151-176`) and the implementation's raw-thumbnail lookup is limited to the JPEG thumbnail tag (`backend/app/analysis/exif.py:138-148`). For non-JPEG containers with an EXIF thumbnail, the declared JPEG/PNG/WEBP/TIFF applicability is not demonstrated by this implementation. This needs a fixture-based check before being called a confirmed runtime failure.

4. **Direction is consistent with the stated premise.** The premise is that internal inconsistency or explicit editing metadata is more suspicious than a self-consistent capture. The statistic increases with thumbnail mismatch, editor metadata, timestamp disagreement, or dimension mismatch; `higher_is_worse` is true in calibration. EXIF absence returns `NOT_APPLICABLE`, so absence is not inverted into evidence (`backend/app/analysis/exif.py:1-5`, `36-39`).

5. **Preprocessing is not the round-16B bound.** EXIF parsing uses the uploaded bytes and the decoded main image; it does not use the shared 1600-side image for the thumbnail comparison (`backend/app/analysis/exif.py:41-66`; `backend/app/analysis/base.py:77-88`). No 1024-side JPEG sweep or other paper preprocessing is being silently applied here.

### Prioritised fixes

1. Replace or supplement the inaccessible/no-paper citation with a source that specifies the detector, or keep this detector explicitly marked unverified. This is needed before asserting paper fidelity.
2. Restrict strong `Software` evidence to documented editor semantics, or lower/document the broader heuristic; otherwise capture software can look like editing.
3. Make thumbnail extraction format-aware and test each declared format, if non-JPEG EXIF thumbnails are in scope.

## `c2pa.py` — MAJOR-DRIFT

### Source and published specification

C2PA is a standard rather than a research paper. The cited sources are the [C2PA Specification 2.1](https://spec.c2pa.org/specifications/specifications/2.1/specs/C2PA_Specification.html) and [c2pa-python](https://github.com/contentauth/c2pa-python). The specification describes a manifest as assertions, a claim, and a signature; it defines **Well-Formed**, **Valid**, and **Trusted** states. In the specification's terms, “Any Trusted manifest is also Valid,” while Valid additionally requires signature, time-validity, and credential checks; Trusted requires a trusted signing credential. The specification also requires a `c2pa.created`/`c2pa.opened` action structure and identifies generative provenance through `digitalSourceType` such as `trainedAlgorithmicMedia`.

The catalog's intended semantics are narrower and correct at a high level: no manifest is not tamper evidence; a broken signature is strong post-signing-modification evidence; a valid generative `c2pa.created` assertion is positive AI-origin evidence (`plan/reference/detector-catalog.yaml:153-163`).

### Our implementation

- The registry imports and registers this detector (`backend/app/analysis/registry.py:14`, `92-109`), contrary to the catalog's stale “DEAD CODE/BROKEN” note (`plan/reference/detector-catalog.yaml:148-152`).
- It uses the current `Reader(...).json()` API, selects the active manifest, and returns `NOT_APPLICABLE` for a missing manifest (`backend/app/analysis/c2pa.py:62-85`).
- It recursively looks for `c2pa.created` plus `trainedAlgorithmicMedia` (`backend/app/analysis/c2pa.py:163-173`).
- It assigns project scores of `0.95` for validation failure, `1.0` for a generative assertion, and `0.05` otherwise, with a project threshold of `0.5` (`backend/app/analysis/c2pa.py:27`, `100-119`).

### Deltas and direction

1. **Central validation-state omission.** `_validation_failed` treats a truthy `validation_state` as failed only if its string contains `invalid` or `error`; otherwise it falls through as valid, and a missing state is also not a failure (`backend/app/analysis/c2pa.py:142-147`). The code does not require or distinguish the specification's Well-Formed, Valid, and Trusted states. It constructs `Reader` without an explicit verification/trust `Context` (`backend/app/analysis/c2pa.py:66-68`). The c2pa-python documentation states that `Context` encapsulates verification/trust configuration. This can turn “not reported invalid” into “valid C2PA manifest” without proving the standard's Valid or Trusted conditions. This is the main reason for the grade.

2. **Invalid manifests can still be called validated generative manifests.** The generated issue is appended whenever `generated` is true, even when `validation_failed` is also true (`backend/app/analysis/c2pa.py:100-108`). Its text says “Validated C2PA manifest identifies generative image creation.” That wording contradicts the standard's validity premise when the same result contains a validation failure. The issue must be gated on the appropriate successful validation state.

3. **Detected validation failures are mislabeled as signature failures.** Any detected validation failure becomes `signature_invalid` with “post-signing modification is possible” (`backend/app/analysis/c2pa.py:101-104`, `142-147`). The specification's validation requirements include malformed assertions, invalid ingredients, credential/time problems, and other conditions; not every failure establishes post-signing modification. The issue taxonomy is therefore stronger than the source semantics.

4. **Non-JPEG byte compatibility is broken.** The compatibility byte path hardcodes `mime="image/jpeg"` (`backend/app/analysis/c2pa.py:47-60`). The normal MIME helper recognizes JPEG, PNG, and TIFF but has no WEBP branch and otherwise defaults to JPEG (`backend/app/analysis/c2pa.py:176-183`), despite declaring WEBP support (`backend/app/analysis/c2pa.py:22-26`, `120-129`). This can prevent correct manifest parsing for valid PNG/WEBP/TIFF bytes.

5. **Generator-name fallback is ours, not C2PA provenance.** `known_ai_generators` and substring matching `claim_generator` are added project heuristics (`backend/app/analysis/c2pa.py:29-33`, `87-91`). They are not equivalent to a validated signed `c2pa.created` assertion and should not be presented with the same origin certainty.

6. **Scores and threshold are adapter policy, not C2PA constants.** The standard defines validation/provenance semantics, not probabilities `0.95/1.0/0.05` or a `0.5` decision threshold. These values are acceptable only as explicitly documented project policy and must not be described as C2PA's threshold.

7. **Direction is not inverted, but its premise is conditional.** Higher generative-assertion or validation-failure evidence produces a higher suspicion score, while absence produces `NOT_APPLICABLE`; this matches the catalog's provenance premise. However, C2PA's trust/validity result must be correct before “higher means worse” is meaningful. A valid non-generative provenance record is given a low score by project policy, not by a C2PA claim that the image is safe.

8. **Preprocessing is appropriate for provenance.** The detector passes original bytes to the reader and does not apply the shared 1600-side bound (`backend/app/analysis/c2pa.py:38-45`; `backend/app/analysis/base.py:77-88`). The MIME defects above are the relevant input-handling drift.

### Prioritised fixes

1. Preserve and expose the SDK's explicit Well-Formed/Valid/Trusted state and configure/document the trust context; never infer Valid from absence of an “invalid” substring.
2. Do not emit “validated generative” for a manifest that failed validation, and distinguish signature, assertion, ingredient, credential, and time failures.
3. Pass the actual MIME for bytes and add tested WEBP handling before advertising all four formats.
4. Separate cryptographically validated C2PA assertions from the project-only generator-name heuristic and label the three numeric scores as adapter policy.
5. Update the stale catalog status; it is a catalog defect, not a reason to remove the now-registered detector.

## `clip_probe.py` — MAJOR-DRIFT

### Source and published specification

The primary source is [Ojha, Li, and Forsyth, “Towards Universal Fake Image Detection with CLIP” (CVPR 2023)](https://arxiv.org/html/2302.10174v2). The paper's premise is a frozen CLIP visual feature space that is not itself trained for real/fake discrimination, followed by a classifier trained on those features. It uses CLIP ViT-L/14, reports a 768-dimensional feature, freezes the encoder, and trains one linear layer with sigmoid/BCE. The paper's main protocol trains with one source generative model and tests on unseen generators; its stated goal is generalization to an arbitrary fake source after training on one kind of generative model.

The paper also specifies training augmentation: the appendix says the ViT:CLIP baseline uses “Blur+JPEG” with probability `0.5`, and says the proposed linear classifier uses blur/JPEG augmentation too. It reports a non-oracle threshold selected on the ProGAN validation set, typically near `0.5`. The paper's data are generated at `256x256`; that is a corpus construction detail, not a license to claim that every arbitrary uploaded image is paper-distributed.

### Our implementation

- The runtime freezes an OpenCLIP `ViT-L-14` backbone and computes one image embedding, then applies a linear probe and sigmoid (`backend/app/analysis/clip_probe.py:1-1`, `125-167`).
- It uses the LAION checkpoint `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` (`backend/app/analysis/clip_probe.py:14-19`, `143-144`; `models/clip/linear_probe.json:14-18`), not the paper's unspecified-in-code OpenAI CLIP checkpoint trained on 400M image-text pairs.
- The fitting path uses source-image grouping and complete generator holdouts (`scripts/fit_clip_probe.py:1-2`, `47-64`, `111-170`) but trains on multiple repository generators/axes (`scripts/fit_clip_probe.py:22-44`, `147-158`).
- Runtime direction is explicit: `higher_is_worse=True` (`backend/app/analysis/clip_probe.py:26-40`), and the live calibration threshold is `0.7939407755`, not the paper's typical approximately `0.5` (`backend/app/analysis/calibration.json:75-85`).

### Deltas and direction

1. **Published training augmentation is omitted.** `scripts/fit_clip_probe.py` extracts features from preprocessed images in batches but contains no blur or JPEG augmentation (`scripts/fit_clip_probe.py:96-108`). This omits the paper's required/specified `p=0.5` training augmentation. The runtime's OpenCLIP preprocessing is not a substitute for that augmentation. This is a method-level drift.

2. **The training protocol is not the paper's single-source protocol.** The repository trains from multiple heterogeneous generator sources and then uses holdouts. That is a reasonable robustness adaptation and the catalog records it (`plan/reference/detector-catalog.yaml:539-542`), but it is not the paper's “one kind of generative model” setup. The resulting probe and reported AUC cannot be treated as a reproduction of the paper's training/evaluation method.

3. **The encoder checkpoint changes the feature space.** The paper specifies CLIP ViT-L/14 trained on 400M image-text pairs; the repository explicitly selects the LAION 2B checkpoint. The architecture name matches, but the weights/training corpus do not. For a method whose central claim depends on a fixed pretrained feature space, this is material, not a cosmetic model-name difference.

4. **The optimizer/objective constants are repository-defined.** The paper describes a linear classifier trained with BCE. The repository standardizes feature columns, uses L2=`0.05`, 350 iterations, step size `0.08`, and projects weights non-negative (`scripts/calibrate.py:286-320`). The non-negative projection is a project direction guard, not a published CLIP-paper step. It may be useful for fusion, but the trained object is not the paper's unconstrained linear probe.

5. **Threshold is calibrated locally.** The code default is `0.5` (`backend/app/analysis/clip_probe.py:14-19`), but the actual runtime calibration is `0.7939407755` (`backend/app/analysis/calibration.json:75-85`). That is an ours-versus-paper difference; it is not inherently wrong because the paper itself chooses a validation threshold, but it must be described as corpus calibration rather than a paper constant.

6. **Direction is consistent.** The probe's positive class is the generated/deepfake side, so increasing sigmoid output means more fake suspicion. `higher_is_worse=True` matches that direction. There is no round-1-style inversion.

7. **Preprocessing does not use the shared 1024/1600 bound.** The full `ctx.pil_image` is handed to the OpenCLIP transform, which performs the model's own resize/crop before encoding (`backend/app/analysis/clip_probe.py:158-162`). The shared `downscaled_rgb_uint8` cap is not used. This does not reproduce the paper's 256x256 corpus, but it does not silently add the metadata/image-analysis bound; the inference input is still reduced to CLIP's model transform.

### Prioritised fixes

1. If paper fidelity is required, add the paper's blur/JPEG training augmentation, use a paper-matched one-source protocol, and retrain/evaluate; otherwise label this explicitly as a repository adaptation.
2. Decide whether LAION ViT-L/14 is intentional. If so, record that the feature space differs from the cited paper; if not, use the paper-matched checkpoint and retrain the probe.
3. Record the local standardization, L2, iteration, step, and non-negative projection as implementation policy and avoid presenting the resulting model as the paper's exact linear probe.
4. Keep the calibrated threshold separate from the paper's approximately `0.5` validation threshold and report cross-generator performance as this repository's result.

## `learned.py` — UNVERIFIED

### Source and published specification

The catalog entry has no `citation` field (`plan/reference/detector-catalog.yaml:421-451`), so there is no actual paper to fetch and compare. Per R1, this detector is **UNVERIFIED**, not Faithful by inference from the catalog summary.

The available primary artifact is the [Hugging Face ONNX model card](https://huggingface.co/onnx-community/Deep-Fake-Detector-v2-Model-ONNX), which identifies an ONNX version of `prithivMLmods/Deep-Fake-Detector-v2-Model`, based on `google/vit-base-patch16-224-in21k`, fine-tuned for Realism/Deepfake face images. The card describes 224x224 RGB inputs and documents training-time augmentation, but it is a model card, not a research paper establishing a published detector method.

### Our implementation and artifact comparison

- The local `config.json` specifies `ViTForImageClassification`, hidden size 768, image size 224, and labels `0=Realism`, `1=Deepfake` (`models/config.json:1-32`).
- The local `preprocessor_config.json` specifies RGB resize to 224x224 with resample `2`, rescale `0.00392156862745098`, mean/std `[0.5, 0.5, 0.5]`, CHW float32 input (`models/preprocessor_config.json:1-22`). The hardcoded runtime preprocessing matches those values (`backend/app/analysis/learned.py:55-57`).
- Runtime output handling uses `softmax(logits)[1]`, while accepting already-normalized two-value output only when the values sum to one (`backend/app/analysis/learned.py:61-69`), matching the catalog's artifact-derived label rule (`plan/reference/detector-catalog.yaml:444-447`).
- The model is optional and lazy-loaded; missing weights or a missing detectable face returns `NOT_APPLICABLE` (`backend/app/analysis/learned.py:31-53`, `80-101`).

### Deltas and direction

1. **No paper-level completeness claim is possible.** The model card documents an artifact and training history, not a paper's complete inference method, dataset protocol, threshold, or signal premise. No paper constants can be supplied without guessing.

2. **The Haar face gate is added repository behavior.** The model artifact's 224x224 preprocessing is matched, but the runtime first applies OpenCV's Haar cascade to a shared image capped at 1600 pixels (`backend/app/analysis/learned.py:80-91`; `backend/app/analysis/base.py:77-88`). The catalog explicitly requires this gate (`plan/reference/detector-catalog.yaml:442-443`), so it is an intentional applicability restriction outside the model input, not a paper-derived preprocessing step. It can abstain on a missed/small face and therefore changes coverage.

3. **Training augmentations are not reimplemented, appropriately.** The model card's rotation, sharpness, and resize/crop items are training history, not inference requirements. The repository correctly does not attempt to recreate them at inference, but without a paper they cannot establish fidelity.

4. **Direction is consistent with the artifact labels.** Higher `Deepfake` probability becomes higher suspicion in the detector and calibration (`backend/app/analysis/learned.py:70-78`; `backend/app/analysis/calibration.json:243-253`). This is a label direction, not a physical premise validated by a paper. The face-only limitation is stated in the implementation (`backend/app/analysis/learned.py:20-29`) and catalog (`plan/reference/detector-catalog.yaml:448-449`).

5. **No 1024-side preprocessing is added to model inference.** The model consumes the full RGB image before its fixed 224 resize; only the separate face gate uses the shared 1600-side bound. There is no evidence here that the round-16B bound invalidates the model's actual input normalization.

6. **Threshold and scale are ours.** The runtime's threshold/scale are calibration values, not model-card or paper constants. They should not be described as published model settings.

### Prioritised fixes

1. Add the actual training-paper citation and its primary source, if one exists; otherwise keep this detector artifact-verified but paper-unverified.
2. Preserve the explicit face-only scope and report the face gate's abstention coverage separately from model accuracy.
3. Keep the model-card preprocessing assertions tied to the pinned artifact revision; do not infer a general-purpose tamper method or paper threshold from the model card.
