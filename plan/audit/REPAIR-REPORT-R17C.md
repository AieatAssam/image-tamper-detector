# Round 17C — reconcile the scientific record

Date: 2026-09-01
Status: complete for the permitted documentation files; no commit made.

## Outcome

The scientific reference now treats metadata shortcuts and encoding variants
as first-order validity conditions. The old catalog measurement block was
replaced with one current primary measurement for every runtime detector ID.
Each row records AUC and Hanley–McNeil SE when an AUC exists, the corpus,
variant, applicable count, scope, and date. `null` is used for an unavailable
comparison, never as a zero score. The latest values come only from the
committed R15C, R16A, and R16B reports.

The final 17A/17B numbers are not present at this writing. Those reports may
supersede rows in the next consolidation; this report does not invent
placeholders or silently change the current record.

## Corrections made

### Shortcut caveat moved to the front

`docs/detection-principles.md`, `docs/corpus.md`, `docs/calibration.md`, and
`README.md` now state before the detector claims that:

- the current AI screen fails the metadata-only gate at held-out AUC
  `0.8750 +/- 0.0598` and pooled AUC `0.9583 +/- 0.0137`, selected on width;
- this invalidates the interpretation of Round 10's per-generator table as
  forensic generation skill;
- Round 12's CLIP `1.0000 +/- 0.0000` is retired as a forensic result because
  it was a container-format/domain artifact;
- Round 14's WildFake sample independently failed the same gate at `1.0000`
  on PNG versus JPEG, so it does not repair the corpus construction problem.

The R15C parity copy passes all four metadata ablations at exactly `0.500`
held out and pooled, but it is not a free normalisation: its quality
distribution remains class-correlated. AI rows have median/mean quality 66 /
63.24; camera negatives have median/mean 81.5 / 74.58. The docs therefore do
not treat parity as proof that every forensic shortcut is gone.

### Native/parity distinction made explicit

The documents now define `native` as the supplied bytes and `parity` as the
R15C exact 120,000-byte, 1024×1024 JPEG re-save. The current R16C scope is
recorded consistently in the catalog and docs:

| scope | detector IDs |
|---|---|
| parity only | `aeroblade`, `clip_probe`, `learned`, `npr`, `spectral`, `entropy` |
| native only | `c2pa`, `qtable`, `exif`, `cfa`, `ela` |
| both | `copy_move`, `double_jpeg`, `jpeg_ghosts`, `prnu`, `resampling`, `splicebuster`, `zero` |

The R16C train/serve limitation is stated plainly: calibration and benchmark
scope checks do not make a variant-blind upload call safe. A serving
orchestrator must select bytes matching the fitted variant. The committed
`calibration.json` remains a native legacy artifact and was not refit here.

### Current measurement record

The catalog's `measurements.detectors` map and the evidence table in
`docs/detection-principles.md` now contain the following primary rows. The
pooled AI-axis rows are unpaired generated-versus-camera screens, not
source-paired reconstructions.

| detector | AUC +/- SE | corpus | variant | date |
|---|---:|---|---|---|
| aeroblade | 0.416 +/- 0.088 | R15C byte budget | parity | 2026-09-01 |
| c2pa | N/A | local manifest | native | 2026-08-31 |
| cfa | N/A | local manifest | native | 2026-08-31 |
| clip_probe | 0.999585 +/- 0.000757 | R15C byte budget | parity | 2026-09-01 |
| copy_move | 0.386 +/- 0.127 | R15C | native | 2026-08-31 |
| double_jpeg | 0.192 +/- 0.082 | R15C | native | 2026-08-31 |
| ela | 0.4190 +/- 0.0985 | R16B bounded | native | 2026-09-01 |
| entropy | 0.7616 +/- 0.0559 | R16B bounded | parity | 2026-09-01 |
| exif | 0.083 +/- 0.058 | R15C | native | 2026-08-31 |
| jpeg_ghosts | 0.4667 +/- 0.0984 | R16B bounded | native | 2026-09-01 |
| learned | 0.184 +/- 0.131 | R15C byte budget | parity | 2026-09-01 |
| npr | 0.2803 +/- 0.0846 | R16B bounded | parity | 2026-09-01 |
| prnu | 0.6282 +/- 0.0743 | R16B bounded | native | 2026-09-01 |
| qtable | N/A | local manifest | native | 2026-08-31 |
| resampling | 0.298 +/- 0.094 | R15C | native | 2026-08-31 |
| spectral | 0.508 +/- 0.084 | R15C byte budget | parity | 2026-08-31 |
| splicebuster | 0.720 +/- 0.110 | R15C | native | 2026-08-31 |
| zero | 0.275 +/- 0.084 | R15C | native | 2026-08-31 |

Where R16A/B measured both variants, the catalog retains a `comparison` row
without mixing it into the primary result. This preserves the observed cost of
re-encoding: AEROBLADE native/parity `0.547/0.416`, learned `0.423/0.184`, NPR
`0.3696/0.2803`, entropy `0.5305/0.7616`, PRNU `0.6282/0.6490`, ELA
`0.4190/0.3601`, and JPEG ghosts `0.4667/0.5220`, each with its report SE.

### Superseded figures removed

The following values were removed from the scientific reference and current
catalog rather than left beside newer measurements:

- the R12 AEROBLADE source-paired `0.511013 +/- 0.025584` and AI-axis
  `0.539957 +/- 0.082240`;
- the R12 learned values, including source-paired `0.623529` and AI-axis
  `0.423853 +/- 0.136642`;
- the prior NPR `0.341667 +/- 0.087205`, entropy `0.472885`, ELA `0.437642`,
  double-JPEG `0.659864`, JPEG ghosts `0.538549`, spectral `0.535792`, ZERO
  `0.506508`, resampling `0.474926`, and related pre-parity catalog values.

The old fused calibration AUC `0.5784615384615385` remains mentioned only as a
property of the still-committed native legacy `calibration.json`; it is not
presented as a current parity result. The older `0.6521739130434783` value was
removed as superseded.

### AEROBLADE contradiction recorded

R16A measures the current adapter at parity AUC `0.416 +/- 0.088` and native
`0.547 +/- 0.082` on 414 AI-axis rows. The per-generator parity result falls
from `0.668 +/- 0.079` native to `0.456 +/- 0.093` for FLUX, and from
`0.386 +/- 0.100` to `0.189 +/- 0.082` for stable-diffusion-1-3. The R8
finding records the paper's mean AP as `0.992` across Stable Diffusion 1.1,
1.5, and 2.1, Kandinsky 2.1, and Midjourney.

This is a contradiction between this repository's AUC and the paper's AP; the
metrics are not identical, but the discrepancy is material. It is not a
disproof of the paper. The honest candidate
explanations are: this adapter uses a distilled TAESD stand-in rather than the
paper's exact autoencoder, the corpus and post-processing differ, or this
implementation contains an error. The documents state all three possibilities
and do not hide the discrepancy.

CLIP is treated with the same discipline. R16A's parity result is
`0.999585 +/- 0.000757`, which rounds to 1.000, but only 12 real-camera
negatives exist. The surviving separation is therefore a content/corpus
confound, not a forensic generation result.

## Licensing record

The implementation-provenance table remains in `docs/detection-principles.md`
and was extended with the D6 boundary:

- Splicebuster, Kirchner/Popescu–Farid resampling, ZERO, CFA, AEROBLADE, CLIP
  direction, and the primary JPEG/EXIF/spectral/copy-move/double-JPEG methods
  are independently reimplemented from papers or specifications.
- The blind noise residual adapts the permissive Apache-2.0 Noisesniffer
  publication; TAESD and the CLIP/runtime artifacts are permissive external
  dependencies. The face model is an Apache-2.0 catalogued ONNX artifact.
- TruFor, Noiseprint++, and Comprint remain excluded under D6 because the
  available implementations are nonprofit-use-only and lack a verified
  compatible ONNX export/direct weight path for this runtime. No substitute
  code or weights are claimed.
- IMD2020 and the inpainting candidate remain local-only because the published
  download pages provide no explicit redistribution licence; the inpainting
  candidate also failed its metadata gate.

No image, model, calibration, manifest, script, detector module, or sample was
changed by this round. Concurrent R17A/R17B work is visible in the working
tree (`MANIFEST.yaml`, `benchmark.py`, `calibrate.py`, their new test/report,
and `check_content_shortcut.py`); those out-of-scope changes were preserved and
not edited.

## Files reconciled

- `docs/detection-principles.md`: early shortcut warning, current per-detector
  evidence, native/parity scope, AEROBLADE/CLIP qualification, and licensing.
- `docs/corpus.md`: parity quality caveat, R15 matched-pair and inpainting
  findings, current scope status, and retired CLIP wording.
- `docs/calibration.md`: pre-refit artifact warning and variant-aware current
  measurement record.
- `docs/index.md` and `README.md`: prominent links and accuracy caveats.
- `plan/reference/detector-catalog.yaml`: catalog version 3 and the complete
  current measurement map with per-row corpus, variant, date, and SE.

## Checks

The catalog was parsed with the repository's Python YAML loader, and the final
diff was checked for whitespace errors. The working tree was checked against
the seven requested paths; concurrent out-of-scope changes listed above were
left untouched. No commit was created.
