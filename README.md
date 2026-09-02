# Image Tamper Detector

Experimental image-forensics service. It reports calibrated signals, not proof of origin or editing history. The implementation plan in [`plan/plan.yaml`](plan/plan.yaml) is authoritative.

Accuracy warning: no AI-generation AUC in this repository is an open-web
accuracy claim. The current corpus fails the metadata shortcut gate (held-out
AUC `0.8750 +/- 0.0598`), so Round 10's per-generator table and Round 12's
CLIP `1.0000` result are retired or qualified. The R15C parity variant passes
the metadata ablations, but its R16A CLIP result still rounds to 1.000 with
only 12 real-camera negatives and remains a content/corpus confound. See
[`docs/detection-principles.md`](docs/detection-principles.md) before quoting
any detector number.

## Run locally

The repository pins Python 3.13.13 in `.python-version` for the usable local runtime:

```bash
pyenv install 3.13.13
pyenv local 3.13.13
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Start the API with `.venv/bin/uvicorn backend.app.main:app --reload --port 8000`. The frontend uses Node 24 and runs with `cd frontend && npm ci && npm run dev`.

Run the checks with `.venv/bin/python -m pytest backend/tests -q` and `cd frontend && npm run test && npm run build`.

## API

- `GET /healthz` — liveness.
- `GET /api/v1/detectors` — available detector metadata and limitations.
- `POST /api/v1/analyze` — multipart upload with `file`; optional comma-separated `detectors` and `include_maps=false`.

The response contains a three-valued result for each detector (`applicable`, `not_applicable`, or `error`) and a calibrated weighted-logit fusion. A detector that cannot speak about an image does not count as evidence.

## Detector limits

ELA is meaningful only for JPEG input. The noise-residual detector is not camera attribution: real PRNU attribution requires a reference fingerprint from the same camera. C2PA absence is not evidence of tampering. The optional learned model is face-deepfake-specific and is not a general splice, document, or receipt detector.

The service has `native` and `parity` evidence variants. AI-axis detectors are
validated on parity; provenance and JPEG-history detectors use native bytes.
The upload path is not yet variant-aware, so this scope is a documented serving
precondition, not an automatic guarantee.

The committed calibration artifact is the Round 19 consolidation refit over
both variants: fused held-out AUC `0.6079881656804734` on 1022 held-out rows
from a 1530-row, partly synthetic corpus. It is not an AI-generation accuracy
claim; the corpus is small and is not representative of the open web. The
detector repairs behind this refit are recorded in
[`plan/audit/REPAIR-REPORT-R19.md`](plan/audit/REPAIR-REPORT-R19.md). See
[`docs/corpus.md`](docs/corpus.md) and [`docs/calibration.md`](docs/calibration.md).

## Corpus and optional model

Regenerate the deterministic synthetic corpus with:

```bash
.venv/bin/python scripts/make_corpus.py --seed 20260828 --out data/corpus/synthetic
.venv/bin/python scripts/benchmark.py --out /tmp/benchmark.json --corpus synthetic
```

The optional local IMD2020 archive contains real-life manipulated images with
binary masks and corresponding real images. Cite Novozamsky, Mahdian, and
Saic, “IMD2020: A Large-Scale Annotated Dataset Tailored for Detecting
Manipulated Images,” IEEE WACV Workshops 2020. It is downloaded to the
gitignored `data/corpus/imd2020/` directory with
`.venv/bin/python scripts/fetch_imd2020.py --download`; image bytes are never
committed because the publication provides no explicit redistribution license.
The real-life archive has no machine-readable manipulation-type mapping, so
the sample is stratified by its verified source directories instead. The
current reproducible sample uses 200 manipulated pairs (400 rows), one pair
per source directory, with a per-source cap of 2 and seed `20260828`.

The optional ONNX learned detector is documented in [`docs/learned-detector.md`](docs/learned-detector.md). It is not installed or enabled by default.

## Container

`docker build -t image-tamper-detector . && docker run --rm -p 8080:80 image-tamper-detector` serves the frontend and proxies the API through Nginx. The container runs the application as a non-root user.
