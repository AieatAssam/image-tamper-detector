# Image Tamper Detector

Experimental image-forensics service. It reports calibrated signals, not proof of origin or editing history. The implementation plan in [`plan/plan.yaml`](plan/plan.yaml) is authoritative.

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

The committed calibration reports a held-out AUC of 0.855 on 30 deterministic synthetic holdout images. That is a small, partly synthetic corpus: the result is valid only for images resembling it and is not representative of the open web. See [`docs/corpus.md`](docs/corpus.md) and [`docs/calibration.md`](docs/calibration.md).

## Corpus and optional model

Regenerate the deterministic synthetic corpus with:

```bash
.venv/bin/python scripts/make_corpus.py --seed 20260828 --out data/corpus/synthetic
.venv/bin/python scripts/benchmark.py --out /tmp/benchmark.json --corpus synthetic
```

The optional ONNX learned detector is documented in [`docs/learned-detector.md`](docs/learned-detector.md). It is not installed or enabled by default.

## Container

`docker build -t image-tamper-detector . && docker run --rm -p 8080:80 image-tamper-detector` serves the frontend and proxies the API through Nginx. The container runs the application as a non-root user.
