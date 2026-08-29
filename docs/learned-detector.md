# Optional learned detector

Install `requirements-learned.txt`, then run `python scripts/fetch_model.py`. The Apache-2.0 model is pinned to a Hugging Face revision and is stored under ignored `models/`. Without the extra or its weights, the detector is registered but returns `not_applicable`; the default service does not depend on it.

The model is trained for face deepfake detection. It is not a general splice detector and is not a receipt/document forgery detector.
