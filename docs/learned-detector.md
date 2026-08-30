# Optional learned detector

Install `requirements-learned.txt`, then run `python scripts/fetch_model.py`. The Apache-2.0 model is pinned to a Hugging Face revision and is stored under ignored `models/`. The adapter first runs OpenCV's bundled `haarcascade_frontalface_default.xml` on the uploaded image. With no detectable face, it returns `not_applicable` before checking the optional runtime or weights. Without the extra or its weights, it is also `not_applicable`; the default service does not depend on it.

This is a face-deepfake detector, not a general AI-generation detector. Round
11 measured an AI-axis AUC of `0.423853 +/- 0.136642` after the face gate, so
calibration assigns it zero fusion weight. The negative scope is five
applicable `real_camera` images; the 109 applicable generated images come from
the eleven named generators and are not image-level paired with those camera
negatives. No torch is required.

The model is trained for face deepfake detection. It is not a general splice detector and is not a receipt/document forgery detector.
