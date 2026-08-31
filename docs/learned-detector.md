# Optional learned detector

Install `requirements-learned.txt`, then run `python scripts/fetch_model.py face`. The Apache-2.0 face model is pinned to a Hugging Face revision and is stored under ignored `models/`. The optional TAESD/LPIPS and CLIP artifacts are fetched with `python scripts/fetch_model.py taesd` and `python scripts/fetch_model.py clip`. The adapter first runs OpenCV's bundled `haarcascade_frontalface_default.xml` on the uploaded image. With no detectable face, it returns `not_applicable` before checking the optional runtime or weights. Without the extra or its weights, it is also `not_applicable`; the default service does not depend on it.

This is a face-deepfake detector, not a general AI-generation detector. Round
11 measured an AI-axis AUC of `0.423853 +/- 0.136642` after the face gate, so
calibration assigns it zero fusion weight. The negative scope is five
applicable `real_camera` images; the 109 applicable generated images come from
the eleven named generators and are not image-level paired with those camera
negatives. The ONNX face model itself does not require torch, but the optional
learned extra also pins torch and torchvision for other learned detectors.
Neither is installed by `requirements.txt`, CI, or the Docker image.

The model is trained for face deepfake detection. It is not a general splice detector and is not a receipt/document forgery detector.

## Frozen CLIP probe

`clip_probe` uses the frozen MIT-licensed LAION ViT-L/14 backbone through
open-clip-torch and a corpus-fitted linear probe. `scripts/fit_clip_probe.py`
groups the manifest by `source_image` and holds out complete generators; it
reports both seen-generator ID and unseen-generator OOD AUC. Round 12 held out
`glide`, `stable-diffusion-1-4`, `stable-diffusion-3.5-medium`, and
`stable-diffusion-xl`. The result was `1.0000 +/- 0.0000` for both ID and OOD,
with four `real_camera` negatives in each test partition. Because every AI row
is PNG and every strict camera negative is JPEG, this separation may include a
format/domain cue. The result is therefore recorded as a measurement, not a
floor or a universal claim.

The CLIP backbone and probe are optional, image-side gated, and never loaded
from manifest metadata. Missing dependencies, backbone, or probe return
`not_applicable`; the default requirements, CI, and Docker image remain
torch-free.
