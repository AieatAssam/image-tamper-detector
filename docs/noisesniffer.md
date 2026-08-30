# Noisesniffer attribution

The noise-residual detector adapts the block-selection and a-contrario NFA
method from:

> Marina Gardella, Pablo Musé, Miguel Colom, and Jean-Michel Morel, “Image
> Forgery Detection Based on Noise Inspection: Analysis and Refinement of the
> Noisesniffer Method,” *Image Processing On Line* 14 (2024), article 462.

Reference article and source: <https://www.ipol.im/pub/art/2024/462/>.

The IPOL reference files `Noisesniffer.py` and `functions.py` are licensed under
the Apache License, Version 2.0. This repository's
`backend/app/analysis/prnu.py` is a modified adaptation under that license,
integrated with the repository's detector protocol and array inputs. The
license text is available at <https://www.apache.org/licenses/LICENSE-2.0>.

This detector is a blind noise-inconsistency cue. It is not camera-sensor PRNU
attribution, which requires a reference fingerprint from the same camera.
