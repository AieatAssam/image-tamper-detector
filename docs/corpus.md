# Corpus and benchmark

S05 has two distinct corpus roles:

- `data/corpus/synthetic/` measures processing-history cues: authentic recompression, splices, copy-move, double JPEG compression, local retouching, and resized authentic content. `index.json` and JSON sidecars are reviewable; image and mask bytes are reproducible and ignored by Git.
- `data/corpus/MANIFEST.yaml` defines optional real-image downloads. `scripts/fetch_corpus.py` verifies both SHA-256 and byte count, and refuses mismatches.

The synthetic corpus cannot validate sensor provenance. A generator can faithfully synthesise PROCESSING HISTORY (splices, recompression, copy-move, quality changes) but CANNOT synthesise SENSOR PROVENANCE. Re-splicing one Unsplash JPEG creates no genuine Bayer interpolation structure and no genuine sensor noise. Therefore cfa_periodicity, spectral_peaks and the noise-residual detector MUST be validated against real images, never against generated splices.

The current manifest contains one verified real-camera source. It is a deliberate shortfall: Wikimedia was rate-limiting additional downloads during this run, and no checksum, attribution, camera model, or strict EXIF-dimension claim is invented. The required real-camera, real-AI, and C2PA counts remain open until independently fetched sources meet their criteria. Real-corpus benchmark metrics are therefore incomplete; synthetic results are still valid Tier A only for the detector families they validate.

## Commands

```bash
pyenv local 3.13.13
.venv/bin/python scripts/make_corpus.py --seed 20260828 --out data/corpus/synthetic --seed-images data/samples/original
.venv/bin/python scripts/fetch_corpus.py
.venv/bin/python scripts/fetch_corpus.py --check
.venv/bin/python scripts/benchmark.py --out /tmp/bench.json --corpus all
```

Use `--corpus synthetic` for an offline benchmark. Use `--detectors ela,prnu,entropy` to select a subset. The benchmark writes a JSON contract and a matching Markdown table. Its default output is deterministic, including the fixed report timestamp and timing fields, so unchanged runs can be compared byte-for-byte.
