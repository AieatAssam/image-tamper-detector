from io import BytesIO
from math import log10

import numpy as np
from PIL import Image
from scipy.stats import binom

from backend.app.analysis.base import DetectorState, ImageContext
from backend.app.analysis.zero import ZeroDetector, _log10_nfa, _vote_map


def _encoded(image: Image.Image, format: str, quality: int = 70) -> bytes:
    output = BytesIO()
    options = {"format": format}
    if format == "JPEG":
        options["quality"] = quality
    image.save(output, **options)
    return output.getvalue()


def test_zero_finds_a_foreign_grid_and_keeps_clean_jpeg_low():
    rng = np.random.default_rng(123)
    host = Image.fromarray(rng.integers(0, 256, (256, 384, 3), dtype=np.uint8))
    donor = Image.fromarray(rng.integers(0, 256, (256, 384, 3), dtype=np.uint8))
    host_jpeg = Image.open(BytesIO(_encoded(host, "JPEG"))).convert("RGB")
    donor_jpeg = Image.open(BytesIO(_encoded(donor, "JPEG"))).convert("RGB")

    detector = ZeroDetector()
    clean = detector.run(ImageContext(_encoded(host, "JPEG")))
    splice = host_jpeg.copy()
    splice.paste(donor_jpeg.crop((101, 103, 301, 203)), (100, 100))
    forged = detector.run(ImageContext(_encoded(splice, "PNG")))

    assert clean.state is DetectorState.APPLICABLE
    assert forged.state is DetectorState.APPLICABLE
    assert clean.score < 0.5 < forged.score
    assert forged.metrics["dominant_phase"] == 0
    assert forged.metrics["foreign_region_count"] >= 1
    assert forged.visualization is not None
    assert forged.visualization.shape == (256, 384)
    assert forged.visualization.dtype == np.uint8
    assert np.count_nonzero(forged.visualization) > 0
    assert all(np.isfinite(value) for value in forged.metrics.values())


def test_zero_is_not_applicable_to_tiny_images():
    image = Image.new("RGB", (31, 31), "gray")
    result = ZeroDetector().run(ImageContext(_encoded(image, "PNG")))

    assert result.state is DetectorState.NOT_APPLICABLE
    assert result.score is None
    assert result.flagged is None


def test_zero_votes_are_pixel_level_and_border_is_invalid():
    gray = np.random.default_rng(4).integers(0, 256, (64, 80)).astype(np.float32)
    votes, zero_counts = _vote_map(gray)
    assert votes.shape == gray.shape
    assert zero_counts.shape == gray.shape
    assert np.all(votes[:7] == -1)
    assert np.all(votes[-7:] == -1)
    assert np.all(votes[:, :7] == -1)
    assert np.all(votes[:, -7:] == -1)


def test_zero_nfa_uses_paper_conservative_subsampling_factor():
    expected = 2 * log10(64) + 2 * log10(64 * 64) + float(binom.logsf(0, 1, 1 / 64)) / np.log(10)
    assert np.isclose(_log10_nfa(64, 64, (64, 64)), expected)
