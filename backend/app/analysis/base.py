"""Shared detector protocol, image context, and score mapping."""

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image


class DetectorState(str, Enum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


@dataclass(frozen=True)
class DetectorResult:
    detector_id: str
    state: DetectorState
    score: float | None
    flagged: bool | None
    threshold: float
    reason: str
    metrics: dict[str, float]
    visualization: np.ndarray | None
    duration_ms: int
    error: str | None = None


@dataclass
class ImageContext:
    """One validated upload and cached derivations shared by all detectors."""

    raw_bytes: bytes
    _pil_image: Image.Image | None = field(default=None, repr=False)
    _format: str | None = field(default=None, repr=False)
    _rgb_uint8: np.ndarray | None = field(default=None, repr=False)
    _gray_uint8: np.ndarray | None = field(default=None, repr=False)
    _downscaled_rgb_uint8: np.ndarray | None = field(default=None, repr=False)
    _exif: dict[str, Any] | None = field(default=None, repr=False)

    @classmethod
    def from_path(cls, path: str | Path) -> "ImageContext":
        return cls(Path(path).read_bytes())

    @property
    def pil_image(self) -> Image.Image:
        if self._pil_image is None:
            with Image.open(BytesIO(self.raw_bytes)) as image:
                self._format = (image.format or "").upper()
                self._pil_image = image.convert("RGB")
                self._pil_image.load()
        return self._pil_image

    @property
    def rgb_uint8(self) -> np.ndarray:
        if self._rgb_uint8 is None:
            self._rgb_uint8 = np.asarray(self.pil_image.convert("RGB"), dtype=np.uint8).copy()
        return self._rgb_uint8

    @property
    def gray_uint8(self) -> np.ndarray:
        if self._gray_uint8 is None:
            image = self.rgb_uint8
            # Import here to keep the context's core dependencies small.
            import cv2

            self._gray_uint8 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return self._gray_uint8

    @property
    def downscaled_rgb_uint8(self) -> np.ndarray:
        if self._downscaled_rgb_uint8 is None:
            image = self.pil_image.convert("RGB")
            if max(image.size) > 1600:
                ratio = 1600 / max(image.size)
                image = image.resize(
                    tuple(max(1, int(dimension * ratio)) for dimension in image.size),
                    Image.Resampling.LANCZOS,
                )
            self._downscaled_rgb_uint8 = np.asarray(image, dtype=np.uint8).copy()
        return self._downscaled_rgb_uint8

    @property
    def format(self) -> str:
        self.pil_image
        return (self._format or "").upper()

    @property
    def width(self) -> int:
        return self.pil_image.width

    @property
    def height(self) -> int:
        return self.pil_image.height

    @property
    def sha256(self) -> str:
        return sha256(self.raw_bytes).hexdigest()

    @property
    def exif(self) -> dict[str, Any]:
        if self._exif is None:
            self._exif = dict(self.pil_image.getexif())
        return self._exif


class Detector(Protocol):
    id: str
    name: str
    family: str
    applicable_formats: frozenset[str]
    produces_map: bool
    description: str
    limitations: list[str]

    def applicable(self, ctx: ImageContext) -> tuple[bool, str]: ...

    def run(self, ctx: ImageContext) -> DetectorResult: ...


def to_probability(raw: float, threshold: float, scale: float, higher_is_worse: bool) -> float:
    """Map a calibrated raw statistic to P(manipulated | detector evidence)."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    distance = (raw - threshold) / scale
    if not higher_is_worse:
        distance = -distance
    distance = max(-60.0, min(60.0, distance))
    return 1.0 / (1.0 + float(np.exp(-distance)))
