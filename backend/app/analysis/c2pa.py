"""C2PA provenance detector for c2pa-python 0.37.x."""

from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from c2pa import Reader

from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext


class C2PAAnalyzer:
    """Read signed C2PA manifests without treating missing metadata as a finding."""

    id = "c2pa"
    name = "C2PA Provenance"
    family = "metadata"
    applicable_formats = frozenset({"JPEG", "PNG", "WEBP", "TIFF"})
    produces_map = False
    description = "Reads signed Content Credentials and explicit provenance assertions."
    limitations = [
        "Most images contain no manifest; absence is not evidence.",
        "A valid manifest describes the signer’s assertion, not independent truth.",
    ]
    threshold = 0.5

    # Fallback only: structured digitalSourceType is the primary AI signal.
    known_ai_generators = {
        "dalle", "midjourney", "stable diffusion", "openai",
        "adobe firefly", "imagen", "deepmind",
    }

    def applicable(self, ctx: ImageContext) -> tuple[bool, str]:
        return True, "C2PA inspection supports this image format"

    def run(self, ctx: ImageContext) -> DetectorResult:
        started = perf_counter()
        result = self._analyze(ctx.raw_bytes, mime=_mime_for(ctx))
        return DetectorResult(
            self.id, result["state"], result["score"], result["flagged"],
            self.threshold, result["reason"], result["metrics"], None,
            _duration(started), result.get("error"),
        )

    def analyze_image(self, image_input: str | Path | bytes) -> dict[str, Any]:
        """Compatibility API returning the previous ``issues``/``metadata`` shape."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")
            if path.suffix.lower() not in self.get_supported_formats():
                raise ValueError(f"Unsupported file format: {path}")
            result = self._analyze(path.read_bytes(), path=path)
        elif isinstance(image_input, bytes):
            result = self._analyze(image_input, mime="image/jpeg")
        else:
            raise ValueError("Input must be an image path or bytes")
        return result

    def _analyze(
        self, raw: bytes, *, path: Path | None = None, mime: str | None = None
    ) -> dict[str, Any]:
        try:
            reader = Reader(path) if path is not None else Reader(mime or "application/octet-stream", BytesIO(raw))
            with reader as active_reader:
                store = json.loads(active_reader.json())
        except Exception as exc:
            if _is_missing_manifest(exc):
                return _not_applicable()
            return {
                "state": DetectorState.ERROR, "score": None, "flagged": None,
                "reason": "C2PA manifest could not be read", "metrics": {},
                "issues": [{"type": "analysis_error", "severity": "error",
                            "description": "C2PA manifest could not be read",
                            "location": "manifest"}], "metadata": {},
                "error": "C2PA reader failure",
            }

        active_id = store.get("active_manifest")
        manifests = store.get("manifests") or {}
        manifest = manifests.get(active_id) if isinstance(manifests, dict) else None
        if not isinstance(manifest, dict):
            return _not_applicable()

        validation_failed = _validation_failed(store)
        generator = _generator(manifest)
        generated = _has_generative_action(manifest)
        if not generated and generator:
            generated = any(name in generator.lower() for name in self.known_ai_generators)

        metadata = {
            "active_manifest": active_id,
            "claim_generator": generator,
            "manifest": manifest,
            "validation_status": store.get("validation_status"),
            "validation_results": store.get("validation_results"),
        }
        issues = []
        if validation_failed:
            issues.append({"type": "signature_invalid", "severity": "high",
                           "description": "C2PA validation failed; post-signing modification is possible",
                           "location": "manifest.validation"})
        if generated:
            issues.append({"type": "ai_generated", "severity": "identification",
                           "description": "Validated C2PA manifest identifies generative image creation",
                           "location": "manifest.assertions.c2pa.actions"})

        if validation_failed:
            score, flagged = 0.95, True
            reason = "C2PA manifest is present but validation failed"
        elif generated:
            score, flagged = 1.0, True
            suffix = f" by {generator}" if generator else ""
            reason = f"validated C2PA manifest identifies generative image creation{suffix}"
        else:
            score, flagged = 0.05, False
            reason = "valid C2PA manifest contains no generative creation assertion"
        return {
            "state": DetectorState.APPLICABLE, "score": score, "flagged": flagged,
            "reason": reason,
            "metrics": {"manifest_present": 1.0, "validation_failed": float(validation_failed),
                        "generative_assertion": float(generated)},
            "issues": issues, "metadata": metadata,
        }

    def get_supported_formats(self) -> list[str]:
        return [".jpg", ".jpeg", ".png", ".tiff", ".webp"]


def _not_applicable() -> dict[str, Any]:
    return {"state": DetectorState.NOT_APPLICABLE, "score": None, "flagged": None,
            "reason": "no C2PA manifest", "metrics": {}, "issues": [], "metadata": {}}


def _is_missing_manifest(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "manifestnotfound" in text or "no jumbf" in text or "manifest not found" in text


def _validation_failed(store: dict[str, Any]) -> bool:
    text = json.dumps([store.get("validation_status"), store.get("validation_results")], default=str).lower()
    return any(token in text for token in ("fail", "invalid", "error"))


def _generator(manifest: dict[str, Any]) -> str | None:
    for key in ("claim_generator", "claimGenerator"):
        value = manifest.get(key)
        if isinstance(value, str) and value:
            return value
    info = manifest.get("claim_generator_info") or manifest.get("claimGeneratorInfo")
    items = info if isinstance(info, list) else [info]
    for item in items:
        if isinstance(item, dict) and item.get("name"):
            return str(item["name"])
    return None


def _has_generative_action(value: Any) -> bool:
    if isinstance(value, dict):
        source = str(value.get("digitalSourceType", ""))
        if value.get("action") == "c2pa.created" and (
            source == "trainedAlgorithmicMedia" or source.endswith("/trainedAlgorithmicMedia")
        ):
            return True
        return any(_has_generative_action(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_generative_action(item) for item in value)
    return False


def _mime_for(ctx: ImageContext) -> str:
    if ctx.raw_bytes.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if ctx.raw_bytes.startswith(b"\x89PNG"):
        return "image/png"
    if ctx.raw_bytes.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    return "image/jpeg"


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
