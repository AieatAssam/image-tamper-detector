"""C2PA provenance detector for c2pa-python 0.37.x."""

from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from c2pa import Context, Reader

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
            result = self._analyze(image_input, mime=_mime_for_bytes(image_input))
        else:
            raise ValueError("Input must be an image path or bytes")
        return result

    def _analyze(
        self, raw: bytes, *, path: Path | None = None, mime: str | None = None
    ) -> dict[str, Any]:
        try:
            with Context() as context:
                reader = (
                    Reader(path, context=context)
                    if path is not None
                    else Reader(mime or "application/octet-stream", BytesIO(raw), context=context)
                )
                with reader as active_reader:
                    store = json.loads(active_reader.json())
                    state = active_reader.get_validation_state()
                    if isinstance(state, str) and state and "validation_state" not in store:
                        store["validation_state"] = state
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

        validation_state = _validation_state(store)
        validation_failed = validation_state in {"invalid", "error"}
        if validation_state == "unknown" and _has_validation_problem(store):
            validation_failed = True
        generator = _generator(manifest)
        structured_generated = _has_generative_action(manifest)
        generated = validation_state in {"valid", "trusted"} and structured_generated
        if not generated and validation_state in {"valid", "trusted"} and generator:
            generated = any(name in generator.lower() for name in self.known_ai_generators)

        metadata = {
            "active_manifest": active_id,
            "claim_generator": generator,
            "manifest": manifest,
            "validation_state": validation_state,
            "trusted": validation_state == "trusted",
            "validation_status": store.get("validation_status"),
            "validation_results": store.get("validation_results"),
        }
        issues = []
        if validation_failed:
            failure_type, failure_description = _validation_issue(store, validation_state)
            issues.append({"type": failure_type, "severity": "high",
                           "description": failure_description,
                           "location": "manifest.validation"})
        elif validation_state == "unknown":
            issues.append({"type": "validation_unknown", "severity": "info",
                           "description": "C2PA manifest has no established validation state",
                           "location": "manifest.validation"})
        if generated:
            issues.append({"type": "ai_generated", "severity": "identification",
                           "description": "Validated C2PA manifest identifies generative image creation",
                           "location": "manifest.assertions.c2pa.actions"})

        if validation_state == "unknown" and not validation_failed:
            score, flagged = None, None
            reason = "C2PA manifest validation state is unavailable"
        elif validation_failed:
            score, flagged = 0.95, True
            reason = "C2PA manifest is present but validation failed"
        elif generated:
            score, flagged = 1.0, True
            suffix = f" by {generator}" if generator else ""
            reason = f"validated C2PA manifest identifies generative image creation{suffix}"
        elif validation_state in {"valid", "trusted"}:
            score, flagged = 0.05, False
            reason = "valid C2PA manifest contains no generative creation assertion"
        else:
            score, flagged = None, None
            reason = f"C2PA manifest is {validation_state}; validation is not sufficient for provenance scoring"
        return {
            "state": DetectorState.APPLICABLE, "score": score, "flagged": flagged,
            "reason": reason,
            "metrics": {"manifest_present": 1.0, "validation_failed": float(validation_failed),
                        "generative_assertion": float(structured_generated),
                        "validated_generative_assertion": float(generated),
                        "validation_known": float(validation_state != "unknown"),
                        "trusted": float(validation_state == "trusted")},
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


def _validation_state(store: dict[str, Any]) -> str:
    raw = store.get("validation_state")
    if isinstance(raw, str) and raw.strip():
        normalized = raw.strip().lower().replace("_", "-").replace(" ", "-")
        if normalized in {"well-formed", "valid", "trusted", "invalid", "error"}:
            return normalized
        return "unknown"
    return "unknown"


def _has_validation_problem(store: dict[str, Any]) -> bool:
    # A validation payload also lists successes, so presence alone is not a problem.
    # Only an explicit failure token counts when the library reports no usable state.
    text = json.dumps([store.get("validation_status"), store.get("validation_results")], default=str).lower()
    return any(token in text for token in ("fail", "invalid", "mismatch", "error", "untrusted"))


def _validation_issue(store: dict[str, Any], validation_state: str) -> tuple[str, str]:
    text = json.dumps([store.get("validation_status"), store.get("validation_results")], default=str).lower()
    if any(code in text for code in ("assertion.datahash.mismatch", "assertion.hasheduri.mismatch", "claimsignature.mismatch")):
        return "post_signing_mismatch", "C2PA validation found a signed content mismatch; post-signing modification is possible"
    if any(code in text for code in ("claimsignature", "cose")):
        return "signature_invalid", "C2PA signature validation failed"
    if "signingcredential" in text:
        return "credential_invalid", "C2PA signing credential validation failed"
    if validation_state == "unknown" and not _has_validation_problem(store):
        return "validation_unknown", "C2PA manifest has no established validation state"
    if validation_state == "unknown":
        return "validation_failed", "C2PA manifest validation failed; failure type is unspecified"
    return "validation_failed", "C2PA manifest validation failed"


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
    if isinstance(value, dict) and "assertions" in value:
        value = value.get("assertions", [])
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
    return _mime_for_bytes(ctx.raw_bytes, ctx.format)


def _mime_for_bytes(raw: bytes, image_format: str | None = None) -> str:
    if raw.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if raw.startswith(b"\x89PNG"):
        return "image/png"
    if raw.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"
    return {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "TIFF": "image/tiff",
        "WEBP": "image/webp",
    }.get((image_format or "").upper(), "application/octet-stream")


def _duration(started: float) -> int:
    return int((perf_counter() - started) * 1000)
