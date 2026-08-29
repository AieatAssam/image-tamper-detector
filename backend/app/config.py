"""Environment-backed application settings."""

from dataclasses import dataclass
import os


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


@dataclass(frozen=True)
class Settings:
    MAX_UPLOAD_BYTES: int = _int_env("MAX_UPLOAD_BYTES", 12_000_000)
    MAX_IMAGE_PIXELS: int = _int_env("MAX_IMAGE_PIXELS", 50_000_000)
    ALLOWED_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG", "WEBP", "TIFF"})
    ALLOWED_ORIGINS: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:8000"
        ).split(",")
        if origin.strip()
    )
    MAX_CONCURRENT_ANALYSES: int = _int_env("MAX_CONCURRENT_ANALYSES", 4)
    RATE_LIMIT_PER_MINUTE: int = _int_env("RATE_LIMIT_PER_MINUTE", 30)
    ANALYSIS_TIMEOUT_SECONDS: int = _int_env("ANALYSIS_TIMEOUT_SECONDS", 60)


settings = Settings()
