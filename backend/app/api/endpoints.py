"""Versioned API endpoints and upload validation shared with legacy routes."""

import asyncio
import base64
import logging
from collections import defaultdict, deque
from hashlib import sha256
from io import BytesIO
from time import monotonic
from typing import Any

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.app.analysis.base import DetectorResult, DetectorState, ImageContext
from backend.app.analysis.fusion import fuse
from backend.app.analysis.registry import DEFAULT_ENABLED, get_all, get as get_detectors, run_all
from backend.app.config import settings

logger = logging.getLogger(__name__)
api_router = APIRouter()
analysis_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_ANALYSES)
_request_times: dict[str, deque[float]] = defaultdict(deque)


async def read_image_context(file: UploadFile) -> ImageContext:
    data = bytearray()
    while chunk := await file.read(65_536):
        if len(data) + len(chunk) > settings.MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image upload exceeds the maximum allowed size")
        data.extend(chunk)
    raw = bytes(data)
    try:
        with Image.open(BytesIO(raw)) as image:
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            image_format = (image.format or "").upper()
            if image_format not in settings.ALLOWED_FORMATS:
                raise HTTPException(status_code=415, detail="Unsupported image format")
            image.load()
            decoded = image.copy()
    except HTTPException:
        raise
    except Exception:
        logger.exception("image decoder rejected upload")
        raise HTTPException(status_code=415, detail="Uploaded file is not a valid supported image")
    return ImageContext(raw, _pil_image=decoded, _format=image_format)


async def run_analysis(context: ImageContext, detector_ids: list[str] | None = None) -> list[DetectorResult]:
    try:
        async with analysis_semaphore:
            return await asyncio.wait_for(
                run_in_threadpool(run_all, context, detector_ids),
                timeout=settings.ANALYSIS_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError as exc:
        logger.error("analysis timed out after %ss", settings.ANALYSIS_TIMEOUT_SECONDS)
        raise HTTPException(status_code=504, detail="Image analysis timed out") from exc


def _rate_limit(request: Request) -> None:
    # single-process only; use a shared store if this ever runs multi-worker
    now = monotonic()
    client = request.client.host if request.client else "unknown"
    timestamps = _request_times[client]
    while timestamps and timestamps[0] <= now - 60:
        timestamps.popleft()
    if len(timestamps) >= settings.RATE_LIMIT_PER_MINUTE:
        retry_after = max(1, int(60 - (now - timestamps[0])))
        raise HTTPException(
            status_code=429,
            detail="Analysis rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )
    timestamps.append(now)


class DetectorInfo(BaseModel):
    id: str
    name: str
    family: str
    applicable_formats: list[str]
    produces_map: bool
    description: str
    limitations: list[str]
    enabled: bool


class DetectorListResponse(BaseModel):
    detectors: list[DetectorInfo]


class DetectorResponse(BaseModel):
    id: str
    state: DetectorState
    flagged: bool | None
    score: float | None = Field(None, ge=0, le=1)
    threshold: float
    reason: str
    metrics: dict[str, float]
    visualization_png_base64: str | None = None
    duration_ms: int
    error: str | None = None


class ImageResponse(BaseModel):
    width: int
    height: int
    format: str
    bytes: int
    sha256: str


class FusionContribution(BaseModel):
    id: str
    weight: float
    signed_contribution: float


class FusionResponse(BaseModel):
    method: str
    contributions: list[FusionContribution]
    calibration_version: str


class AnalyzeResponse(BaseModel):
    verdict: str
    score: float = Field(..., ge=0, le=1)
    summary: str
    image: ImageResponse
    detectors: list[DetectorResponse]
    fusion: FusionResponse
    warnings: list[str]


@api_router.get("/healthz", tags=["Info"])
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


@api_router.get("/api/v1/detectors", response_model=DetectorListResponse, tags=["Analysis"])
def detector_discovery() -> dict[str, list[dict[str, Any]]]:
    return {
        "detectors": [
            {
                "id": detector.id,
                "name": detector.name,
                "family": detector.family,
                "applicable_formats": sorted(format_name.lower() for format_name in detector.applicable_formats),
                "produces_map": detector.produces_map,
                "description": detector.description,
                "limitations": detector.limitations,
                "enabled": detector.id in DEFAULT_ENABLED,
            }
            for detector in get_all().values()
        ]
    }


@api_router.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    detectors: str | None = Query(None, description="Comma-separated detector ids"),
    include_maps: bool = Query(True),
    _: None = Depends(_rate_limit),
):
    try:
        detector_ids = [item.strip() for item in detectors.split(",") if item.strip()] if detectors is not None else None
        if detector_ids is not None:
            get_detectors(detector_ids)
        context = await read_image_context(file)
        results = await run_analysis(context, detector_ids)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    fused = fuse(results)
    applicable = [result for result in results if result.state is DetectorState.APPLICABLE and result.score is not None]
    payload = AnalyzeResponse(
        verdict=fused["verdict"],
        score=float(fused["score"]),
        summary=_summary(fused["verdict"], applicable),
        image=ImageResponse(
            width=context.width,
            height=context.height,
            format=context.format,
            bytes=len(context.raw_bytes),
            sha256=sha256(context.raw_bytes).hexdigest(),
        ),
        detectors=[_detector_payload(result, include_maps) for result in results],
        fusion=FusionResponse(
            method=fused["method"],
            contributions=[FusionContribution(**contribution) for contribution in fused["contributions"]],
            calibration_version=fused["calibration_version"],
        ),
        warnings=[],
    ).model_dump()
    logger.info(
        "image analysis complete",
        extra={
            "image_sha256": context.sha256,
            "detector_durations": {result.detector_id: result.duration_ms for result in results},
            "verdict": fused["verdict"],
        },
    )
    if not include_maps:
        for item in payload["detectors"]:
            item.pop("visualization_png_base64", None)
        return JSONResponse(content=payload)
    return payload


def _detector_payload(result: DetectorResult, include_maps: bool) -> DetectorResponse:
    visualization = None
    if include_maps and result.visualization is not None:
        visualization = _encode_image_to_base64(result.visualization)
    return DetectorResponse(
        id=result.detector_id,
        state=result.state,
        flagged=result.flagged,
        score=result.score,
        threshold=result.threshold,
        reason=result.reason,
        metrics=result.metrics,
        visualization_png_base64=visualization,
        duration_ms=result.duration_ms,
        error=result.error,
    )


def _encode_image_to_base64(image_array) -> str:
    success, buffer = cv2.imencode(".png", image_array)
    if not success:
        raise ValueError("Failed to encode visualization")
    return base64.b64encode(buffer).decode("ascii")


def _verdict(score: float) -> str:
    if score < 0.15:
        return "authentic"
    if score < 0.35:
        return "likely_authentic"
    if score < 0.55:
        return "inconclusive"
    if score < 0.80:
        return "likely_manipulated"
    return "manipulated"


def _summary(verdict: str, results: list[DetectorResult]) -> str:
    if not results:
        return "Insufficient detector evidence to classify this image."
    top = max(results, key=lambda result: result.score or 0.0)
    return f"The analysis is {verdict.replace('_', ' ')}; {top.reason}."
