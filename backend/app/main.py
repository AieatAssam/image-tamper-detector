"""FastAPI application and versioned API surface."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from PIL import Image

from backend.app.api.endpoints import api_router
from backend.app.config import settings
from backend.app.logging_config import RequestIdMiddleware, configure_logging

configure_logging()
Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS


app = FastAPI(
    title="Image Tampering Detection API",
    description="Image tampering detection using ELA, noise residual, and entropy analysis.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(RequestIdMiddleware)
# FastAPI 0.141 lazily wraps included routers; these routes are dependency-free,
# so register the concrete routes to keep introspection and dispatch identical.
app.router.routes.extend(api_router.routes)


@app.middleware("http")
async def reject_removed_routes(request: Request, call_next):
    if request.url.path in {"/analyze/" + suffix for suffix in ("ela", "prnu", "entropy", "combined")}:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=410,
            content={"detail": "This endpoint was removed; use /api/v1/analyze."},
            headers={"Deprecation": "true"},
        )
    return await call_next(request)


@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Image Tampering Detection API",
        "version": app.version,
        "endpoints": [
            "/api/v1/analyze",
            "/api/v1/detectors",
            "/healthz",
        ],
    }


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    app.openapi_schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
    return app.openapi_schema


app.openapi = custom_openapi
