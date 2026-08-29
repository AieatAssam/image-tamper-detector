"""Small JSON-lines logging and request-id middleware."""

from __future__ import annotations

import contextvars
import json
import logging
import os
import uuid

from starlette.middleware.base import BaseHTTPMiddleware

request_id = contextvars.ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id.get(),
        }
        for key in ("image_sha256", "detector_durations", "verdict"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, sort_keys=True, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        value = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = request_id.set(value)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = value
            return response
        finally:
            request_id.reset(token)
