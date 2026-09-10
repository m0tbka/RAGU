"""
Cross-cutting request handling: identity, correlation, and bounds.

Each of these is here rather than in a route because it has to apply to every
route, including the ones added next.
"""

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from ragu.api.auth import authorize
from ragu.api.config import ServiceSettings
from ragu.api.errors import PayloadTooLargeError, RaguServiceError
from ragu.api.metrics import DURATION, REQUESTS, metrics
from ragu.api.request_context import (
    REQUEST_ID_HEADER,
    reset_request_id,
    set_request_id,
)
from ragu.common.logger import logger

# Paths that must answer without a key, or an orchestrator cannot probe the
# service and a client cannot discover how to talk to it.
PUBLIC_PATHS = frozenset(
    {
        "/health",
        "/health/live",
        "/health/ready",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Give every request an id, and carry it into the response and the log.

    A 500 says only "see the service log"; without a shared id there is no way
    to find the line it refers to.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        token = set_request_id(request_id)
        try:
            with logger.contextualize(request_id=request_id):
                response = await call_next(request)
        finally:
            reset_request_id(token)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Require an API key on everything but the probes and the docs.
    """

    def __init__(self, app: FastAPI, settings: ServiceSettings):
        super().__init__(app)
        self.settings = settings

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path not in PUBLIC_PATHS:
            try:
                authorize(request, self.settings)
            except RaguServiceError as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    content=exc.to_envelope(),
                    headers=exc.headers or None,
                )
        return await call_next(request)


class BodyLimitMiddleware(BaseHTTPMiddleware):
    """
    Refuse a body larger than the service will read.

    Ingestion takes whole documents, so the ceiling is generous; without one it
    is unbounded.
    """

    def __init__(self, app: FastAPI, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit():
            if int(declared) > self.max_bytes:
                error = PayloadTooLargeError(
                    f"Request body of {declared} bytes exceeds the limit of "
                    f"{self.max_bytes}."
                )
                return JSONResponse(
                    status_code=error.status_code, content=error.to_envelope()
                )
        return await call_next(request)


def _route_template(request: Request) -> str:
    """
    The templated path of the matched route, for use as a metric label.

    ``scope["route"].path`` carries only the sub-router's own path, so the two
    mounts of the search router would collapse onto one another. The template is
    rebuilt from the path parameters instead, which also keeps a graph id or a
    job id from opening a time series of its own.

    :param request: The request being handled.
    :return: A path with ``{name}`` in place of every path parameter.
    """
    path = request.url.path
    for name, value in (request.scope.get("path_params") or {}).items():
        text = str(value)
        if text:
            path = path.replace(text, "{" + name + "}", 1)
    return path


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Count and time every request.

    The label is the route template, not the URL: ``/v1/graphs/{graph_id}/...``
    keeps the series count bounded no matter how many graphs or queries there
    are.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        started = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            path = _route_template(request)
            elapsed = time.perf_counter() - started
            metrics.increment(
                REQUESTS,
                (("method", request.method), ("path", path), ("status", status)),
            )
            metrics.observe(DURATION, elapsed, (("path", path),))
