"""
Cross-cutting request handling: identity, correlation, and bounds.

Each of these is here rather than in a route because it has to apply to every
route, including the ones added next.
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ragu.api.auth import authorize
from ragu.api.config import ServiceSettings
from ragu.api.errors import (
    PayloadTooLargeError,
    RaguServiceError,
    TooManyRequestsError,
)
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


class BodyLimitMiddleware:
    """
    Refuse a body larger than the service will read.

    Ingestion takes whole documents, so the ceiling is generous; without one it
    is unbounded.

    Raw ASGI rather than ``BaseHTTPMiddleware``, because ``Content-Length`` is
    only a hint: a chunked request carries none at all, and trusting the header
    alone leaves the one unbounded read this class exists to prevent. Counting
    the bytes as they arrive means wrapping ``receive``, which the higher-level
    base class does not expose.
    """

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    def _error(self, size: str | int) -> PayloadTooLargeError:
        return PayloadTooLargeError(
            f"Request body of {size} bytes exceeds the limit of {self.max_bytes}."
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = Headers(scope=scope).get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            # Refused before a byte is read, which is the whole point of the
            # header when a client sends an honest one.
            error = self._error(declared)
            response = JSONResponse(
                status_code=error.status_code, content=error.to_envelope()
            )
            await response(scope, receive, send)
            return

        received = 0
        over = False
        answered = False

        async def counting_receive() -> Message:
            nonlocal received, over
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    over = True
                    # End the body here rather than raising: an exception thrown
                    # into the body parser is caught there and reported as a
                    # malformed request, which is not what happened. The
                    # endpoint sees a truncated body, and its answer is replaced
                    # below with the real reason.
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal answered
            if over and not answered:
                return
            answered = True
            await send(message)

        await self.app(scope, counting_receive, guarded_send)

        if over and not answered:
            error = self._error(f"more than {self.max_bytes}")
            response = JSONResponse(
                status_code=error.status_code, content=error.to_envelope()
            )
            await response(scope, receive, send)


def _route_template(request: Request) -> str:
    """
    The templated path of the matched route, for use as a metric label.

    ``scope["route"].path`` carries only the sub-router's own path, so the two
    mounts of the search router would collapse onto one another. The template is
    rebuilt from the path parameters instead, which also keeps a graph id or a
    job id from opening a time series of its own.

    Substitution matches whole segments and scans from the right. A graph may
    legally be named ``v1``, ``s`` or ``graphs`` — ``GraphSpec.id`` allows a
    single character — and a parameter always sits after the literal prefix it
    follows, so the rightmost matching segment is the parameter's own. Replacing
    the first occurrence anywhere in the path instead would rewrite the prefix
    and leave the real id in the label, which is the cardinality this function
    exists to bound.

    :param request: The request being handled.
    :return: A path with ``{name}`` in place of every path parameter.
    """
    params = request.scope.get("path_params") or {}
    if not params:
        return request.url.path

    segments = request.url.path.split("/")
    unmatched = {name: str(value) for name, value in params.items() if str(value)}
    for index in range(len(segments) - 1, -1, -1):
        for name, value in unmatched.items():
            if segments[index] == value:
                segments[index] = "{" + name + "}"
                del unmatched[name]
                break
        if not unmatched:
            break
    return "/".join(segments)


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


class Admission:
    """
    A ceiling on how many generations run at once.

    Without one, every accepted request fans straight out to the LLM and the
    provider's rate limit becomes the queue — with the queue held open inside
    this process, holding memory and a socket per waiting request. Refusing at
    the door is cheaper for everyone.
    """

    def __init__(self, limit: int | None):
        self._semaphore = asyncio.Semaphore(limit) if limit else None

    @asynccontextmanager
    async def slot(self):
        """
        Hold a generation slot for the duration of the block.

        :raises TooManyRequestsError: If none is free.
        """
        if self._semaphore is None:
            yield
            return
        if self._semaphore.locked():
            raise TooManyRequestsError(
                "The service is already running as many generations as it will. "
                "Retry shortly."
            )
        await self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()
