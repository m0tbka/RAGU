"""
FastAPI application factory.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ragu.api.backends.base import SearchBackend
from ragu.api.errors import RequestTimeoutError
from ragu.api.jobs import JobManager
from ragu.api.middleware import (
    Admission,
    AuthMiddleware,
    BodyLimitMiddleware,
    MetricsMiddleware,
    RequestContextMiddleware,
)
from ragu.api.registry import GraphRegistry
from ragu.api.request_context import REQUEST_ID_HEADER
from ragu.api.config import ServiceSettings
from ragu.api.errors import InvalidRequestError, RaguServiceError
from ragu.api.routes import router
from ragu.models.scorer import Scorer
from ragu.common.logger import logger

# Returned instead of the exception text: engine and LLM-client errors routinely
# quote the endpoint URL and parts of the request body.
OPENAPI_TAGS = [
    {"name": "search", "description": "Answering questions against a graph."},
    {"name": "graphs", "description": "The catalogue, and reads of graph structure."},
    {"name": "jobs", "description": "Ingestion and reindexing, which run too long for a request."},
    {"name": "service", "description": "Probes, metrics and the ontology."},
]

UNHANDLED_ERROR_MESSAGE = (
    "The service failed to handle this request. See the service log for details."
)


def create_app(
    settings: ServiceSettings | None = None,
    backend: SearchBackend | None = None,
    reranker: Scorer | None = None,
) -> FastAPI:
    """
    Build the service application.

    :param settings: Service settings; read from the environment when omitted.
    :param backend: Pre-built backend, used by tests and by in-process embedding
        to bypass the configured catalogue.
    :param reranker: Reranker shared by every graph. Passed in rather than built
        here: on a CPU-only deployment the model runs in its own container.
    :return: The configured application.
    """
    settings = settings or ServiceSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.startup_error = None
        registry = (
            GraphRegistry.of(settings, backend)
            if backend is not None
            else GraphRegistry(settings, reranker=reranker)
        )
        app.state.registry = registry
        app.state.jobs = JobManager()
        app.state.admission = Admission(settings.max_concurrent_generations)
        if not settings.api_keys_set():
            logger.warning(
                "No RAGU_API_API_KEYS configured: this service is open to anyone "
                "who can reach it, and every request costs LLM calls."
            )
        try:
            await registry.startup()
        except Exception as exc:
            # The service still starts so that /health can report *why* it is
            # not ready; without this the operator only ever sees a container
            # that restarts.
            app.state.startup_error = str(exc)
            logger.opt(exception=True).error("Registry startup failed: {}", exc)
        yield
        await app.state.jobs.shutdown()
        await registry.shutdown()

    app = FastAPI(
        title="RAGU Service",
        description=(
            "Graph-RAG over one or more knowledge graphs.\n\n"
            "Four search modes — `global`, `local`, `naive`, `mix` — each in four "
            "shapes: an answer, retrieval without generation, a batch, and a "
            "Server-Sent Events stream. Graphs are addressed by name under "
            "`/v1/graphs/{graph_id}`; the flat `/v1/search/...` paths address the "
            "default graph and are kept for older clients.\n\n"
            "Every response reports what actually ran (`engines`) and what it cost "
            "(`usage`). Errors share one envelope and carry `request_id`."
        ),
        version="0.2.0",
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
    )

    _install_middleware(app, settings)

    @app.exception_handler(RaguServiceError)
    async def _service_error_handler(_: Request, exc: RaguServiceError) -> JSONResponse:
        detail = getattr(exc, "detail", None)
        if detail:
            logger.error("{} ({}): {}", exc.code, exc.mode, detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_envelope(),
            headers=exc.headers or None,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # The contract specifies 400 for an invalid request, not FastAPI's default 422.
        message = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'][1:])}: {err['msg']}"
            for err in exc.errors()
        )
        error = InvalidRequestError(message or "Invalid request")
        return JSONResponse(status_code=error.status_code, content=error.to_envelope())

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.opt(exception=True).error("Unhandled service error")
        error = RaguServiceError(UNHANDLED_ERROR_MESSAGE)
        return JSONResponse(status_code=error.status_code, content=error.to_envelope())

    app.include_router(router)
    return app


def _install_middleware(app: FastAPI, settings: ServiceSettings) -> None:
    """
    Wrap the application in the concerns that apply to every route.

    Order matters: Starlette runs the last one added first, so the request id
    is bound before anything else can fail and want to report it.
    """
    if settings.request_timeout is not None:

        @app.middleware("http")
        async def _timeout(request: Request, call_next):
            # A request that outlives this is not going to succeed, and it holds
            # an LLM budget open while it waits.
            try:
                return await asyncio.wait_for(
                    call_next(request), settings.request_timeout
                )
            except asyncio.TimeoutError:
                error = RequestTimeoutError(
                    f"The request exceeded {settings.request_timeout}s and was "
                    "abandoned."
                )
                return JSONResponse(
                    status_code=error.status_code,
                    content=error.to_envelope(),
                    headers=error.headers or None,
                )

    app.add_middleware(MetricsMiddleware)
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(AuthMiddleware, settings=settings)

    origins = settings.cors_origins_list()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[REQUEST_ID_HEADER],
        )

    # Added last, so it runs first and every other layer can report the id.
    app.add_middleware(RequestContextMiddleware)
