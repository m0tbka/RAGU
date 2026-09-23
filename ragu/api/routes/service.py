"""
The service's own endpoints: health probes, Prometheus metrics and the
extraction ontology.
"""

from fastapi import APIRouter, Request, Response

from ragu.api.errors import RETRY_AFTER_SECONDS
from ragu.api.models import (
    HealthResponse,
    OntologyResponse,
)
from ragu.api.routes.deps import registry_of
from ragu.api.runtime.metrics import GRAPHS, JOBS, metrics

router = APIRouter()


def _health_of(request: Request) -> HealthResponse:
    """
    Describe the readiness of the service as a whole.

    :param request: Incoming request, for the application state.
    :return: Health payload; ready when at least one graph can answer.
    """
    registry = registry_of(request)
    loaded = bool(registry and registry.any_loaded)
    stats = None
    error = getattr(request.app.state, "startup_error", None)
    if registry is not None:
        default = registry.backend(registry.default_id)
        stats = default.stats if default is not None else None
        error = error or registry.error(registry.default_id)
    return HealthResponse(
        status="ok" if loaded else "degraded",
        graph_loaded=loaded,
        stats=stats.to_response() if stats is not None else None,
        error=error,
    )


@router.get("/health", response_model=HealthResponse, tags=["service"])
async def health(request: Request) -> HealthResponse:
    """
    Report readiness with a 200 regardless, for probes that read the body.
    """
    return _health_of(request)


@router.get("/health/live", response_model=HealthResponse, tags=["service"])
async def health_live(request: Request) -> HealthResponse:
    """
    Liveness: the process is up and serving. Always 200 while it can answer.
    """
    return _health_of(request)


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    tags=["service"],
    responses={503: {"model": HealthResponse, "description": "Graph not loaded"}},
)
async def health_ready(request: Request, response: Response) -> HealthResponse:
    """
    Readiness: 200 only when searches can actually be served, 503 otherwise.

    Loading a large graph takes minutes, so an orchestrator needs this split
    from liveness to avoid restarting a service that is still starting up.
    """
    payload = _health_of(request)
    if not payload.graph_loaded:
        response.status_code = 503
        response.headers["Retry-After"] = str(RETRY_AFTER_SECONDS)
    return payload


@router.get("/metrics", include_in_schema=False, tags=["service"])
async def prometheus_metrics(request: Request) -> Response:
    """
    Everything this process has counted, in Prometheus text format.
    """
    registry = registry_of(request)
    if registry is not None:
        loaded = sum(
            1
            for graph_id in registry.ids
            if (backend := registry.backend(graph_id)) is not None
            and backend.graph_loaded
        )
        metrics.set(GRAPHS, loaded, (("state", "loaded"),))
        metrics.set(GRAPHS, len(registry.ids) - loaded, (("state", "unavailable"),))

    manager = getattr(request.app.state, "jobs", None)
    if manager is not None:
        counts: dict[str, int] = {}
        for job in await manager.store.list():
            counts[job.state] = counts.get(job.state, 0) + 1
        for state in ("queued", "running", "succeeded", "failed", "cancelled"):
            metrics.set(JOBS, counts.get(state, 0), (("state", state),))

    return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")


@router.get("/v1/ontology", response_model=OntologyResponse, tags=["service"])
async def ontology() -> OntologyResponse:
    """
    The NEREL entity and relation types the extractors work with.
    """
    from ragu.triplet import types as ontology_types

    return OntologyResponse(
        entity_types=list(getattr(ontology_types, "NEREL_ENTITY_TYPES", [])),
        relation_types=list(getattr(ontology_types, "NEREL_RELATION_TYPES", [])),
    )
