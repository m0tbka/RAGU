"""
The graph catalogue: which graphs this service serves, what each can answer,
and how it is doing.
"""

from fastapi import APIRouter, Depends, Request

from ragu.api.backends.base import SearchBackend
from ragu.api.backends.capabilities import MODE_REQUIREMENTS
from ragu.api.models import (
    ConsistencyIssueItem,
    ConsistencyReportModel,
    GraphDetail,
    GraphInfo,
    GraphListResponse,
    ModeAvailability,
)
from ragu.api.routes.deps import (
    GRAPH_RESPONSES,
    configured_backend,
    get_backend,
    registry_of,
)

router = APIRouter()


def _mode_availability(backend: SearchBackend) -> list[ModeAvailability]:
    """
    Render which modes a graph can serve, and what the others would need.
    """
    return [
        ModeAvailability(
            mode=mode,
            available=missing is None,
            missing_capability=missing,
            reason=None if missing is None else MODE_REQUIREMENTS[mode].missing_message,
        )
        for mode, missing in backend.capabilities().items()
    ]


def _graph_info(request: Request, graph_id: str) -> GraphInfo:
    """
    Describe one graph, loaded or not.

    :raises GraphNotFoundError: If it is not configured.
    """
    registry = registry_of(request)
    backend = configured_backend(request, graph_id)
    stats = backend.stats
    return GraphInfo(
        id=graph_id,
        loaded=backend.graph_loaded,
        language=backend.language,
        stats=stats.to_response() if stats is not None else None,
        modes=_mode_availability(backend),
        error=registry.error(graph_id) if registry is not None else None,
    )


@router.get("/v1/graphs", response_model=GraphListResponse, tags=["graphs"])
async def list_graphs(request: Request) -> GraphListResponse:
    """
    Every graph this service serves, with its sizes and available modes.
    """
    registry = registry_of(request)
    if registry is None:
        return GraphListResponse(default="", graphs=[])
    return GraphListResponse(
        default=registry.default_id,
        graphs=[_graph_info(request, graph_id) for graph_id in registry.ids],
    )


@router.get(
    "/v1/graphs/{graph_id}",
    response_model=GraphInfo,
    responses=GRAPH_RESPONSES,
    tags=["graphs"],
)
async def get_graph(request: Request, graph_id: str) -> GraphInfo:
    return _graph_info(request, graph_id)


@router.get(
    "/v1/graphs/{graph_id}/capabilities",
    response_model=list[ModeAvailability],
    responses=GRAPH_RESPONSES,
    tags=["graphs"],
)
async def graph_capabilities(request: Request, graph_id: str) -> list[ModeAvailability]:
    """
    Which search modes this graph can serve, and why the others cannot.
    """
    return _graph_info(request, graph_id).modes


@router.get(
    "/v1/graphs/{graph_id}/stats",
    response_model=GraphDetail,
    responses=GRAPH_RESPONSES,
    tags=["graphs"],
)
async def graph_stats(request: Request, graph_id: str) -> GraphDetail:
    """
    Sizes, embedding dimension and which modes this corpus can serve.
    """
    info = _graph_info(request, graph_id)
    backend = registry_of(request).backend(graph_id)
    detail = await backend.graph_detail() if backend.graph_loaded else {}
    return GraphDetail(
        id=graph_id,
        loaded=info.loaded,
        language=info.language,
        accepts_documents=backend.accepts_documents,
        modes=info.modes,
        **detail,
    )


@router.get(
    "/v1/graphs/{graph_id}/consistency",
    response_model=ConsistencyReportModel,
    responses=GRAPH_RESPONSES,
    tags=["graphs"],
)
async def graph_consistency(
    graph_id: str,
    backend: SearchBackend = Depends(get_backend),
) -> ConsistencyReportModel:
    """
    Audit the graph's cross-storage invariants.
    """
    report = await backend.consistency()
    if report is None:
        return ConsistencyReportModel(consistent=True)
    return ConsistencyReportModel(
        consistent=report.is_consistent,
        issues=[
            ConsistencyIssueItem(
                check=issue.check, message=issue.message, details=issue.details
            )
            for issue in report.errors
        ],
    )
