"""
HTTP routes: one physical resource per search mode.

Separate paths let a gateway apply per-mode timeouts and rate limits without
any change in the service (see the spec, "Per-route policy").
"""

import json
from typing import Any

from fastapi import APIRouter, Depends, Header, Request, Response
from fastapi.responses import StreamingResponse

from ragu.api.backends.base import (
    MODE_REQUIREMENTS,
    RetrieveOutcome,
    SearchStreamEvent,
    SearchBackend,
    SearchCall,
    SearchOutcome,
)
from ragu.api.errors import (
    RETRY_AFTER_SECONDS,
    GraphNotFoundError,
    InvalidRequestError,
    JobNotFoundError,
    ServiceNotReadyError,
)
from ragu.api.jobs import Job, JobManager
from ragu.api.metrics import GRAPHS, JOBS, SEARCHES, metrics
from ragu.api.middleware import Admission
from ragu.api import usage
from ragu.api.models import (
    BatchSearchItem,
    BatchSearchResponse,
    BuildRequest,
    EngineReport,
    ErrorBody,
    ErrorResponse,
    GraphInfo,
    GraphListResponse,
    JobListResponse,
    JobResponse,
    GlobalBatchRequest,
    GlobalRetrieveRequest,
    GlobalSearchRequest,
    HealthResponse,
    LocalBatchRequest,
    LocalRetrieveRequest,
    LocalSearchRequest,
    MixBatchRequest,
    MixRetrieveRequest,
    MixSearchRequest,
    ModeAvailability,
    NaiveBatchRequest,
    NaiveRetrieveRequest,
    NaiveSearchRequest,
    RetrieveResponse,
    SearchMode,
    SearchResponse,
    StageUsageModel,
    UsageModel,
)

async def generation_slot(request: Request):
    """
    Open cost accounting and hold a generation slot for one request.

    Attached to the whole search router rather than to each of its sixteen
    routes, so a route added later cannot forget it.

    Note that a streamed response is produced after the handler returns, so its
    slot is released before the last token is sent; admission bounds the
    retrieval and the first generation, not the tail of a stream.
    """
    _begin_usage(request)
    async with _admission(request).slot():
        yield


router = APIRouter()

# Mounted under both /v1/graphs/{graph_id} and /v1: the graph is read from the
# path when it is there and falls back to the default when it is not, so the
# flat paths that predate the catalogue keep working.
search_router = APIRouter(dependencies=[Depends(generation_slot)])


SEARCH_RESPONSES = {
    409: {
        "model": ErrorResponse,
        "description": "Capability unavailable for this graph",
    },
    503: {
        "model": ErrorResponse,
        "description": "Graph is not loaded / service not ready",
    },
    500: {"model": ErrorResponse, "description": "Internal error"},
}


def _response(call: SearchCall, outcome: SearchOutcome) -> SearchResponse:
    """
    Render one search outcome as the wire response.

    :param call: The resolved request.
    :param outcome: Normalized backend outcome.
    :return: The response body.
    """
    report = outcome.engines
    metrics.increment(
        SEARCHES,
        (
            ("mode", call.mode),
            ("outcome", "degraded" if report and report.degraded else "ok"),
        ),
    )
    return SearchResponse(
        query=call.query,
        mode=call.mode,
        used_query_plan=call.use_query_plan,
        answer=outcome.answer,
        sources=outcome.sources,
        subqueries=outcome.subqueries,
        usage=_usage_model(),
        engines=outcome.engines
        or EngineReport(
            requested=call.mode, used="unknown", query_plan=call.use_query_plan
        ),
    )


def _admission(request: Request) -> Admission:
    """
    The service's generation ceiling.
    """
    admission = getattr(request.app.state, "admission", None)
    return admission if admission is not None else Admission(None)


def _begin_usage(request: Request) -> None:
    """
    Start cost accounting for this request, with the service's budget.
    """
    settings = getattr(request.app.state, "settings", None)
    usage.start(
        max_calls=getattr(settings, "max_llm_calls_per_request", None),
        max_tokens=getattr(settings, "max_tokens_per_request", None),
    )


def _usage_model() -> UsageModel | None:
    """
    Render what this request has cost so far.
    """
    record = usage.current()
    if record is None:
        return None
    return UsageModel(
        calls=record.calls,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        total_tokens=record.total_tokens,
        stages={
            name: StageUsageModel(
                calls=stage.calls,
                prompt_tokens=stage.prompt_tokens,
                completion_tokens=stage.completion_tokens,
            )
            for name, stage in record.stages.items()
        },
    )


def _registry(request: Request):
    """
    The graph catalogue, or ``None`` before the lifespan has run.
    """
    return getattr(request.app.state, "registry", None)


def _health_of(request: Request) -> HealthResponse:
    """
    Describe the readiness of the service as a whole.

    :param request: Incoming request, for the application state.
    :return: Health payload; ready when at least one graph can answer.
    """
    registry = _registry(request)
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


def get_backend(request: Request) -> SearchBackend:
    """
    Resolve the graph this request addresses.

    :raises GraphNotFoundError: If the path names a graph that is not configured.
    :raises ServiceNotReadyError: If that graph is not loaded.
    """
    registry = _registry(request)
    if registry is None:
        raise ServiceNotReadyError("The service is still starting up.")
    return registry.resolve(request.path_params.get("graph_id"))


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
    registry = _registry(request)
    backend = registry.backend(graph_id) if registry is not None else None
    if backend is None:
        known = ", ".join(registry.ids) if registry is not None else "none"
        raise GraphNotFoundError(
            f"No graph named '{graph_id}'. Configured graphs: {known or 'none'}."
        )
    stats = backend.stats
    return GraphInfo(
        id=graph_id,
        loaded=backend.graph_loaded,
        language=backend.language,
        stats=stats.to_response() if stats is not None else None,
        modes=_mode_availability(backend),
        error=registry.error(graph_id) if registry is not None else None,
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """
    Report readiness with a 200 regardless, for probes that read the body.
    """
    return _health_of(request)


@router.get("/health/live", response_model=HealthResponse)
async def health_live(request: Request) -> HealthResponse:
    """
    Liveness: the process is up and serving. Always 200 while it can answer.
    """
    return _health_of(request)


@router.get(
    "/health/ready",
    response_model=HealthResponse,
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


def _call(mode: SearchMode, payload: Any) -> SearchCall:
    """
    Build a backend call from any of the search or retrieve request models.

    :param mode: Mode the route serves.
    :param payload: The validated request body.
    :return: The resolved call.
    """
    return SearchCall(
        mode=mode,
        queries=(payload.query,),
        params=payload.params,
        local_params=getattr(payload, "local_params", None),
        naive_params=getattr(payload, "naive_params", None),
        use_query_plan=getattr(payload, "use_query_plan", False),
        language=getattr(payload, "language", None),
        rerank=getattr(payload, "rerank", True),
    )


async def _one(backend: SearchBackend, awaitable: Any, mode: SearchMode) -> Any:
    """
    Take the single outcome of a one-query call, refusing an empty one.

    :raises CapabilityUnavailableError: If the query retrieved nothing.
    """
    outcome = (await awaitable)[0]
    backend.require_evidence(mode, outcome)
    return outcome


def _retrieved(call: SearchCall, outcome: RetrieveOutcome) -> RetrieveResponse:
    return RetrieveResponse(
        query=call.query,
        mode=call.mode,
        sources=outcome.sources,
        usage=_usage_model(),
        engines=outcome.engines or EngineReport(requested=call.mode, used="unknown"),
    )




GRAPH_RESPONSES = {404: {"model": ErrorResponse, "description": "No such graph"}}


@router.get("/v1/graphs", response_model=GraphListResponse)
async def list_graphs(request: Request) -> GraphListResponse:
    """
    Every graph this service serves, with its sizes and available modes.
    """
    registry = _registry(request)
    if registry is None:
        return GraphListResponse(default="", graphs=[])
    return GraphListResponse(
        default=registry.default_id,
        graphs=[_graph_info(request, graph_id) for graph_id in registry.ids],
    )


@router.get(
    "/v1/graphs/{graph_id}", response_model=GraphInfo, responses=GRAPH_RESPONSES
)
async def get_graph(request: Request, graph_id: str) -> GraphInfo:
    return _graph_info(request, graph_id)


@router.get(
    "/v1/graphs/{graph_id}/capabilities",
    response_model=list[ModeAvailability],
    responses=GRAPH_RESPONSES,
)
async def graph_capabilities(request: Request, graph_id: str) -> list[ModeAvailability]:
    """
    Which search modes this graph can serve, and why the others cannot.
    """
    return _graph_info(request, graph_id).modes


@search_router.post(
    "/search/global", response_model=SearchResponse, responses=SEARCH_RESPONSES
)
async def search_global(
    payload: GlobalSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> SearchResponse:
    call = _call("global", payload)
    return _response(call, await _one(backend, backend.search(call), call.mode))


@search_router.post(
    "/search/local", response_model=SearchResponse, responses=SEARCH_RESPONSES
)
async def search_local(
    payload: LocalSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> SearchResponse:
    call = _call("local", payload)
    return _response(call, await _one(backend, backend.search(call), call.mode))


@search_router.post(
    "/search/naive", response_model=SearchResponse, responses=SEARCH_RESPONSES
)
async def search_naive(
    payload: NaiveSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> SearchResponse:
    call = _call("naive", payload)
    return _response(call, await _one(backend, backend.search(call), call.mode))


@search_router.post(
    "/search/mix", response_model=SearchResponse, responses=SEARCH_RESPONSES
)
async def search_mix(
    payload: MixSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> SearchResponse:
    """
    Ensemble the local and naive engines over one query.

    Child failures are tolerated by the ensemble, so ``engines`` in the response
    reports which children actually contributed.
    """
    call = _call("mix", payload)
    return _response(call, await _one(backend, backend.search(call), call.mode))


@search_router.post(
    "/search/global/retrieve",
    response_model=RetrieveResponse,
    responses=SEARCH_RESPONSES,
)
async def retrieve_global(
    payload: GlobalRetrieveRequest,
    backend: SearchBackend = Depends(get_backend),
) -> RetrieveResponse:
    call = _call("global", payload)
    return _retrieved(call, await _one(backend, backend.retrieve(call), call.mode))


@search_router.post(
    "/search/local/retrieve",
    response_model=RetrieveResponse,
    responses=SEARCH_RESPONSES,
)
async def retrieve_local(
    payload: LocalRetrieveRequest,
    backend: SearchBackend = Depends(get_backend),
) -> RetrieveResponse:
    call = _call("local", payload)
    return _retrieved(call, await _one(backend, backend.retrieve(call), call.mode))


@search_router.post(
    "/search/naive/retrieve",
    response_model=RetrieveResponse,
    responses=SEARCH_RESPONSES,
)
async def retrieve_naive(
    payload: NaiveRetrieveRequest,
    backend: SearchBackend = Depends(get_backend),
) -> RetrieveResponse:
    call = _call("naive", payload)
    return _retrieved(call, await _one(backend, backend.retrieve(call), call.mode))


@search_router.post(
    "/search/mix/retrieve",
    response_model=RetrieveResponse,
    responses=SEARCH_RESPONSES,
)
async def retrieve_mix(
    payload: MixRetrieveRequest,
    backend: SearchBackend = Depends(get_backend),
) -> RetrieveResponse:
    """
    Gather context from both child engines without generating an answer.
    """
    call = _call("mix", payload)
    return _retrieved(call, await _one(backend, backend.retrieve(call), call.mode))


def _batch_call(mode: SearchMode, payload: Any) -> SearchCall:
    return SearchCall(
        mode=mode,
        queries=tuple(payload.queries),
        params=payload.params,
        local_params=getattr(payload, "local_params", None),
        naive_params=getattr(payload, "naive_params", None),
        use_query_plan=getattr(payload, "use_query_plan", False),
        language=getattr(payload, "language", None),
        rerank=getattr(payload, "rerank", True),
    )


async def _batched(
    mode: SearchMode, payload: Any, backend: SearchBackend
) -> BatchSearchResponse:
    """
    Answer a batch, reporting per-query emptiness instead of failing the batch.
    """
    call = _batch_call(mode, payload)
    backend.require_batch_size(len(call.queries))
    outcomes = await backend.search(call)

    results = []
    for query, outcome in zip(call.queries, outcomes):
        if outcome.sources:
            results.append(
                BatchSearchItem(
                    query=query,
                    answer=outcome.answer,
                    sources=outcome.sources,
                    subqueries=outcome.subqueries,
                )
            )
        else:
            empty = backend.no_evidence(mode)
            results.append(
                BatchSearchItem(
                    query=query, error=ErrorBody(**empty.to_envelope()["error"])
                )
            )

    engines = outcomes[0].engines if outcomes else None
    return BatchSearchResponse(
        mode=mode,
        usage=_usage_model(),
        used_query_plan=call.use_query_plan,
        engines=engines or EngineReport(requested=mode, used="unknown"),
        results=results,
    )


@search_router.post(
    "/search/global/batch",
    response_model=BatchSearchResponse,
    responses=SEARCH_RESPONSES,
)
async def batch_global(
    payload: GlobalBatchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> BatchSearchResponse:
    return await _batched("global", payload, backend)


@search_router.post(
    "/search/local/batch",
    response_model=BatchSearchResponse,
    responses=SEARCH_RESPONSES,
)
async def batch_local(
    payload: LocalBatchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> BatchSearchResponse:
    return await _batched("local", payload, backend)


@search_router.post(
    "/search/naive/batch",
    response_model=BatchSearchResponse,
    responses=SEARCH_RESPONSES,
)
async def batch_naive(
    payload: NaiveBatchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> BatchSearchResponse:
    """
    Answer many queries in one pass.

    This is where the engines earn their keep: retrieval is shared across the
    whole list and, with a query plan, independent subqueries from different
    top-level queries are answered in the same child batch.
    """
    return await _batched("naive", payload, backend)


@search_router.post(
    "/search/mix/batch",
    response_model=BatchSearchResponse,
    responses=SEARCH_RESPONSES,
)
async def batch_mix(
    payload: MixBatchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> BatchSearchResponse:
    return await _batched("mix", payload, backend)


def _sse(event: SearchStreamEvent) -> str:
    """
    Frame one event as a Server-Sent Event.
    """
    payload = json.dumps(event.data, ensure_ascii=False)
    return f"event: {event.event}\ndata: {payload}\n\n"


async def _stream(mode: SearchMode, payload: Any, backend: SearchBackend) -> Response:
    """
    Stream one answer, refusing an unservable mode before the response starts.
    """
    call = _call(mode, payload)
    # Inside an open stream a 409 could only be an SSE event, so the capability
    # is checked while a status code can still carry it.
    backend.require_capability(mode)

    async def events():
        async for event in backend.stream(call):
            yield _sse(event)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@search_router.post("/search/global/stream", responses=SEARCH_RESPONSES)
async def stream_global(
    payload: GlobalSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> Response:
    return await _stream("global", payload, backend)


@search_router.post("/search/local/stream", responses=SEARCH_RESPONSES)
async def stream_local(
    payload: LocalSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> Response:
    return await _stream("local", payload, backend)


@search_router.post("/search/naive/stream", responses=SEARCH_RESPONSES)
async def stream_naive(
    payload: NaiveSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> Response:
    """
    Stream the answer as Server-Sent Events.

    One ``meta`` event carries the retrieval and the engine report, then
    ``delta`` events carry the text, then ``done`` closes with the final engine
    report. A failure after the headers are sent arrives as an ``error`` event.
    """
    return await _stream("naive", payload, backend)


@search_router.post("/search/mix/stream", responses=SEARCH_RESPONSES)
async def stream_mix(
    payload: MixSearchRequest,
    backend: SearchBackend = Depends(get_backend),
) -> Response:
    return await _stream("mix", payload, backend)


JOB_RESPONSES = {
    404: {"model": ErrorResponse, "description": "No such job or graph"},
    400: {"model": ErrorResponse, "description": "Graph does not accept documents"},
}


def _job_response(job: Job) -> JobResponse:
    return JobResponse(
        id=job.id,
        kind=job.kind,
        graph_id=job.graph_id,
        state=job.state,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        result=job.result,
    )


def _jobs(request: Request) -> JobManager:
    """
    The job manager, or a 503 before the lifespan has run.
    """
    manager = getattr(request.app.state, "jobs", None)
    if manager is None:
        raise ServiceNotReadyError("The service is still starting up.")
    return manager


@router.post(
    "/v1/graphs/{graph_id}/documents",
    response_model=JobResponse,
    status_code=202,
    responses=JOB_RESPONSES,
)
async def add_documents(
    request: Request,
    graph_id: str,
    payload: BuildRequest,
    response: Response,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JobResponse:
    """
    Add documents to a graph, as a job.

    Building takes minutes to hours, which no request can hold open, so this
    answers 202 with a job to poll. Searches on this graph answer 409 while it
    runs: the build writes into the stores they read.

    Send ``Idempotency-Key`` so a retry does not build the same corpus twice.
    """
    registry = _registry(request)
    backend = registry.backend(graph_id) if registry is not None else None
    if backend is None:
        known = ", ".join(registry.ids) if registry is not None else "none"
        raise GraphNotFoundError(
            f"No graph named '{graph_id}'. Configured graphs: {known or 'none'}."
        )
    if not backend.accepts_documents:
        raise InvalidRequestError(
            f"Graph '{graph_id}' does not accept documents. Enable ingestion in its "
            "configuration to build it over HTTP."
        )

    documents = list(payload.documents)
    job = await _jobs(request).submit(
        "build",
        graph_id,
        lambda: backend.build(documents),
        idempotency_key=idempotency_key,
    )
    response.headers["Location"] = f"/v1/jobs/{job.id}"
    return _job_response(job)


@router.get("/v1/jobs", response_model=JobListResponse)
async def list_jobs(request: Request, graph_id: str | None = None) -> JobListResponse:
    """
    Every job this process knows about, newest first.
    """
    jobs = await _jobs(request).store.list(graph_id)
    return JobListResponse(jobs=[_job_response(job) for job in jobs])


@router.get("/v1/jobs/{job_id}", response_model=JobResponse, responses=JOB_RESPONSES)
async def get_job(request: Request, job_id: str) -> JobResponse:
    job = await _jobs(request).store.get(job_id)
    if job is None:
        raise JobNotFoundError(f"No job with id '{job_id}'.")
    return _job_response(job)


@router.delete("/v1/jobs/{job_id}", response_model=JobResponse, responses=JOB_RESPONSES)
async def cancel_job(request: Request, job_id: str) -> JobResponse:
    """
    Ask a running job to stop. A finished job is returned unchanged.
    """
    job = await _jobs(request).cancel(job_id)
    if job is None:
        raise JobNotFoundError(f"No job with id '{job_id}'.")
    return _job_response(job)


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics(request: Request) -> Response:
    """
    Everything this process has counted, in Prometheus text format.
    """
    registry = _registry(request)
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


# The catalogue path is canonical; the flat one is kept for clients written
# before the service served more than one graph.
router.include_router(search_router, prefix="/v1/graphs/{graph_id}", tags=["search"])
router.include_router(search_router, prefix="/v1", tags=["search"], deprecated=True)
