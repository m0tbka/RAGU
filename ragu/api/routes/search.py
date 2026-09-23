"""
Search routes: one physical resource per search mode.

Separate paths let a gateway apply per-mode timeouts and rate limits without
any change in the service (see the spec, "Per-route policy"). Each mode has
four shapes: an answer, retrieval alone, a batch, and a stream.
"""

import json
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from ragu.api.backends.base import (
    RetrieveOutcome,
    SearchBackend,
    SearchCall,
    SearchOutcome,
    SearchStreamEvent,
)
from ragu.api.models import (
    BatchSearchItem,
    BatchSearchResponse,
    ChildEngineReport,
    DEFAULT_MIX_ENGINES,
    EngineReport,
    ErrorBody,
    ErrorResponse,
    GlobalBatchRequest,
    GlobalRetrieveRequest,
    GlobalSearchRequest,
    LocalBatchRequest,
    LocalRetrieveRequest,
    LocalSearchRequest,
    MixBatchRequest,
    MixRetrieveRequest,
    MixSearchRequest,
    NaiveBatchRequest,
    NaiveRetrieveRequest,
    NaiveSearchRequest,
    RetrieveResponse,
    SearchMode,
    SearchResponse,
    StageUsageModel,
    UsageModel,
)
from ragu.api.routes.deps import get_backend
from ragu.api.runtime.metrics import SEARCHES, metrics
from ragu.api.runtime.middleware import Admission
from ragu.api.search import usage


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


# Mounted under both /v1/graphs/{graph_id} and /v1: the graph is read from the
# path when it is there and falls back to the default when it is not, so the
# flat paths that predate the catalogue keep working.
search_router = APIRouter(dependencies=[Depends(generation_slot)])


SEARCH_RESPONSES = {
    409: {
        "model": ErrorResponse,
        "description": "Capability unavailable for this graph",
    },
    429: {
        "model": ErrorResponse,
        "description": "BUDGET_EXCEEDED: the request cannot fit its LLM budget. "
        "TOO_MANY_REQUESTS: every generation slot is taken",
    },
    503: {
        "model": ErrorResponse,
        "description": "Graph is not loaded / service not ready",
    },
    500: {"model": ErrorResponse, "description": "Internal error"},
}


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
                retrieval_ms=(
                    round(stage.retrieval_ms, 1)
                    if stage.retrieval_ms is not None
                    else None
                ),
                generation_ms=round(stage.generation_ms, 1) or None,
                rerank_ms=round(stage.rerank_ms, 1) or None,
            )
            for name, stage in record.stages.items()
        },
    )


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
        global_params=getattr(payload, "global_params", None),
        mix_engines=tuple(getattr(payload, "engines", None) or DEFAULT_MIX_ENGINES),
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
        global_params=getattr(payload, "global_params", None),
        mix_engines=tuple(getattr(payload, "engines", None) or DEFAULT_MIX_ENGINES),
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

    return BatchSearchResponse(
        mode=mode,
        usage=_usage_model(),
        used_query_plan=call.use_query_plan,
        engines=_merged_report(mode, outcomes),
        results=results,
    )


def _merged_report(mode: SearchMode, outcomes: list[SearchOutcome]) -> EngineReport:
    """
    Fold every query's engine report into one for the batch.

    Reporting only the first query's would hide the case that matters: a batch
    where most queries ran clean and one lost a child engine or its reranker.
    Degradation is therefore the union over the batch, not a sample of it.

    :param mode: Mode the batch asked for.
    :param outcomes: One outcome per query, in request order.
    :return: The report for the batch as a whole.
    """
    reports = [outcome.engines for outcome in outcomes if outcome.engines]
    if not reports:
        return EngineReport(requested=mode, used="unknown")

    children: dict[tuple[str, str | None], ChildEngineReport] = {}
    for report in reports:
        for child in report.children:
            # A child that failed for any query is reported as failed for the
            # batch; the first failure carries the reason.
            key = (child.engine, child.mode)
            kept = children.get(key)
            if kept is None or (kept.ok and not child.ok):
                children[key] = child

    return EngineReport(
        requested=mode,
        used=reports[0].used,
        query_plan=any(report.query_plan for report in reports),
        degraded=any(report.degraded for report in reports),
        children=list(children.values()),
        reranked=any(report.reranked for report in reports),
        rerank_error=next(
            (report.rerank_error for report in reports if report.rerank_error), None
        ),
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
    Stream one answer, refusing an unservable mode or an unaffordable one before
    the response starts.
    """
    call = _call(mode, payload)
    # Inside an open stream a 409 could only be an SSE event, so the capability
    # is checked while a status code can still carry it.
    backend.require_capability(mode, call.mix_engines)
    # The budget too, for the same reason: refused inside the stream it could only
    # be an event after a 200, and refused here it costs nothing at all.
    await backend.require_budget(call)

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
