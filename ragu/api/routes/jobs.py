"""
Long-running operations: ingesting documents and reindexing, as jobs to poll.
"""

from fastapi import APIRouter, Depends, Header, Request, Response

from ragu.api.backends.base import SearchBackend
from ragu.api.errors import InvalidRequestError, JobNotFoundError, ServiceNotReadyError
from ragu.api.models import (
    BuildRequest,
    ErrorResponse,
    JobListResponse,
    JobResponse,
)
from ragu.api.routes.deps import configured_backend, get_backend
from ragu.api.runtime.jobs import Job, JobManager

router = APIRouter()


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
    tags=["jobs"],
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
    backend = configured_backend(request, graph_id)
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


@router.get("/v1/jobs", response_model=JobListResponse, tags=["jobs"])
async def list_jobs(request: Request, graph_id: str | None = None) -> JobListResponse:
    """
    Every job this process knows about, newest first.
    """
    jobs = await _jobs(request).store.list(graph_id)
    return JobListResponse(jobs=[_job_response(job) for job in jobs])


@router.get(
    "/v1/jobs/{job_id}",
    response_model=JobResponse,
    responses=JOB_RESPONSES,
    tags=["jobs"],
)
async def get_job(request: Request, job_id: str) -> JobResponse:
    job = await _jobs(request).store.get(job_id)
    if job is None:
        raise JobNotFoundError(f"No job with id '{job_id}'.")
    return _job_response(job)


@router.delete(
    "/v1/jobs/{job_id}",
    response_model=JobResponse,
    responses=JOB_RESPONSES,
    tags=["jobs"],
)
async def cancel_job(request: Request, job_id: str) -> JobResponse:
    """
    Ask a running job to stop. A finished job is returned unchanged.
    """
    job = await _jobs(request).cancel(job_id)
    if job is None:
        raise JobNotFoundError(f"No job with id '{job_id}'.")
    return _job_response(job)


@router.post(
    "/v1/graphs/{graph_id}/reindex/{kind}",
    response_model=JobResponse,
    status_code=202,
    responses=JOB_RESPONSES,
    tags=["jobs"],
)
async def reindex_graph(
    request: Request,
    graph_id: str,
    kind: str,
    response: Response,
    backend: SearchBackend = Depends(get_backend),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JobResponse:
    """
    Rebuild communities, descriptions or the whole graph, as a job.

    Reindexing writes into the stores searches read, so it takes the same write
    lock ingestion does and the graph answers 409 GRAPH_BUSY while it runs.
    """
    job = await _jobs(request).submit(
        "reindex",
        graph_id,
        lambda: backend.reindex(kind),
        idempotency_key=idempotency_key,
    )
    response.headers["Location"] = f"/v1/jobs/{job.id}"
    return _job_response(job)
