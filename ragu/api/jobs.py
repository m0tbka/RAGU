"""
Long-running work, tracked so HTTP does not have to wait for it.

Building a graph takes minutes to hours, which no request can hold open. Work is
submitted, given an id, and polled.

The store is an interface with one in-memory implementation. Jobs therefore live
as long as the process: a single replica, and a restart loses what was running.
Swapping in a shared store is the way to lift that, and is deliberately not done
here.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal

from ragu.common.logger import logger

JobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]

# Work a job can do. Named rather than free-form so the listing stays filterable.
JobKind = Literal["build", "reindex"]

FINISHED: frozenset[str] = frozenset({"succeeded", "failed", "cancelled"})


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Job:
    """
    One unit of long-running work and what became of it.
    """

    id: str
    kind: JobKind
    graph_id: str
    state: JobState = "queued"
    created_at: datetime = field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    detail: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None

    @property
    def finished(self) -> bool:
        return self.state in FINISHED


class JobStore(ABC):
    """
    Where jobs are kept. One implementation today; the seam is for the next one.
    """

    @abstractmethod
    async def put(self, job: Job) -> None: ...

    @abstractmethod
    async def get(self, job_id: str) -> Job | None: ...

    @abstractmethod
    async def list(self, graph_id: str | None = None) -> list[Job]: ...

    @abstractmethod
    async def find_by_key(self, key: str) -> Job | None: ...

    @abstractmethod
    async def remember_key(self, key: str, job_id: str) -> None: ...


class InMemoryJobStore(JobStore):
    """
    Jobs in a dict. Lost on restart, invisible to other replicas.
    """

    def __init__(self, limit: int = 1000):
        """
        :param limit: How many finished jobs to keep before dropping the oldest.
        """
        self._jobs: dict[str, Job] = {}
        self._keys: dict[str, str] = {}
        self._limit = limit

    async def put(self, job: Job) -> None:
        self._jobs[job.id] = job
        self._evict()

    async def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def list(self, graph_id: str | None = None) -> list[Job]:
        jobs = list(self._jobs.values())
        if graph_id is not None:
            jobs = [job for job in jobs if job.graph_id == graph_id]
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    async def find_by_key(self, key: str) -> Job | None:
        job_id = self._keys.get(key)
        return self._jobs.get(job_id) if job_id else None

    async def remember_key(self, key: str, job_id: str) -> None:
        self._keys[key] = job_id

    def _evict(self) -> None:
        finished = [job for job in self._jobs.values() if job.finished]
        while len(self._jobs) > self._limit and finished:
            oldest = min(finished, key=lambda job: job.created_at)
            finished.remove(oldest)
            self._jobs.pop(oldest.id, None)


class JobManager:
    """
    Runs jobs as background tasks and records what happens to them.
    """

    def __init__(self, store: JobStore | None = None):
        self.store = store or InMemoryJobStore()
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def submit(
        self,
        kind: JobKind,
        graph_id: str,
        work: Callable[[], Awaitable[dict[str, Any] | None]],
        idempotency_key: str | None = None,
    ) -> Job:
        """
        Start a job and return it immediately.

        :param kind: What the job does.
        :param graph_id: Graph it operates on.
        :param work: The coroutine factory to run.
        :param idempotency_key: When given, a repeat of the same key returns the
            original job instead of starting a second one — a client retry must
            not build the same corpus twice.
        :return: The job, already queued or running.
        """
        if idempotency_key:
            existing = await self.store.find_by_key(idempotency_key)
            if existing is not None:
                return existing

        job = Job(id=uuid.uuid4().hex, kind=kind, graph_id=graph_id)
        await self.store.put(job)
        if idempotency_key:
            await self.store.remember_key(idempotency_key, job.id)

        self._tasks[job.id] = asyncio.create_task(self._run(job, work))
        return job

    async def _run(
        self,
        job: Job,
        work: Callable[[], Awaitable[dict[str, Any] | None]],
    ) -> None:
        await self._update(job.id, state="running", started_at=_now())
        try:
            result = await work()
        except asyncio.CancelledError:
            await self._update(job.id, state="cancelled", finished_at=_now())
            raise
        except Exception as exc:
            logger.opt(exception=True).error(
                "Job {} ({} on '{}') failed: {}", job.id, job.kind, job.graph_id, exc
            )
            await self._update(
                job.id, state="failed", finished_at=_now(), error=str(exc)
            )
        else:
            await self._update(
                job.id, state="succeeded", finished_at=_now(), result=result
            )
        finally:
            self._tasks.pop(job.id, None)

    async def _update(self, job_id: str, **changes: Any) -> None:
        job = await self.store.get(job_id)
        if job is not None:
            await self.store.put(replace(job, **changes))

    async def cancel(self, job_id: str) -> Job | None:
        """
        Ask a running job to stop.

        :param job_id: The job to cancel.
        :return: The job, or ``None`` if there is no such job.
        """
        job = await self.store.get(job_id)
        if job is None or job.finished:
            return job
        task = self._tasks.get(job_id)
        if task is not None:
            task.cancel()
        else:
            await self._update(job_id, state="cancelled", finished_at=_now())
        return await self.store.get(job_id)

    async def shutdown(self) -> None:
        """
        Cancel everything still running and wait for it to unwind.
        """
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
