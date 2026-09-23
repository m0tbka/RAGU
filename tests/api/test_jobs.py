"""
Ingestion and reindexing, run as jobs.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "fastapi", reason="install the 'api' extra to test the search service"
)

from ragu.api.backends.base import SearchCall
from ragu.api.config import ServiceSettings
from tests.api.support import (
    build_client,
    ingest_client,
)


class TestIngestion:
    """Building is a job, because no request can hold a graph build open."""

    def test_documents_are_accepted_as_a_job(self):
        with ingest_client() as client:
            response = client.post(
                "/v1/graphs/corpus/documents", json={"documents": ["a", "b"]}
            )
        assert response.status_code == 202
        body = response.json()
        assert body["kind"] == "build"
        assert body["graph_id"] == "corpus"
        assert response.headers["Location"] == f"/v1/jobs/{body['id']}"

    def test_the_job_can_be_polled_to_completion(self):
        with ingest_client() as client:
            job_id = client.post(
                "/v1/graphs/corpus/documents", json={"documents": ["a", "b"]}
            ).json()["id"]
            for _ in range(50):
                job = client.get(f"/v1/jobs/{job_id}").json()
                if job["state"] in ("succeeded", "failed"):
                    break
        assert job["state"] == "succeeded"
        assert job["result"]["documents"] == 2
        assert job["finished_at"] is not None

    def test_a_graph_that_does_not_ingest_says_so(self):
        with ingest_client() as client:
            response = client.post(
                "/v1/graphs/readonly/documents", json={"documents": ["a"]}
            )
        assert response.status_code == 400
        assert "does not accept documents" in response.json()["error"]["message"]

    def test_an_unknown_graph_answers_404(self):
        with ingest_client() as client:
            response = client.post(
                "/v1/graphs/missing/documents", json={"documents": ["a"]}
            )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "GRAPH_NOT_FOUND"

    def test_an_empty_document_list_is_rejected(self):
        with ingest_client() as client:
            response = client.post("/v1/graphs/corpus/documents", json={"documents": []})
        assert response.status_code == 400

    def test_a_retry_with_the_same_key_does_not_build_twice(self):
        with ingest_client() as client:
            headers = {"Idempotency-Key": "upload-42"}
            first = client.post(
                "/v1/graphs/corpus/documents",
                json={"documents": ["a"]},
                headers=headers,
            ).json()
            second = client.post(
                "/v1/graphs/corpus/documents",
                json={"documents": ["a"]},
                headers=headers,
            ).json()
        assert first["id"] == second["id"]

    def test_jobs_are_listed_newest_first_and_filterable(self):
        with ingest_client() as client:
            client.post("/v1/graphs/corpus/documents", json={"documents": ["a"]})
            client.post("/v1/graphs/corpus/documents", json={"documents": ["b"]})
            everything = client.get("/v1/jobs").json()["jobs"]
            filtered = client.get("/v1/jobs", params={"graph_id": "corpus"}).json()["jobs"]
            none = client.get("/v1/jobs", params={"graph_id": "readonly"}).json()["jobs"]
        assert len(everything) == 2
        assert len(filtered) == 2
        assert none == []

    def test_an_unknown_job_answers_404(self):
        with ingest_client() as client:
            for call in (client.get, client.delete):
                response = call("/v1/jobs/nope")
                assert response.status_code == 404
                assert response.json()["error"]["code"] == "JOB_NOT_FOUND"


class TestBuildLock:
    """A build and a search must not interleave on one graph."""

    async def test_searching_a_graph_under_construction_is_refused(self):
        from ragu.api.backends.stub import StubBackend
        from ragu.api.config import GraphSpec
        from ragu.api.errors import GraphBusyError

        spec = GraphSpec(id="corpus", storage_folder="a", build={"enabled": True})
        backend = StubBackend(ServiceSettings(backend="stub"), spec)
        await backend.startup()

        backend._building = True
        with pytest.raises(GraphBusyError) as failure:
            await backend.search(SearchCall(mode="naive", queries=("q",)))

        assert failure.value.status_code == 409
        assert failure.value.headers["Retry-After"] == "30"
        assert "being built" in failure.value.message

    async def test_the_graph_is_searchable_again_afterwards(self):
        from ragu.api.backends.stub import StubBackend
        from ragu.api.config import GraphSpec

        spec = GraphSpec(id="corpus", storage_folder="a", build={"enabled": True})
        backend = StubBackend(ServiceSettings(backend="stub"), spec)
        await backend.startup()

        await backend.build(["a"])
        outcome, = await backend.search(SearchCall(mode="naive", queries=("q",)))

        assert outcome.sources
        assert backend._building is False


class TestJobManager:
    async def test_a_failing_job_keeps_its_reason(self):
        import asyncio

        from ragu.api.runtime.jobs import JobManager

        manager = JobManager()

        async def boom():
            raise RuntimeError("extractor exploded")

        job = await manager.submit("build", "corpus", boom)
        for _ in range(50):
            await asyncio.sleep(0)
            job = await manager.store.get(job.id)
            if job.finished:
                break
        assert job.state == "failed"
        assert "extractor exploded" in job.error

    async def test_a_running_job_can_be_cancelled(self):
        import asyncio

        from ragu.api.runtime.jobs import JobManager

        manager = JobManager()

        async def forever():
            await asyncio.sleep(60)

        job = await manager.submit("build", "corpus", forever)
        await asyncio.sleep(0)
        cancelled = await manager.cancel(job.id)

        for _ in range(50):
            cancelled = await manager.store.get(job.id)
            if cancelled.finished:
                break
            await asyncio.sleep(0)
        assert cancelled.state == "cancelled"

    async def test_shutdown_stops_what_is_still_running(self):
        import asyncio

        from ragu.api.runtime.jobs import JobManager

        manager = JobManager()

        async def forever():
            await asyncio.sleep(60)

        await manager.submit("build", "corpus", forever)
        await asyncio.sleep(0)
        await manager.shutdown()

        assert manager._tasks == {}


class TestReindex:
    def test_reindex_is_a_job(self):
        with build_client() as client:
            response = client.post("/v1/graphs/default/reindex/community")
        assert response.status_code == 202
        body = response.json()
        assert body["kind"] == "reindex"
        assert response.headers["Location"] == f"/v1/jobs/{body['id']}"

    def test_an_unknown_reindex_fails_the_job(self):
        with build_client() as client:
            job_id = client.post("/v1/graphs/default/reindex/nonsense").json()["id"]
            for _ in range(50):
                job = client.get(f"/v1/jobs/{job_id}").json()
                if job["state"] in ("succeeded", "failed"):
                    break
        assert job["state"] == "failed"
        assert "Unknown reindex" in job["error"]
