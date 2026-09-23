"""
Regressions for concurrency defects that were reproduced at runtime.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "fastapi", reason="install the 'api' extra to test the search service"
)

from ragu.api.config import ServiceSettings
from tests.api.support import (
    build_client,
)


class TestConcurrencyHazards:
    """
    Regressions for defects a max-effort review reproduced at runtime.

    Each of these passed a static reading and failed when actually executed, so
    they are pinned here rather than left to the next reviewer.
    """

    async def test_a_rerank_failure_survives_the_engines_gather(self):
        # ForgivingScorer records into a ContextVar, and every engine reaches the
        # scorer through asyncio.gather, which copies the context per child task.
        # Recording with set() there was discarded with the task, so a reranker
        # that was completely down still reported rerank_error: null.
        import asyncio

        from ragu.api.search.reranking import (
            ForgivingScorer,
            rerank_failure,
            reset_rerank_report,
        )
        from ragu.models.scorer import Scorer

        class Broken(Scorer):
            async def score(self, text_1, text_2, **kwargs):
                raise RuntimeError("reranker is down")

        scorer = ForgivingScorer(Broken())

        reset_rerank_report()
        await asyncio.gather(scorer.score("q", ["a", "b"]))
        assert rerank_failure() == "RuntimeError: reranker is down"

        reset_rerank_report()
        await scorer.batch_score([("q", ["a", "b"])])
        assert rerank_failure() == "RuntimeError: reranker is down"

    async def test_the_degraded_order_is_the_retrieval_order(self):
        from ragu.api.search.reranking import ForgivingScorer
        from ragu.models.scorer import Scorer

        class Broken(Scorer):
            async def score(self, text_1, text_2, **kwargs):
                raise RuntimeError("down")

        import math

        degraded = await ForgivingScorer(Broken()).score("q", ["a", "b", "c"])
        # The retrieval order survives, and the scores are NaN rather than 0.0:
        # a zero is a real score and would reach a client as one.
        assert [index for index, _ in degraded] == [0, 1, 2]
        assert all(math.isnan(score) for _, score in degraded)

    async def test_two_graph_builds_do_not_cross_the_settings_singleton(self):
        # Settings is process-global and the write lock is per backend, so two
        # ingestion jobs on different graphs used to interleave their snapshots:
        # each built into the folder of the other and the singleton was left
        # restored to the wrong one.
        import asyncio

        from ragu.api.backends.ragu_backend import exclusive_settings
        from ragu.common.global_parameters import Settings

        original = Settings.storage_folder
        Settings.storage_folder = "original"
        seen = []
        try:

            async def build(folder, hold):
                async with exclusive_settings():
                    Settings.storage_folder = folder
                    await asyncio.sleep(hold)
                    seen.append((folder, Settings.storage_folder))

            await asyncio.gather(build("graph-a", 0.02), build("graph-b", 0.01))

            # Whatever the order, neither block ever saw the folder of the other.
            assert sorted(seen) == [("graph-a", "graph-a"), ("graph-b", "graph-b")]
            assert Settings.storage_folder == "original"
        finally:
            Settings.storage_folder = original

    def test_a_non_ascii_key_is_refused_not_crashed(self):
        # Headers arrive latin-1 decoded and hmac.compare_digest refuses a str
        # carrying anything above ASCII, which turned a junk key into a 500.
        from starlette.requests import Request

        from ragu.api.runtime.auth import authorize
        from ragu.api.errors import UnauthorizedError

        settings = ServiceSettings(backend="stub", api_keys="secret-key")
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/search/naive",
            "headers": [(b"x-api-key", b"\xff-key")],
            "query_string": b"",
        }
        with pytest.raises(UnauthorizedError):
            authorize(Request(scope), settings)

    def test_a_chunked_body_is_bounded_too(self):
        # The limit used to read Content-Length only, and a chunked request
        # carries none: the ceiling simply did not apply to it.
        import json

        def chunks():
            payload = json.dumps({"query": "x" * 5000}).encode()
            for start in range(0, len(payload), 256):
                yield payload[start : start + 256]

        with build_client(max_body_bytes=200) as client:
            response = client.post(
                "/v1/search/naive",
                content=chunks(),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"

    def test_a_graph_named_like_the_prefix_keeps_one_metric_series(self):
        # The template was rebuilt with str.replace, which hit the first match
        # anywhere in the path: a graph named "v1" rewrote the prefix and left
        # its own id in the label, one time series per graph.
        from starlette.requests import Request

        from ragu.api.runtime.middleware import _route_template

        def template(path, params):
            return _route_template(
                Request(
                    {
                        "type": "http",
                        "method": "GET",
                        "path": path,
                        "headers": [],
                        "query_string": b"",
                        "path_params": params,
                    }
                )
            )

        assert (
            template("/v1/graphs/v1/capabilities", {"graph_id": "v1"})
            == "/v1/graphs/{graph_id}/capabilities"
        )
        assert (
            template("/v1/graphs/graphs/capabilities", {"graph_id": "graphs"})
            == "/v1/graphs/{graph_id}/capabilities"
        )
        assert (
            template("/v1/graphs/s/search/naive", {"graph_id": "s"})
            == "/v1/graphs/{graph_id}/search/naive"
        )
        assert template("/v1/jobs/1", {"job_id": "1"}) == "/v1/jobs/{job_id}"

    async def test_one_idempotency_key_starts_one_job(self):
        # submit() checks then creates across two awaits; without a lock a client
        # retry racing its own first attempt built the same corpus twice.
        import asyncio

        from ragu.api.runtime.jobs import JobManager

        manager = JobManager()

        async def work():
            return None

        try:
            jobs = await asyncio.gather(
                *[
                    manager.submit("build", "corpus", work, idempotency_key="k")
                    for _ in range(8)
                ]
            )
            assert len({job.id for job in jobs}) == 1
        finally:
            await manager.shutdown()

    async def test_evicting_a_job_forgets_its_idempotency_key(self):
        # _evict trimmed the job map and left the key map growing forever, one
        # dead entry per submission for the life of the process.
        from ragu.api.runtime.jobs import InMemoryJobStore, Job

        store = InMemoryJobStore(limit=1)
        for index in range(4):
            job = Job(
                id=f"job-{index}",
                kind="build",
                graph_id="corpus",
                state="succeeded",
            )
            await store.put(job)
            await store.remember_key(f"key-{index}", job.id)

        assert len(await store.list()) <= 1
        assert len(store._keys) <= 1
