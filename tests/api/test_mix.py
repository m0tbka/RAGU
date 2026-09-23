"""
The mix ensemble: what it runs, and how honestly it reports what ran.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "fastapi", reason="install the 'api' extra to test the search service"
)

from ragu.api.backends.base import SearchCall
from ragu.api.backends.capabilities import GraphStats
from ragu.search_engine.local_search import (
    LocalParams,
)
from ragu.search_engine.mix_search import MixQueryParams
from ragu.search_engine.naive_search import (
    NaiveSearchParams,
    NaiveSearchResult,
    NaiveSearchRetrieve,
)
from tests.api.support import (
    build_client,
    make_chunk,
    make_backend,
)


class TestMixSearch:
    """The ensemble mode, and the honesty of its engine report."""

    def test_mix_combines_entity_and_chunk_sources(self):
        with build_client() as client:
            body = client.post(
                "/v1/search/mix",
                json={"query": "q", "naive_params": {"top_k": 2}},
            ).json()
        assert body["mode"] == "mix"
        # Each child contributes what it would answer alone — the local engine
        # its entity and chunk, the naive one its chunks — and the chunk they
        # both return appears once.
        assert [source["type"] for source in body["sources"]] == [
            "entity",
            "chunk",
            "chunk",
        ]

    def test_mix_reports_which_children_ran(self):
        with build_client() as client:
            engines = client.post("/v1/search/mix", json={"query": "q"}).json()["engines"]
        assert engines["requested"] == "mix"
        assert engines["degraded"] is False
        assert [child["mode"] for child in engines["children"]] == ["local", "naive"]

    def test_mix_needs_both_stores(self):
        # It ensembles local and naive, so either one missing makes it impossible.
        with build_client(missing="entity_graph") as client:
            error = client.post("/v1/search/mix", json={"query": "q"}).json()["error"]
        assert error["missing_capability"] == "entity_graph"

        with build_client(missing="vector_index") as client:
            error = client.post("/v1/search/mix", json={"query": "q"}).json()["error"]
        assert error["missing_capability"] == "vector_index"

    def test_child_params_are_named_separately(self):
        # MixSearchEngine takes child parameters from its constructor, so the
        # request cannot fold them into one `params` object.
        with build_client() as client:
            response = client.post(
                "/v1/search/mix", json={"query": "q", "params": {"top_k": 3}}
            )
        assert response.status_code == 400

    def test_child_params_are_bounded_like_any_other(self):
        with build_client(max_top_k=4) as client:
            body = client.post(
                "/v1/search/mix",
                json={"query": "q", "naive_params": {"top_k": 900}},
            ).json()
        assert len([s for s in body["sources"] if s["type"] == "chunk"]) == 4

    def test_every_mode_reports_the_engine_that_ran(self):
        with build_client() as client:
            for path in ("global", "local", "naive", "mix"):
                body = client.post(f"/v1/search/{path}", json={"query": "q"}).json()
                assert body["engines"]["requested"] == path
                assert body["engines"]["used"]


class TestRecordingEngine:
    """The proxy that makes MixSearchEngine's swallowed failures visible."""

    def make_child(self, fail: bool = False):
        from ragu.api.backends.ragu_backend.engines import RecordingEngine

        class Child:
            llm = object()

            async def batch_search(self, queries, params=None):
                if fail:
                    raise RuntimeError("qdrant unreachable")
                return ["context"]

            async def batch_query(self, queries, params=None):
                return ["answer"]

        return RecordingEngine(Child(), "local")

    async def test_a_successful_child_reports_ok(self):
        child = self.make_child()
        await child.batch_search(["q"])
        report = child.report()
        assert (report.ok, report.error, report.mode) == (True, None, "local")

    async def test_a_failing_child_keeps_the_reason(self):
        child = self.make_child(fail=True)
        with pytest.raises(RuntimeError):
            await child.batch_search(["q"])
        report = child.report()
        assert report.ok is False
        assert "qdrant unreachable" in report.error

    async def test_an_unused_child_is_not_reported_as_ok(self):
        assert self.make_child().report().ok is False

    def test_the_proxy_exposes_the_llm_the_ensemble_needs(self):
        # MixSearchEngine reads engine.llm off its children.
        child = self.make_child()
        assert child.llm is child.engine.llm


class TestMixDegradation:
    """A failing child must not look like a healthy ensemble."""

    class FakeLLM:
        async def batch_chat_completion(self, conversations, output_schema=None, **kw):
            return ["синтез"] * len(conversations)

    class WorkingChild:
        def __init__(self, llm):
            self.llm = llm

        async def batch_search(self, queries, params=None):
            return [
                NaiveSearchRetrieve(
                    query=query,
                    result=NaiveSearchResult(
                        chunks=[make_chunk("chunk text")], scores=[0.5]
                    ),
                )
                for query in queries
            ]

        async def batch_query(self, queries, params=None):
            raise AssertionError("not used in this test")

    class FailingChild:
        def __init__(self, llm):
            self.llm = llm

        async def batch_search(self, queries, params=None):
            raise RuntimeError("entity vector store unreachable")

        async def batch_query(self, queries, params=None):
            raise RuntimeError("entity vector store unreachable")

    def build_backend(self):
        backend = make_backend()
        backend.graph = object()
        backend._stats = GraphStats(
            entities=1, relations=1, chunks=1, community_summaries=1
        )
        llm = self.FakeLLM()
        backend._llm = llm
        language = backend.language
        backend._engines = {
            ("local", language, False): self.FailingChild(llm),
            ("naive", language, False): self.WorkingChild(llm),
        }
        return backend

    async def test_a_dropped_child_is_reported_not_hidden(self):
        # MixSearchEngine runs with allow_partial_failures=True, so the answer
        # still comes back — built on chunks alone.
        backend = self.build_backend()

        outcome, = await backend.search(
            SearchCall(
                mode="mix",
                queries=("q",),
                params=MixQueryParams(),
                local_params=LocalParams(),
                naive_params=NaiveSearchParams(top_k=1),
            )
        )

        assert outcome.answer == "синтез"
        assert outcome.engines.used == "MixSearchEngine"
        assert outcome.engines.degraded is True

        by_mode = {child.mode: child for child in outcome.engines.children}
        assert by_mode["local"].ok is False
        assert "entity vector store unreachable" in by_mode["local"].error
        assert by_mode["naive"].ok is True
        assert by_mode["naive"].error is None

    async def test_a_healthy_ensemble_is_not_marked_degraded(self):
        backend = self.build_backend()
        backend._engines[("local", backend.language, False)] = self.WorkingChild(
            backend._llm
        )

        outcome, = await backend.search(
            SearchCall(
                mode="mix",
                queries=("q",),
                params=MixQueryParams(),
                local_params=LocalParams(),
                naive_params=NaiveSearchParams(top_k=1),
            )
        )

        assert outcome.engines.degraded is False
        assert all(child.ok for child in outcome.engines.children)


class TestMixEnsembleSelection:
    """The request chooses the ensemble, and the graph is checked against it."""

    def test_the_default_ensemble_is_local_and_naive(self):
        with build_client() as client:
            body = client.post("/v1/search/mix", json={"query": "q"}).json()
        assert [c["mode"] for c in body["engines"]["children"]] == ["local", "naive"]

    def test_a_chosen_ensemble_is_the_one_that_runs(self):
        with build_client() as client:
            body = client.post(
                "/v1/search/mix",
                json={"query": "q", "engines": ["local", "naive", "global"]},
            ).json()
        assert [c["mode"] for c in body["engines"]["children"]] == [
            "local",
            "naive",
            "global",
        ]

    def test_requirements_follow_the_ensemble_not_the_mode(self):
        # The point of making this per-request: a graph with no community
        # summaries still serves the default pair, and only refuses the
        # ensemble that actually needs them.
        with build_client(missing="community_summaries") as client:
            default = client.post("/v1/search/mix", json={"query": "q"})
            with_global = client.post(
                "/v1/search/mix", json={"query": "q", "engines": ["local", "global"]}
            )
        assert default.status_code == 200
        assert with_global.status_code == 409
        assert with_global.json()["error"]["missing_capability"] == "community_summaries"

    def test_a_single_child_ensemble_needs_only_that_child(self):
        with build_client(missing="vector_index") as client:
            response = client.post(
                "/v1/search/mix", json={"query": "q", "engines": ["local"]}
            )
        assert response.status_code == 200

    def test_an_unknown_engine_is_refused(self):
        with build_client() as client:
            unknown = client.post(
                "/v1/search/mix", json={"query": "q", "engines": ["bogus"]}
            )
            empty = client.post("/v1/search/mix", json={"query": "q", "engines": []})
        assert unknown.status_code == 400
        assert empty.status_code == 400

    def test_the_flat_paths_older_clients_use_are_untouched(self):
        # sgr-agent-ragu posts exactly these three bodies.
        with build_client() as client:
            assert client.post("/v1/search/global", json={"query": "q"}).status_code == 200
            assert client.post(
                "/v1/search/local",
                json={"query": "q", "use_query_plan": True,
                      "params": {"use_summary": True, "use_chunks": True}},
            ).status_code == 200
            assert client.post(
                "/v1/search/naive",
                json={"query": "q", "use_query_plan": True, "params": {"top_k": 5}},
            ).status_code == 200
