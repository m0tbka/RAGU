"""
The stub backend, which clients are built against when there is no graph.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "fastapi", reason="install the 'api' extra to test the search service"
)

from ragu.api.backends.base import SearchCall
from ragu.api.backends.capabilities import GraphStats
from ragu.api.config import ServiceSettings
from ragu.search_engine.base_engine import SearchEngineResponse
from tests.api.support import (
    make_backend,
    make_retrieval,
)


class TestBackendContract:
    """The stub exists so clients can be built without a graph.

    That only holds if it answers in the same shape the real backend does, and
    nothing checked it until now.
    """

    def real_backend(self):
        backend = make_backend()
        backend.graph = object()
        backend._llm = object()
        backend._embedder = object()
        backend._stats = GraphStats(
            entities=1, relations=1, chunks=1, community_summaries=1
        )

        class FakeEngine:
            def __init__(self, mode, language, rerank=True):
                self.llm = object()

            async def batch_query(self, queries, params=None):
                return [
                    SearchEngineResponse(
                        query=query,
                        response="ответ",
                        retrieval=make_retrieval("text"),
                    )
                    for query in queries
                ]

            async def batch_search(self, queries, params=None):
                return [make_retrieval("text") for _ in queries]

        backend._build_engine = FakeEngine
        return backend

    def stub_backend(self):
        from ragu.api.backends.stub import StubBackend

        backend = StubBackend(ServiceSettings(backend="stub"))
        backend._stats = backend._simulated_stats()
        return backend

    @pytest.fixture(params=["stub", "ragu"])
    def backend(self, request):
        return self.stub_backend() if request.param == "stub" else self.real_backend()

    async def test_search_answers_in_one_shape(self, backend):
        outcomes = await backend.search(SearchCall(mode="naive", queries=("q",)))

        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert isinstance(outcome.answer, str) and outcome.answer
        assert outcome.sources and outcome.sources[0].type == "chunk"
        assert outcome.engines is not None
        assert outcome.engines.requested == "naive"

    async def test_retrieve_answers_in_one_shape(self, backend):
        outcomes = await backend.retrieve(SearchCall(mode="naive", queries=("q",)))

        assert len(outcomes) == 1
        assert outcomes[0].sources
        assert outcomes[0].engines is not None

    async def test_a_batch_is_aligned_with_its_queries(self, backend):
        outcomes = await backend.search(
            SearchCall(mode="naive", queries=("a", "b", "c"))
        )

        assert len(outcomes) == 3

    async def test_both_refuse_a_mode_the_graph_cannot_serve(self, backend):
        from ragu.api.errors import CapabilityUnavailableError

        backend._stats = GraphStats(chunks=1)

        with pytest.raises(CapabilityUnavailableError) as failure:
            await backend.search(SearchCall(mode="local", queries=("q",)))
        assert failure.value.missing_capability == "entity_graph"

    async def test_both_expose_the_same_capability_map(self, backend):
        assert set(backend.capabilities()) == {"global", "local", "naive", "mix"}

    async def test_both_page_entities_the_same_way(self, backend):
        # The stub answers from a canned graph, the real one from the loaded
        # one; the shape of the answer is what has to match.
        class FakeGraphBackend:
            async def get_all_nodes(self):
                return []

        class FakeIndex:
            graph_backend = FakeGraphBackend()

        class FakeGraph:
            index = FakeIndex()

        if type(backend).__name__ == "RaguBackend":
            backend.graph = FakeGraph()

        total, entities = await backend.list_entities(limit=10, offset=0)
        assert isinstance(total, int)
        assert isinstance(entities, list)
