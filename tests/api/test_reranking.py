"""
The reranker: built from the environment, and forgiven when it fails.
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
    build_client,
    make_retrieval,
)


class TestReranking:
    """A reranker outage costs ranking quality, not the answer."""

    def make_backend(self, scorer):
        settings = ServiceSettings(backend="ragu")
        from ragu.api.backends.ragu_backend import RaguBackend

        backend = RaguBackend(settings, settings.resolved_graphs()[0], reranker=scorer)
        backend.graph = object()
        backend._llm = object()
        backend._embedder = object()
        backend._stats = GraphStats(entities=1, chunks=1, community_summaries=1)

        class FakeEngine:
            """Calls the reranker the way the real engines do."""

            def __init__(self, mode, language, rerank=True):
                self.reranker = backend._reranker if rerank else None
                self.llm = object()

            async def batch_query(self, queries, params=None):
                if self.reranker is not None:
                    await self.reranker.score(queries[0], ["a", "b"])
                return [
                    SearchEngineResponse(
                        query=query,
                        response="ответ",
                        retrieval=make_retrieval("text"),
                    )
                    for query in queries
                ]

        backend._build_engine = FakeEngine
        return backend

    async def test_a_failing_reranker_still_answers(self):
        from ragu.models.scorer import Scorer

        class BrokenScorer(Scorer):
            async def score(self, text_1, text_2, **kwargs):
                raise RuntimeError("reranker container is down")

        backend = self.make_backend(BrokenScorer())

        outcome, = await backend.search(SearchCall(mode="naive", queries=("q",)))

        assert outcome.answer == "ответ"
        assert outcome.engines.reranked is False
        assert "reranker container is down" in outcome.engines.rerank_error
        assert outcome.engines.degraded is True

    async def test_a_failing_reranker_keeps_the_retrieval_order(self):
        from ragu.models.scorer import Scorer
        from ragu.api.search.reranking import ForgivingScorer

        class BrokenScorer(Scorer):
            async def score(self, text_1, text_2, **kwargs):
                raise RuntimeError("boom")

        ranked = await ForgivingScorer(BrokenScorer()).score("q", ["a", "b", "c"])
        assert [index for index, _ in ranked] == [0, 1, 2]

    async def test_a_slow_reranker_is_given_up_on(self):
        import asyncio

        from ragu.models.scorer import Scorer
        from ragu.api.search.reranking import ForgivingScorer, rerank_failure, reset_rerank_report

        class SlowScorer(Scorer):
            async def score(self, text_1, text_2, **kwargs):
                await asyncio.sleep(5)
                return []

        reset_rerank_report()
        ranked = await ForgivingScorer(SlowScorer(), timeout=0.01).score("q", ["a"])

        assert [index for index, _ in ranked] == [0]
        assert "timed out" in rerank_failure()

    async def test_a_working_reranker_is_reported_as_used(self):
        from ragu.models.scorer import Scorer

        class GoodScorer(Scorer):
            async def score(self, text_1, text_2, **kwargs):
                return [(index, 1.0) for index in reversed(range(len(text_2)))]

        backend = self.make_backend(GoodScorer())

        outcome, = await backend.search(SearchCall(mode="naive", queries=("q",)))

        assert outcome.engines.reranked is True
        assert outcome.engines.rerank_error is None
        assert outcome.engines.degraded is False

    async def test_a_client_can_turn_reranking_off(self):
        from ragu.models.scorer import Scorer

        class GoodScorer(Scorer):
            async def score(self, text_1, text_2, **kwargs):
                return []

        backend = self.make_backend(GoodScorer())

        outcome, = await backend.search(
            SearchCall(mode="naive", queries=("q",), rerank=False)
        )

        assert outcome.engines.reranked is False
        assert outcome.engines.rerank_error is None

    async def test_no_reranker_configured_is_not_a_degradation(self):
        backend = self.make_backend(None)

        outcome, = await backend.search(SearchCall(mode="naive", queries=("q",)))

        assert outcome.engines.reranked is False
        assert outcome.engines.degraded is False

    def test_rerank_is_a_request_field(self):
        with build_client() as client:
            response = client.post(
                "/v1/search/naive",
                json={"query": "q", "rerank": False, "params": {"rerank_top_k": 3}},
            )
        assert response.status_code == 200


class TestRerankerFromTheCommandLine:
    """
    ``python -m ragu.api`` builds the reranker the environment names.

    ``create_app`` takes a reranker as a parameter, and the command line never
    passed one: the documented injection point was unreachable from the only
    entry point most deployments use.
    """

    @staticmethod
    def _clear(monkeypatch):
        import os

        for key in list(os.environ):
            if key.startswith(("LLM_", "RERANKER_", "EMBEDDER_")):
                monkeypatch.delenv(key, raising=False)

    def test_none_when_the_environment_names_none(self, monkeypatch):
        from ragu.api.search.reranking import reranker_from_env

        self._clear(monkeypatch)
        monkeypatch.setenv("LLM_BASE_URL", "http://llm")
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_MODEL_NAME", "m")
        assert reranker_from_env() is None

    def test_missing_llm_credentials_do_not_crash_startup(self, monkeypatch):
        # The backend reports missing credentials through /health; raising here
        # would turn that into a container that restarts without saying why.
        from ragu.api.search.reranking import reranker_from_env

        self._clear(monkeypatch)
        assert reranker_from_env() is None

    def test_a_named_reranker_is_built(self, monkeypatch):
        from ragu.api.search.reranking import reranker_from_env
        from ragu.models.scorer import ScorerOpenAI

        self._clear(monkeypatch)
        monkeypatch.setenv("LLM_BASE_URL", "http://llm")
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_MODEL_NAME", "m")
        monkeypatch.setenv("RERANKER_BASE_URL", "http://reranker:8000/v1")
        monkeypatch.setenv("RERANKER_MODEL_NAME", "bge-reranker")

        reranker = reranker_from_env()
        assert isinstance(reranker, ScorerOpenAI)
        assert reranker.model_name == "bge-reranker"

    def test_a_reranker_without_a_model_is_left_out(self, monkeypatch):
        from ragu.api.search.reranking import reranker_from_env

        self._clear(monkeypatch)
        monkeypatch.setenv("LLM_BASE_URL", "http://llm")
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_MODEL_NAME", "m")
        monkeypatch.setenv("RERANKER_BASE_URL", "http://reranker:8000/v1")
        assert reranker_from_env() is None

    def test_the_command_line_passes_it_to_the_app(self, monkeypatch):
        import sys

        import ragu.api.__main__ as entry

        sentinel = object()
        seen = {}
        monkeypatch.setattr(entry, "reranker_from_env", lambda: sentinel)
        monkeypatch.setattr(
            entry, "create_app",
            lambda settings, reranker=None: seen.setdefault("reranker", reranker),
        )
        monkeypatch.setattr(entry.uvicorn, "run", lambda *a, **k: None)
        monkeypatch.setattr(entry, "configure_logging", lambda level: None)
        monkeypatch.setattr(sys, "argv", ["ragu.api", "--backend", "ragu"])

        entry.main()
        assert seen["reranker"] is sentinel
