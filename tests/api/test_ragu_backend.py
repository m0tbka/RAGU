"""
The real backend: how it drives the engines, starts, stops, and keeps one
graph's settings out of the next.
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
from ragu.search_engine.global_search import (
    GlobalSearchParams,
)
from ragu.search_engine.local_search import (
    LocalParams,
)
from ragu.search_engine.naive_search import (
    NaiveSearchParams,
    NaiveSearchResult,
    NaiveSearchRetrieve,
)
from tests.api.support import (
    make_chunk,
    make_backend,
)


class TestEngineInvocation:
    """The adapter passes what today's engines accept.

    Engine parameter classes come and go between RAGU versions (0.0.5 dropped
    ``GlobalSearchParams``), and a mismatch only shows up as a 500 at request
    time, so the call into the engine is exercised here.
    """

    class FakeEngine:
        """Stands in for a leaf engine; batch_* is the real entry point."""

        def __init__(self):
            self.calls = []
            self.retrievals = []
            self.llm = object()
            self.result = NaiveSearchResult(chunks=[make_chunk("text")], scores=[0.5])

        async def batch_query(self, queries, params=None):
            self.calls.extend((query, params) for query in queries)
            return [
                SearchEngineResponse(
                    query=query,
                    response="ответ",
                    retrieval=NaiveSearchRetrieve(query=query, result=self.result),
                )
                for query in queries
            ]

        async def batch_search(self, queries, params=None):
            self.retrievals.extend((query, params) for query in queries)
            return [
                NaiveSearchRetrieve(query=query, result=self.result)
                for query in queries
            ]

    def build_backend(self, **overrides):
        backend = make_backend(**overrides)
        backend.graph = object()
        backend._llm = object()
        backend._stats = GraphStats(
            entities=1, relations=1, chunks=1, community_summaries=1
        )
        engines = {mode: self.FakeEngine() for mode in ("global", "local", "naive")}
        # Engines are cached per (mode, language): every engine bakes the answer
        # language in at construction.
        backend._engines = {
            (mode, backend.language, False): engine
            for mode, engine in engines.items()
        }
        return backend, engines

    async def test_local_search_passes_context_flags_as_params(self):
        backend, engines = self.build_backend()

        outcome, = await backend.search(
            SearchCall(
                mode="local",
                queries=("кто написал",),
                params=LocalParams(top_k=5, use_summary=False, use_chunks=True),
            )
        )

        query, params = engines["local"].calls[0]
        assert query == "кто написал"
        assert (params.use_summary, params.use_chunks) == (False, True)
        assert params.top_k == 5
        assert outcome.answer == "ответ"
        assert outcome.sources[0].type == "chunk"

    async def test_naive_search_passes_top_k(self):
        backend, engines = self.build_backend()

        await backend.search(
            SearchCall(mode="naive", queries=("версия",), params=NaiveSearchParams(top_k=7))
        )

        _, params = engines["naive"].calls[0]
        assert params.top_k == 7

    async def test_global_search_passes_its_params(self):
        backend, engines = self.build_backend()
        params = GlobalSearchParams(min_cluster_size=3)

        await backend.search(SearchCall(mode="global", queries=("тренды",), params=params))

        assert engines["global"].calls == [("тренды", params)]

    @pytest.mark.parametrize("mode", ["global", "local", "naive"])
    async def test_a_search_without_evidence_is_reported_not_answered(self, mode):
        # Every engine answers from whatever it retrieved, empty included, so an
        # empty retrieval must not reach the client as an answer.
        from ragu.api.errors import CapabilityUnavailableError

        backend, engines = self.build_backend()
        engines[mode].result = NaiveSearchResult()

        outcome, = await backend.search(SearchCall(mode=mode, queries=("q",)))
        assert outcome.sources == []

        # A batch reports emptiness per query; a single-query route turns it
        # into a 409 rather than returning an answer built on nothing.
        with pytest.raises(CapabilityUnavailableError) as failure:
            backend.require_evidence(mode, outcome)
        assert failure.value.mode == mode
        assert failure.value.status_code == 409
        assert failure.value.missing_capability is None

    @pytest.mark.parametrize(
        "mode, stats, capability",
        [
            ("global", GraphStats(entities=1, chunks=1), "community_summaries"),
            ("local", GraphStats(chunks=1, community_summaries=1), "entity_graph"),
            ("naive", GraphStats(entities=1, community_summaries=1), "vector_index"),
        ],
    )
    async def test_an_unsupported_mode_is_refused_before_generating(
        self, mode, stats, capability
    ):
        # The whole point of measuring the graph at startup: no generation call
        # is paid for on a graph that cannot serve the mode.
        from ragu.api.errors import CapabilityUnavailableError

        backend, engines = self.build_backend()
        backend._stats = stats

        with pytest.raises(CapabilityUnavailableError) as failure:
            await backend.search(SearchCall(mode=mode, queries=("q",)))

        assert failure.value.missing_capability == capability
        assert engines[mode].calls == []

    async def test_evidence_is_returned_as_an_answer(self):
        backend, _ = self.build_backend()

        outcome, = await backend.search(SearchCall(mode="naive", queries=("q",)))

        assert outcome.answer == "ответ"
        assert len(outcome.sources) == 1

    async def test_retrieval_skips_generation_entirely(self):
        backend, engines = self.build_backend()

        outcome, = await backend.retrieve(
            SearchCall(mode="naive", queries=("q",), params=NaiveSearchParams(top_k=2))
        )

        assert engines["naive"].calls == []
        assert engines["naive"].retrievals[0][0] == "q"
        assert [source.type for source in outcome.sources] == ["chunk"]
        assert outcome.engines.query_plan is False

    async def test_retrieval_is_not_wrapped_in_a_query_plan(self, monkeypatch):
        # QueryPlanEngine.batch_search delegates to the wrapped engine and does
        # no planning, so wrapping would only imply a decomposition that never
        # happens.
        class ForbiddenPlanEngine:
            def __init__(self, engine, *args, **kwargs):
                raise AssertionError("retrieval must not be wrapped in a query plan")

        monkeypatch.setattr(
            "ragu.api.backends.ragu_backend.backend.QueryPlanEngine", ForbiddenPlanEngine
        )
        backend, _ = self.build_backend()

        outcome, = await backend.retrieve(
            SearchCall(mode="naive", queries=("q",), use_query_plan=True)
        )

        assert outcome.sources

    async def test_query_plan_wraps_the_engine_with_no_extra_arguments(
        self, monkeypatch
    ):
        wrapped = {}

        class FakePlanEngine:
            def __init__(self, engine, *args, **kwargs):
                wrapped["engine"] = engine
                wrapped["args"] = (args, kwargs)
                self.engine = engine

            async def batch_query(self, queries, params=None):
                return await self.engine.batch_query(queries, params)

        # The backend imports the symbol into its own namespace, so patching
        # ``ragu.QueryPlanEngine`` would not reach the code under test.
        monkeypatch.setattr(
            "ragu.api.backends.ragu_backend.backend.QueryPlanEngine", FakePlanEngine
        )
        backend, engines = self.build_backend()

        await backend.search(
            SearchCall(
                mode="naive",
                queries=("версия",),
                params=NaiveSearchParams(top_k=3),
                use_query_plan=True,
            )
        )

        # RAGU 0.0.5 dropped the `language` argument; anything extra here is a
        # TypeError at request time, on the default path of two of three modes.
        assert wrapped["engine"] is engines["naive"]
        assert wrapped["args"] == ((), {})
        assert engines["naive"].calls[0][1].top_k == 3

    async def test_engine_construction_failures_report_their_mode(self, monkeypatch):
        from ragu.api.errors import BackendExecutionError

        class ExplodingPlanEngine:
            def __init__(self, engine, *args, **kwargs):
                raise TypeError("unexpected keyword argument")

        monkeypatch.setattr(
            "ragu.api.backends.ragu_backend.backend.QueryPlanEngine", ExplodingPlanEngine
        )
        backend, _ = self.build_backend()

        with pytest.raises(BackendExecutionError) as failure:
            await backend.search(
                SearchCall(mode="local", queries=("q",), use_query_plan=True)
            )
        assert failure.value.mode == "local"
        assert failure.value.detail == "unexpected keyword argument"
        # The exception text belongs in the log, not in the client's message.
        assert "unexpected keyword argument" not in failure.value.message

    async def test_top_k_is_clamped_to_the_service_limit(self):
        # The engine parameter classes carry no bounds and now arrive straight
        # from the request body.
        backend, engines = self.build_backend(max_top_k=50)

        await backend.search(
            SearchCall(
                mode="naive", queries=("версия",), params=NaiveSearchParams(top_k=10_000)
            )
        )

        assert engines["naive"].calls[0][1].top_k == 50

    async def test_rerank_top_k_is_clamped_too(self):
        backend, engines = self.build_backend(max_top_k=50)

        await backend.search(
            SearchCall(
                mode="naive",
                queries=("версия",),
                params=NaiveSearchParams(top_k=10, rerank_top_k=9_000),
            )
        )

        assert engines["naive"].calls[0][1].rerank_top_k == 50

    async def test_params_within_the_limit_are_passed_through_untouched(self):
        backend, engines = self.build_backend()
        params = NaiveSearchParams(top_k=5, rerank_top_k=2)

        await backend.search(SearchCall(mode="naive", queries=("версия",), params=params))

        assert engines["naive"].calls[0][1] is params

    async def test_global_search_never_reaches_the_planner(self, monkeypatch):
        class ForbiddenPlanEngine:
            def __init__(self, engine, *args, **kwargs):
                raise AssertionError(
                    "global search must not be wrapped in a query plan"
                )

        monkeypatch.setattr(
            "ragu.api.backends.ragu_backend.backend.QueryPlanEngine", ForbiddenPlanEngine
        )
        backend, engines = self.build_backend()

        await backend.search(SearchCall(mode="global", queries=("тренды",)))

        assert len(engines["global"].calls) == 1

    async def test_a_budget_refusal_is_not_flattened_into_a_500(self):
        # _query wraps engine failures as BackendExecutionError; a service error
        # already carries its own status and must pass through.
        from ragu.api.errors import BudgetExceededError

        backend, engines = self.build_backend()

        class BudgetedEngine:
            llm = object()

            async def batch_query(self, queries, params=None):
                raise BudgetExceededError("out of budget")

        backend._engines[("naive", backend.language, False)] = BudgetedEngine()

        with pytest.raises(BudgetExceededError):
            await backend.search(SearchCall(mode="naive", queries=("q",)))


class TestStorageFolderValidation:
    """A typo in the storage folder must not look like a healthy service."""

    def test_a_missing_folder_is_refused_and_not_created(self, tmp_path):
        from ragu.api.backends.ragu_backend import RaguBackend
        from ragu.api.errors import ServiceNotReadyError

        missing = tmp_path / "typo"
        with pytest.raises(ServiceNotReadyError) as failure:
            RaguBackend._require_storage_folder(str(missing))

        # Index.__init__ would have created it via Settings.init_storage_folder().
        assert not missing.exists()
        assert "does not exist" in failure.value.message

    def test_an_empty_folder_is_refused(self, tmp_path):
        from ragu.api.backends.ragu_backend import RaguBackend
        from ragu.api.errors import ServiceNotReadyError

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ServiceNotReadyError) as failure:
            RaguBackend._require_storage_folder(str(empty))
        assert "is empty" in failure.value.message

    def test_a_populated_folder_is_accepted(self, tmp_path):
        from ragu.api.backends.ragu_backend import RaguBackend

        populated = tmp_path / "graph"
        populated.mkdir()
        (populated / "knowledge_graph.gml").write_text("graph [ ]", encoding="utf-8")
        RaguBackend._require_storage_folder(str(populated))


class TestShutdown:
    async def test_shutdown_closes_the_index_and_the_model_clients(self):

        closed = []

        class FakeIndex:
            async def close(self):
                closed.append("index")

        class FakeGraph:
            index = FakeIndex()

        class FakeHTTPClient:
            async def close(self):
                closed.append("client")

        class FakeClient:
            client = FakeHTTPClient()

        backend = make_backend()
        backend.graph = FakeGraph()
        backend._stats = GraphStats(entities=1)
        backend._clients = [FakeClient()]

        await backend.shutdown()

        assert closed == ["index", "client"]
        assert backend.graph_loaded is False


class TestSettingsIsolation:
    """One graph's settings must not leak into the next one built."""

    def test_the_singleton_is_restored(self):
        from ragu.api.backends.ragu_backend import isolated_settings
        from ragu.common.global_parameters import Settings

        before = (Settings.storage_folder, Settings.language, Settings.llm_context_token_limit)

        with isolated_settings():
            Settings.storage_folder = "somewhere-else"
            Settings.language = "portuguese"
            Settings.llm_context_token_limit = 123

        assert (
            Settings.storage_folder,
            Settings.language,
            Settings.llm_context_token_limit,
        ) == before

    def test_it_restores_even_when_the_build_fails(self):
        from ragu.api.backends.ragu_backend import isolated_settings
        from ragu.common.global_parameters import Settings

        before = Settings.language
        with pytest.raises(RuntimeError):
            with isolated_settings():
                Settings.language = "portuguese"
                raise RuntimeError("graph failed to load")
        assert Settings.language == before

    async def test_a_failing_graph_does_not_take_down_the_others(self):
        from ragu.api.backends.stub import StubBackend
        from ragu.api.runtime.registry import GraphRegistry

        settings = ServiceSettings(
            backend="stub",
            graphs=[
                {"id": "broken", "storage_folder": "a"},
                {"id": "fine", "storage_folder": "b"},
            ],
        )

        def factory(service_settings, spec):
            if spec.id == "broken":
                class Broken(StubBackend):
                    async def startup(self):
                        raise RuntimeError("storage folder is empty")

                return Broken(service_settings, spec)
            return StubBackend(service_settings, spec)

        registry = GraphRegistry(settings, factory=factory)
        await registry.startup()

        assert registry.any_loaded is True
        assert "storage folder is empty" in registry.error("broken")
        assert registry.resolve("fine").graph_id == "fine"
        with pytest.raises(Exception) as failure:
            registry.resolve("broken")
        assert "storage folder is empty" in str(failure.value)


class TestEngineCacheEviction:
    """The cache exists to avoid rebuilding an engine; evicting the hot one defeats it."""

    def test_eviction_drops_the_least_recently_used_not_the_first_built(self):
        backend = make_backend(engine_cache_size=2)
        built = []

        def fake_build(mode, language, rerank=True):
            built.append((mode, language))
            return object()

        backend._build_engine = fake_build

        first = backend._leaf_engine("naive", "russian")
        backend._leaf_engine("naive", "english")
        # Touching the first one makes it the most recent, so the second is the
        # one that goes when a third arrives.
        assert backend._leaf_engine("naive", "russian") is first
        backend._leaf_engine("naive", "german")

        assert backend._leaf_engine("naive", "russian") is first
        assert len(built) == 3
