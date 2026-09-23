"""
Building the engines a request runs on.

Leaf engines are built per mode, language and reranker, and kept in a small LRU
because building one can load a tokenizer. ``mix`` is assembled per request
around them, and ``RecordingEngine`` notes which of its children actually
contributed.
"""

from typing import Any

from ragu import (
    GlobalSearchEngine,
    LocalSearchEngine,
    MixSearchEngine,
    NaiveSearchEngine,
)
from ragu.api.backends.base import SearchCall
from ragu.api.config import ServiceSettings
from ragu.api.errors import ServiceNotReadyError
from ragu.api.models import (
    DEFAULT_MIX_ENGINES,
    ChildEngineReport,
    EngineReport,
    SearchMode,
)
from ragu.api.search.reranking import ForgivingScorer, rerank_failure, reset_rerank_report
from ragu.models.llm import LLM
from ragu.search_engine.base_engine import BaseEngine
from ragu.search_engine.params import GlobalSearchParams, LocalParams, NaiveSearchParams


class RecordingEngine:
    """
    Child-engine proxy that remembers whether the child actually contributed.

    ``MixSearchEngine`` is constructed with ``allow_partial_failures=True``: a
    child that raises is logged and dropped from the ensemble, so from the
    outside "graph and chunks" is indistinguishable from "chunks only". This
    records the failure on the way past, before the ensemble swallows it.
    """

    def __init__(self, engine: BaseEngine[Any, Any], mode: SearchMode):
        self.engine = engine
        self.mode = mode
        self.error: str | None = None
        self.called = False

    @property
    def llm(self) -> LLM:
        return self.engine.llm

    async def batch_search(self, queries: list[str], params: Any = None) -> Any:
        return await self._record(self.engine.batch_search, queries, params)

    async def batch_query(self, queries: list[str], params: Any = None) -> Any:
        return await self._record(self.engine.batch_query, queries, params)

    async def _record(self, call: Any, queries: list[str], params: Any) -> Any:
        self.called = True
        try:
            return await call(queries, params)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            raise

    def report(self) -> ChildEngineReport:
        """
        Describe what this child did, for the response's engine report.
        """
        return ChildEngineReport(
            engine=type(self.engine).__name__,
            mode=self.mode,
            ok=self.called and self.error is None,
            error=self.error,
        )


class EngineAssembly:
    """
    The engine-building part of ``RaguBackend``, kept in its own module.

    A mixin rather than a component of its own: every method here works on the
    backend's state — the loaded graph and models, the reranker, the capability
    checks of ``SearchBackend`` — so a separate object would only hold a
    reference back to the backend.
    """

    # Provided by RaguBackend.
    settings: ServiceSettings
    language: str
    _engine_kwargs: dict[str, Any]
    _engines: dict[tuple[SearchMode, str, bool], BaseEngine[Any, Any]]
    _reranker: ForgivingScorer | None
    _llm: LLM | None

    def _build_engine(
        self, mode: SearchMode, language: str, rerank: bool = True
    ) -> BaseEngine[Any, Any]:
        """
        Construct one leaf engine for a mode in a language.

        :param mode: ``global``, ``local`` or ``naive``.
        :param language: Answer language baked into the engine.
        :param rerank: Whether this engine carries the reranker.
        :return: The engine.
        """
        graph, llm, embedder = self._require_loaded()
        if mode == "global":
            return GlobalSearchEngine(
                llm=llm,
                knowledge_graph=graph,
                language=language,
                **self._engine_kwargs,
            )
        engine_cls = LocalSearchEngine if mode == "local" else NaiveSearchEngine
        return engine_cls(
            llm=llm,
            knowledge_graph=graph,
            embedder=embedder,
            language=language,
            reranker=self._reranker if rerank else None,
            **self._engine_kwargs,
        )

    def _leaf_engine(
        self, mode: SearchMode, language: str, rerank: bool = True
    ) -> BaseEngine[Any, Any]:
        """
        Return a cached leaf engine, building it on first use.

        The engines bake ``language`` in at construction — every engine reads
        ``self.language`` when it renders the answer prompt — so a per-request
        language means a per-language engine. They are cached because with
        ``tokenizer_llm_backend="local"`` the constructor loads a HuggingFace
        tokenizer, which is far too expensive to repeat per request.

        :param mode: Leaf mode to build.
        :param language: Answer language.
        :return: The engine for that pair.
        """
        key = (mode, language, rerank)
        engine = self._engines.pop(key, None)
        if engine is not None:
            # Re-inserted, so it sits at the end: a dict keeps insertion order,
            # which makes "oldest key" mean least-recently-used rather than
            # first-built. Evicting by build order would drop the
            # default-language engines warmed at startup — the ones nearly every
            # request uses — as soon as a second language appeared.
            self._engines[key] = engine
            return engine

        engine = self._build_engine(mode, language, rerank)
        # A client picks the language, so the cache is bounded.
        while len(self._engines) >= self.settings.engine_cache_size:
            self._engines.pop(next(iter(self._engines)))
        self._engines[key] = engine
        return engine

    def _engine_for(self, call: SearchCall) -> tuple[Any, tuple[RecordingEngine, ...]]:
        """
        Build the engine that answers this call.

        ``mix`` is assembled per request: ``MixSearchEngine`` reads its
        children's parameters from the constructor — ``batch_search`` ignores its
        ``params`` argument and ``batch_query`` reads only ``ensemble_responses``
        — so the ensemble cannot be built once and parameterized later. The
        request also chooses *which* children run, so the ensemble is not one
        fixed pair.

        :param call: The resolved request.
        :return: The engine, and the recording proxies of its children if any.
        :raises ServiceNotReadyError: If the graph is not loaded.
        :raises CapabilityUnavailableError: If the graph cannot serve the mode.
        """
        if not self.graph_loaded:
            raise ServiceNotReadyError(
                "Knowledge graph is not loaded yet.", mode=call.mode
            )
        self.require_idle()
        self.require_capability(call.mode, call.mix_engines)
        language = call.language or self.language

        reset_rerank_report()
        rerank = call.rerank and self._reranker is not None

        if call.mode != "mix":
            return self._leaf_engine(call.mode, language, rerank), ()

        selected = tuple(call.mix_engines) or DEFAULT_MIX_ENGINES
        children = tuple(
            RecordingEngine(self._leaf_engine(child, language, rerank), child)
            for child in selected
        )
        engine = MixSearchEngine(
            llm=self._require_llm(),
            engines=list(children),
            # Positional, aligned with `engines`: MixSearchEngine zips the two.
            engine_params=[self._child_params(child, call) for child in selected],
            language=language,
        )
        return engine, children

    def _child_params(self, child: str, call: SearchCall) -> Any:
        """
        The parameter object one mix child runs with, bounded by the service.

        :param child: Leaf mode the child engine serves.
        :param call: The resolved request.
        :return: That child's parameters.
        """
        defaults: dict[str, Any] = {
            "local": call.local_params or LocalParams(),
            "naive": call.naive_params or NaiveSearchParams(),
            "global": call.global_params or GlobalSearchParams(),
        }
        return self.bound_params(defaults[child])

    def _report(
        self,
        call: SearchCall,
        engine_name: str,
        children: tuple[RecordingEngine, ...],
        *,
        query_plan: bool,
    ) -> EngineReport:
        reports = [child.report() for child in children]
        failure = rerank_failure()
        wanted_rerank = call.rerank and self._reranker is not None
        return EngineReport(
            requested=call.mode,
            used=engine_name,
            query_plan=query_plan,
            degraded=any(not report.ok for report in reports) or failure is not None,
            children=reports,
            reranked=wanted_rerank and failure is None,
            rerank_error=failure,
        )

    def _require_llm(self) -> LLM:
        """
        Return the LLM built at startup.

        :raises ServiceNotReadyError: If the backend has not started.
        """
        if self._llm is None:
            raise ServiceNotReadyError("Knowledge graph is not loaded yet.", mode="mix")
        return self._llm
