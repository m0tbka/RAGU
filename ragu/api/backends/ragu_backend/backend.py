"""
The real backend: a prebuilt RAGU graph, loaded once and served by the engines.

This module holds the lifecycle and the search entry points. The rest of the
backend lives beside it, one concern per module: reading the graph in
``graph_view``, building engines in ``engines``, writing to the graph in
``ingest``, and sharing the process-wide settings in ``settings``.
"""

import asyncio
import os
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from ragu import (
    CachedAsyncOpenAI,
    EmbedderOpenAI,
    Env,
    KnowledgeGraph,
    LLMOpenAI,
    Settings,
)
from ragu.api.backends.base import (
    RetrieveOutcome,
    SearchBackend,
    SearchCall,
    SearchOutcome,
    SearchStreamEvent,
)
from ragu.api.backends.capabilities import GraphStats
from ragu.api.backends.ragu_backend.engines import EngineAssembly
from ragu.api.backends.ragu_backend.graph_view import GraphView
from ragu.api.backends.ragu_backend.ingest import Ingestion
from ragu.api.backends.ragu_backend.settings import exclusive_settings
from ragu.api.config import GraphSpec, ServiceSettings
from ragu.api.errors import (
    BackendExecutionError,
    BudgetExceededError,
    RaguServiceError,
    ServiceNotReadyError,
)
from ragu.api.models import SearchMode
from ragu.api.search.mapping import extract_sources, to_outcome
from ragu.api.search.reranking import ForgivingScorer
from ragu.api.search.usage import CountingLLM, measure_retrieval, remaining_calls
from ragu.common.logger import logger
from ragu.models.embedder import Embedder
from ragu.models.llm import LLM
from ragu.models.scorer import Scorer
from ragu.search_engine.base_engine import BaseEngine
from ragu.search_engine.params import GlobalSearchParams
from ragu.search_engine.query_plan import QueryPlanEngine


def _encoder():
    """
    The tokenizer used to size prompts and answers, or ``None`` if unavailable.

    Counting is a convenience, not a contract, so a missing tokenizer costs the
    numbers rather than the request.
    """
    try:
        import tiktoken

        return tiktoken.encoding_for_model(Settings.tokenizer_llm_name)
    except Exception:
        return None


def _folder_timestamps(folder: str) -> tuple[datetime | None, datetime | None]:
    """
    When a graph's storage folder was created and last written.

    The graph records no build date of its own, so the filesystem is the only
    witness. Modification times exist everywhere; creation time does not —
    ``os.stat`` reports it on Windows and macOS but not on Linux, which is where
    the service usually runs. A missing value is returned as ``None`` rather
    than guessed: the oldest modification time is not a creation time, since an
    in-place rebuild rewrites every file.

    :param folder: The graph's storage folder.
    :return: ``(created_at, updated_at)`` in UTC; either may be ``None``.
    """
    try:
        files = [entry for entry in os.scandir(folder) if entry.is_file()]
        born = getattr(os.stat(folder), "st_birthtime", None)
    except OSError:
        return None, None

    def moment(timestamp: float | None) -> datetime | None:
        if timestamp is None:
            return None
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    updated = max((entry.stat().st_mtime for entry in files), default=None)
    return moment(born), moment(updated)


class RaguBackend(EngineAssembly, Ingestion, SearchBackend):
    """
    Real backend: a loaded graph plus one engine per search mode.
    """

    def __init__(
        self,
        settings: ServiceSettings,
        spec: GraphSpec,
        reranker: Scorer | None = None,
    ):
        super().__init__(
            settings, graph_id=spec.id, language=spec.language or settings.language
        )
        self.spec = spec
        # Supplied from outside: the model runs in its own process on the
        # consumer's deployment, so the service wraps one rather than making one.
        self._reranker = (
            ForgivingScorer(reranker, settings.rerank_timeout)
            if reranker is not None
            else None
        )
        # Token limits and tokenizer names are read off the Settings singleton by
        # the engine constructors. They are captured here, while this graph's
        # settings are applied, and passed explicitly afterwards so an engine
        # built lazily for another language does not pick up whatever the
        # singleton holds by then.
        self._engine_kwargs: dict[str, Any] = {}
        self.graph: KnowledgeGraph | None = None
        self._view: GraphView | None = None
        self._engines: dict[
            tuple[SearchMode, str, bool], BaseEngine[Any, Any]
        ] = {}
        self._clients: list[CachedAsyncOpenAI] = []
        self._llm: LLM | None = None
        self._embedder: Embedder | None = None

    async def startup(self) -> None:
        """
        Load this graph and warm its default-language engines.

        :raises ServiceNotReadyError: If credentials, the embedder endpoint or
            the storage folder make the graph unusable.
        """
        settings = self.settings
        spec = self.spec

        self._require_storage_folder(spec.storage_folder)
        env = self._env()

        llm_client = CachedAsyncOpenAI(
            base_url=env.llm_base_url,
            api_key=env.llm_api_key,
            rate_min_delay=settings.rate_min_delay,
            rate_max_simultaneous=settings.rate_max_simultaneous,
            cache=settings.llm_cache,
        )
        self._clients.append(llm_client)

        embedder_client = llm_client
        if env.embedder_base_url:
            embedder_client = CachedAsyncOpenAI(
                base_url=env.embedder_base_url,
                api_key=env.embedder_api_key or env.llm_api_key,
            )
            self._clients.append(embedder_client)

        async with exclusive_settings():
            if spec.settings_file:
                Settings.load(spec.settings_file)
            Settings.storage_folder = spec.storage_folder
            Settings.language = self.language

            # Wrapped once, here: every engine takes this object, so every
            # call they make is accounted for without threading a counter
            # through the engines.
            llm = CountingLLM(
                LLMOpenAI(client=llm_client, model_name=env.llm_model_name),
                encoder=_encoder(),
            )
            embedder = EmbedderOpenAI(
                client=embedder_client,
                model_name=env.embedder_model_name or env.llm_model_name,
                dim=spec.embedder_dim,
            )
            if spec.embedder_dim is None:
                try:
                    await embedder.initialize()
                except Exception as exc:
                    raise ServiceNotReadyError(
                        "Could not detect the embedding dimension: the embedder endpoint is "
                        "unreachable. Set the graph's embedder_dim to the dimension it was "
                        f"built with, or make the endpoint reachable ({exc})."
                    ) from exc

            try:
                # Constructing the graph opens every storage and reads the whole
                # graph file, which takes minutes on a large corpus. Off the
                # event loop, so /health keeps answering while it happens.
                graph = await asyncio.to_thread(
                    KnowledgeGraph,
                    llm=llm,
                    embedder=embedder,
                    language=self.language,
                    **self._pipeline(llm, embedder),
                )
            except Exception as exc:
                raise ServiceNotReadyError(
                    f"Failed to load graph '{spec.id}' from '{spec.storage_folder}': {exc}"
                ) from exc

            self._engine_kwargs = {
                "max_context_length": Settings.llm_context_token_limit,
                "tokenizer_backend": Settings.tokenizer_llm_backend,
                "tokenizer_model": Settings.tokenizer_llm_name,
            }
            self.graph = graph
            self._llm = llm
            self._embedder = embedder
            # Warm the default language so a misconfigured engine fails at
            # startup rather than on the first request.
            for mode in ("global", "local", "naive"):
                self._leaf_engine(mode, self.language)

        self._stats = await self._measure(graph)

        if self._stats.is_empty:
            raise ServiceNotReadyError(
                f"Graph '{spec.id}' at '{spec.storage_folder}' is empty: no entities, "
                "chunks or community summaries."
            )

        logger.info(
            "Graph '{}' loaded from '{}' (language={}): {}",
            spec.id,
            spec.storage_folder,
            self.language,
            self._stats,
        )

    @staticmethod
    def _env() -> Env:
        """
        Read the model credentials shared by every graph.

        :raises ServiceNotReadyError: If they are missing.
        """
        try:
            return Env.from_env()
        except Exception as exc:
            raise ServiceNotReadyError(
                "LLM credentials are missing: set LLM_MODEL_NAME, LLM_BASE_URL and "
                f"LLM_API_KEY in the environment or in .env ({exc})."
            ) from exc

    async def shutdown(self) -> None:
        """
        Close the graph storages and the model clients.

        File-backed storages do nothing here, but server-backed ones (Neo4j,
        remote Qdrant) and the HTTP pool behind every model client stay open
        until closed.
        """
        if self.graph is not None:
            try:
                await self.graph.index.close()
            except Exception:
                logger.opt(exception=True).error("Failed to close the graph index")
        for client in self._clients:
            try:
                await client.client.close()
            except Exception:
                logger.opt(exception=True).error("Failed to close a model client")
        self._clients.clear()
        self.graph = None
        self._view = None
        self._llm = None
        self._embedder = None
        self._engines = {}
        self._stats = None

    @staticmethod
    def _require_storage_folder(storage_folder: str) -> None:
        """
        Reject a storage folder that holds no graph.

        ``Index.__init__`` calls ``Settings.init_storage_folder()``, which
        *creates* the folder when it is missing. Without this check a typo in
        ``RAGU_API_STORAGE_FOLDER`` produces an empty directory and a backend
        that looks healthy while every search comes back empty.

        :param storage_folder: Folder the service was pointed at.
        :raises ServiceNotReadyError: If it is missing, not a directory, or empty.
        """
        if not os.path.isdir(storage_folder):
            raise ServiceNotReadyError(
                f"Storage folder '{storage_folder}' does not exist. Set "
                "RAGU_API_STORAGE_FOLDER to the folder a build run produced; the service "
                "serves a prebuilt graph and never builds one."
            )
        if not os.listdir(storage_folder):
            raise ServiceNotReadyError(
                f"Storage folder '{storage_folder}' is empty. Set RAGU_API_STORAGE_FOLDER "
                "to the folder a build run produced."
            )

    @staticmethod
    async def _measure(graph: KnowledgeGraph) -> GraphStats:
        """
        Count what each search mode has to read.

        Vector-store ids are used rather than materialized entities: the counts
        only decide which modes can run, and listing ids stays cheap on a large
        graph.

        :param graph: The loaded graph.
        :return: Sizes of the stores behind the three search modes.
        """
        index = graph.index
        entities, relations, chunks, summaries = await asyncio.gather(
            index.nodes_vector_db.get_all_ids(),
            index.edges_vector_db.get_all_ids(),
            index.chunks_kv_storage.all_keys(),
            index.community_summary_kv_storage.all_keys(),
        )
        return GraphStats(
            entities=len(entities),
            relations=len(relations),
            chunks=len(chunks),
            community_summaries=len(summaries),
        )

    def _require_loaded(self) -> tuple[KnowledgeGraph, LLM, Embedder]:
        """
        Return the graph and models built at startup.

        :raises ServiceNotReadyError: If the backend has not started.
        """
        if self.graph is None or self._llm is None or self._embedder is None:
            raise ServiceNotReadyError("Knowledge graph is not loaded yet.")
        return self.graph, self._llm, self._embedder

    # --- the graph surface ---------------------------------------------------
    #
    # Every read belongs to GraphView. What stays here is the part that is about
    # this backend rather than the graph: a read is refused while a build or a
    # reindex is writing, and there is nothing to read before startup.

    @property
    def view(self) -> GraphView:
        """
        Reads over the loaded graph, with the lists they page through.

        Built on first use and kept for as long as the graph object is the same
        one, so its cache lives exactly as long as the data it was built from.

        :raises ServiceNotReadyError: If the backend has not started.
        """
        graph, _, _ = self._require_loaded()
        if self._view is None or self._view.graph is not graph:
            self._view = GraphView(graph, self.settings.graph_cache_max_items)
        return self._view

    def _drop_graph_cache(self) -> None:
        """
        Forget the materialized lists after a write changed the graph.
        """
        if self._view is not None:
            self._view.drop_cache()

    async def graph_detail(self) -> dict[str, Any]:
        _, _, embedder = self._require_loaded()
        counts = await self.view.counts()
        stats = self._stats or GraphStats()
        created_at, updated_at = _folder_timestamps(self.spec.storage_folder)
        return {
            "entities": stats.entities,
            "relations": stats.relations,
            "chunks": stats.chunks,
            "communities": counts["communities"],
            "community_summaries": stats.community_summaries,
            "documents": counts["documents"],
            "embedding_dim": getattr(embedder, "dim", None),
            "created_at": created_at,
            "updated_at": updated_at,
        }

    async def list_entities(self, **query: Any) -> tuple[int, list[Any]]:
        self.require_idle()
        return await self.view.list_entities(**query)

    async def get_entity(self, entity_id: str) -> Any:
        self.require_idle()
        return await self.view.get_entity(entity_id)

    async def list_chunks(self, **query: Any) -> tuple[int, list[Any]]:
        self.require_idle()
        return await self.view.list_chunks(**query)

    async def list_relations(self, **query: Any) -> tuple[int, list[Any]]:
        self.require_idle()
        return await self.view.list_relations(**query)

    async def select_relations(self, **query: Any) -> tuple[int, list[Any]]:
        self.require_idle()
        return await self.view.select_relations(**query)

    async def neighbors(self, entity_id: str, depth: int, limit: int) -> dict[str, Any]:
        self.require_idle()
        return await self.view.neighbors(entity_id, depth, limit)

    async def list_communities(self, **query: Any) -> tuple[int, list[Any]]:
        self.require_idle()
        return await self.view.list_communities(**query)

    async def get_community(self, community_id: str) -> tuple[Any, Any]:
        self.require_idle()
        return await self.view.get_community(community_id)

    async def get_chunk(self, chunk_id: str) -> Any:
        self.require_idle()
        return await self.view.get_chunk(chunk_id)

    async def consistency(self) -> Any:
        self.require_idle()
        return await self.view.consistency()

    # --- searching ------------------------------------------------------------

    async def require_budget(self, call: SearchCall, *, generate: bool = True) -> None:
        """
        Refuse a global search the budget cannot cover, before its first call.

        Global search is the mode whose cost is both large and known up front: it
        rates every community that survives ``min_cluster_size`` against every
        query before it writes a word. Counting them first makes an over-budget
        request a free 429. The per-call check alone would refuse a rating pass
        that does not fit, but one that just fits would be paid for in full and
        the answer after it refused. In ``mix`` only the global child's rating
        pass is counted — a lower bound on what the ensemble spends, which is all
        a refusal needs. With no budget configured nothing is counted at all.
        """
        remaining = remaining_calls()
        if remaining is None:
            return
        if call.mode == "global":
            params = self.bound_params(call.params) if call.params is not None else None
            who, remedy = "Global search", "raise min_cluster_size or the budget"
        elif call.mode == "mix" and "global" in call.mix_engines:
            params = self._child_params("global", call)
            # The ensemble reads its children's context; it writes the answer.
            generate = False
            who = "The global child of this ensemble"
            remedy = (
                "raise global_params.min_cluster_size, drop global from engines, or "
                "raise the budget"
            )
        else:
            return

        self.require_idle()
        rerank = call.rerank and self._reranker is not None
        engine = self._leaf_engine("global", call.language or self.language, rerank)
        planned_calls = getattr(engine, "planned_calls", None)
        if planned_calls is None:
            return
        planned = await planned_calls(list(call.queries), params, generate=generate)
        if planned <= remaining:
            return

        min_cluster_size = getattr(
            params, "min_cluster_size", GlobalSearchParams().min_cluster_size
        )
        answers = ", plus one answer per query" if generate else ""
        raise BudgetExceededError(
            f"{who} would make {planned} LLM calls here — one for every community "
            f"that survives min_cluster_size={min_cluster_size}, for every "
            f"query{answers} — and this request may make {remaining} more. Refused "
            f"before the first one: {remedy}.",
            mode=call.mode,
        )

    async def stream(self, call: SearchCall) -> AsyncIterator[SearchStreamEvent]:
        engine, children = self._engine_for(call)
        engine_name = type(engine).__name__
        params = self.bound_params(call.params) if call.params is not None else None
        if call.use_query_plan:
            engine = QueryPlanEngine(engine)

        started = False
        try:
            async for event in engine.stream_query(call.query, params):
                if not started:
                    started = True
                    yield SearchStreamEvent(
                        "meta",
                        {
                            "query": call.query,
                            "mode": call.mode,
                            "sources": [
                                source.model_dump()
                                for source in extract_sources(event.retrieval)
                            ],
                            "engines": self._report(
                                call,
                                engine_name,
                                children,
                                query_plan=call.use_query_plan,
                            ).model_dump(),
                        },
                    )
                if event.delta:
                    yield SearchStreamEvent("delta", {"text": event.delta})
        except RaguServiceError as exc:
            # A refusal that means something — a budget, a busy graph — keeps its
            # code: inside an open stream this event is all the client gets.
            yield SearchStreamEvent("error", exc.to_envelope()["error"])
            return
        except Exception:
            logger.opt(exception=True).error("RAGU {} stream failed", call.mode)
            yield SearchStreamEvent(
                "error",
                {"code": "INTERNAL_ERROR", "message": f"The {call.mode} search engine failed."},
            )
            return

        yield SearchStreamEvent(
            "done",
            {
                "engines": self._report(
                    call, engine_name, children, query_plan=call.use_query_plan
                ).model_dump()
            },
        )

    async def search(self, call: SearchCall) -> list[SearchOutcome]:
        engine, children = self._engine_for(call)
        await self.require_budget(call)
        engine_name = type(engine).__name__
        params = self.bound_params(call.params) if call.params is not None else None

        try:
            if call.use_query_plan:
                engine = QueryPlanEngine(engine)
            with measure_retrieval(call.mode):
                responses = await engine.batch_query(list(call.queries), params)
        except RaguServiceError:
            # A budget or capability refusal already carries its own status.
            raise
        except Exception as exc:
            logger.opt(exception=True).error("RAGU {} search failed", call.mode)
            raise BackendExecutionError(mode=call.mode, detail=str(exc)) from exc

        report = self._report(
            call, engine_name, children, query_plan=call.use_query_plan
        )
        outcomes = []
        for response in responses:
            answer, sources, subqueries = to_outcome(
                response, used_query_plan=call.use_query_plan
            )
            outcomes.append(
                SearchOutcome(
                    answer=answer,
                    sources=sources,
                    subqueries=subqueries,
                    engines=report,
                )
            )
        return outcomes

    async def retrieve(self, call: SearchCall) -> list[RetrieveOutcome]:
        engine, children = self._engine_for(call)
        await self.require_budget(call, generate=False)
        params = self.bound_params(call.params) if call.params is not None else None

        try:
            # No QueryPlanEngine here: its batch_search delegates straight to the
            # wrapped engine, so wrapping would only imply planning that does not
            # happen.
            with measure_retrieval(call.mode):
                retrievals = await engine.batch_search(list(call.queries), params)
        except RaguServiceError:
            raise
        except Exception as exc:
            logger.opt(exception=True).error("RAGU {} retrieval failed", call.mode)
            raise BackendExecutionError(mode=call.mode, detail=str(exc)) from exc

        report = self._report(call, type(engine).__name__, children, query_plan=False)
        return [
            RetrieveOutcome(sources=extract_sources(retrieval), engines=report)
            for retrieval in retrievals
        ]
