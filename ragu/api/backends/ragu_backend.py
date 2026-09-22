"""
Adapter over the RAGU engines.
"""

import asyncio
import os
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import Any, get_type_hints

from ragu import (
    ArtifactsExtractorLLM,
    BuilderArguments,
    CachedAsyncOpenAI,
    EmbedderOpenAI,
    Env,
    GlobalSearchEngine,
    KnowledgeGraph,
    LLMOpenAI,
    LocalSearchEngine,
    MixSearchEngine,
    NaiveSearchEngine,
    Settings,
    SimpleChunker,
)
from ragu.api.backends.base import (
    DEFAULT_MIX_ENGINES,
    GraphStats,
    SearchStreamEvent,
    RetrieveOutcome,
    SearchBackend,
    SearchCall,
    SearchOutcome,
)
from ragu.api.config import GraphSpec, ServiceSettings
from ragu.api.errors import (
    BackendExecutionError,
    InvalidRequestError,
    NotFoundError,
    RaguServiceError,
    ServiceNotReadyError,
)
from ragu.api.mapping import extract_sources, to_outcome
from ragu.api.reranking import ForgivingScorer, rerank_failure, reset_rerank_report
from ragu.api.usage import CountingLLM, measure_retrieval
from ragu.api.models import ChildEngineReport, EngineReport, SearchMode
from ragu.common.logger import logger
from ragu.models.embedder import Embedder
from ragu.models.llm import LLM
from ragu.models.scorer import Scorer
from ragu.search_engine.base_engine import BaseEngine
from ragu.search_engine.global_search import GlobalSearchParams
from ragu.search_engine.local_search import LocalParams
from ragu.search_engine.naive_search import NaiveSearchParams
from ragu.search_engine.query_plan import QueryPlanEngine


# The public, annotated fields of the settings singleton. Snapshotting them by
# name keeps the isolation below out of GlobalSettings' internals.
_SETTINGS_FIELDS = tuple(
    name for name in get_type_hints(type(Settings)) if not name.startswith("_")
)


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


@contextmanager
def isolated_settings() -> Iterator[None]:
    """
    Apply changes to the ``Settings`` singleton and roll them back afterwards.

    ``Settings`` is process-global, so one graph's storage folder, language and
    token limits would otherwise leak into the next graph constructed. Every
    per-graph value is read inside the constructors that run in this block —
    ``Index`` reads the storage folder, the embedder reads its token limit — so
    restoring afterwards is enough.
    """
    snapshot = {name: getattr(Settings, name) for name in _SETTINGS_FIELDS}
    storage_folder = Settings.storage_folder
    try:
        yield
    finally:
        for name, value in snapshot.items():
            setattr(Settings, name, value)
        Settings.storage_folder = storage_folder


# Snapshotting is only isolation if the blocks do not overlap, and there is one
# singleton for the whole process. The registry serializes startup, but a build
# and a reindex are background jobs against whatever graph a client names: two
# of those on different graphs would each snapshot the other's folder, build
# into it, and leave the singleton restored to the wrong one.
_settings_lock = asyncio.Lock()


@asynccontextmanager
async def exclusive_settings() -> AsyncIterator[None]:
    """
    Hold the ``Settings`` singleton for one graph, then roll it back.

    The async counterpart of :func:`isolated_settings`, and the one every path
    that can run concurrently with another graph's must use. Blocks take minutes
    to hours, so a waiting job waits that long — which is the point: they cannot
    safely run at the same time.
    """
    async with _settings_lock:
        with isolated_settings():
            yield


def _select_relations(
    edges: list[Any],
    wanted: set[str],
    edge_scope: str,
    min_strength: float | None,
) -> list[Any]:
    """
    Relations whose ends fall inside a set of entities.

    :param edges: Every relation in the graph.
    :param wanted: Entity ids the selection is restricted to.
    :param edge_scope: ``induced`` for both ends inside, ``incident`` for either.
    :param min_strength: Keep only relations at least this strong.
    :return: The matching relations, in storage order.
    """
    both = edge_scope == "induced"
    kept = []
    for edge in edges:
        inside = (edge.subject_id in wanted, edge.object_id in wanted)
        if not (all(inside) if both else any(inside)):
            continue
        if min_strength is not None:
            if float(getattr(edge, "relation_strength", 1.0)) < min_strength:
                continue
        kept.append(edge)
    return kept


def _count_degrees(edges: list[Any]) -> dict[str, int]:
    """
    How many relations touch each entity id.

    :param edges: Every relation in the graph.
    :return: Entity id to the number of relations touching it.
    """
    degrees: dict[str, int] = {}
    for edge in edges:
        for side in (edge.subject_id, edge.object_id):
            degrees[side] = degrees.get(side, 0) + 1
    return degrees


def _select_entities(
    nodes: list[Any],
    entity_type: str | None,
    search: str | None,
    community_id: str | None,
    sort: str | None,
    order: str,
    degrees: dict[str, int] | None,
) -> list[Any]:
    """
    Filter and order the whole entity list.

    Pure CPU over a materialized list, kept in one function so the caller pays
    a single hop off the event loop rather than one per step.

    :param nodes: Every entity in the graph.
    :param entity_type: Keep only this exact type.
    :param search: Keep only names containing this substring.
    :param community_id: Keep only members of this community.
    :param sort: ``degree``, ``name`` or ``None`` for storage order.
    :param order: ``asc`` or ``desc``.
    :param degrees: Degree map, required when sorting by degree.
    :return: The filtered, ordered list.
    """
    if entity_type:
        wanted = entity_type.casefold()
        nodes = [n for n in nodes if (n.entity_type or "").casefold() == wanted]
    if search:
        needle = search.casefold()
        nodes = [n for n in nodes if needle in (n.entity_name or "").casefold()]
    if community_id is not None:
        nodes = [n for n in nodes if community_id in _cluster_ids(n)]

    if sort == "name":
        nodes = sorted(
            nodes, key=lambda n: (n.entity_name or "").casefold(), reverse=order == "desc"
        )
    elif sort == "degree":
        counts = degrees or {}
        nodes = sorted(nodes, key=lambda n: counts.get(n.id, 0), reverse=order == "desc")
    return nodes


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


def _cluster_ids(entity: Any) -> set[str]:
    """
    The community ids an entity belongs to, as the wire spells them.
    """
    return {
        str(cluster.get("cluster_id"))
        for cluster in (getattr(entity, "clusters", None) or [])
        if cluster.get("cluster_id") is not None
    }


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


class RaguBackend(SearchBackend):
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
        self._node_cache: list[Any] | None = None
        self._edge_cache: list[Any] | None = None
        self._degree_cache: dict[str, int] | None = None
        self.graph: KnowledgeGraph | None = None
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

    def _pipeline(self, llm: LLM, embedder: Embedder) -> dict[str, Any]:
        """
        The chunker, extractor and builder settings this graph is built with.

        A graph that only serves gets none of them: constructing an extractor
        costs nothing at rest, but it is configuration the deployment has not
        asked for.

        :param llm: LLM the extractor uses.
        :param embedder: Embedder the extractor may use for ICL examples.
        :return: Keyword arguments for ``KnowledgeGraph``.
        """
        build = self.spec.build
        if not build.enabled:
            return {}

        chunker = (
            SimpleChunker(
                max_chunk_size=build.chunk_size, overlap=build.chunk_overlap
            )
            if build.chunker == "simple"
            else None
        )
        extractor = (
            None if build.vector_only else ArtifactsExtractorLLM(llm, embedder=embedder)
        )
        return {
            "chunker": chunker,
            "artifact_extractor": extractor,
            "builder_settings": BuilderArguments(
                build_only_vector_context=build.vector_only,
                make_community_summary=build.make_community_summary,
            ),
        }

    @property
    def accepts_documents(self) -> bool:
        return self.spec.build.enabled

    async def build(self, documents: list[str]) -> dict[str, Any]:
        """
        Add documents to this graph and re-measure it.

        Searches on this graph are refused while the build runs: it writes into
        the same stores they read, and the file-backed ones tolerate no
        concurrent access.

        :param documents: Raw document texts.
        :return: The graph sizes after the build.
        :raises InvalidRequestError: If this graph does not accept documents.
        """
        if not self.accepts_documents:
            return await super().build(documents)

        graph, _, _ = self._require_loaded()
        async with self._write_lock:
            self._building = True
            try:
                async with exclusive_settings():
                    if self.spec.settings_file:
                        Settings.load(self.spec.settings_file)
                    Settings.storage_folder = self.spec.storage_folder
                    Settings.language = self.language
                    await graph.build_from_docs(documents)
                self._drop_graph_cache()
                self._stats = await self._measure(graph)
            finally:
                self._building = False

        logger.info(
            "Graph '{}' rebuilt from {} document(s): {}",
            self.graph_id,
            len(documents),
            self._stats,
        )
        return {"documents": len(documents), "stats": self._stats.to_response().model_dump()}

    # --- the graph surface ---------------------------------------------------

    def _worth_caching(self, items: list[Any]) -> bool:
        """
        Whether a materialized list is small enough to keep.

        The cache trades memory for paging speed, and on a graph of a few hundred
        thousand relations that trade is a permanent floor under the process for
        the benefit of one endpoint. Above the ceiling the list is rebuilt per
        page instead — slower, but the service survives the request.
        """
        limit = self.settings.graph_cache_max_items
        return limit > 0 and len(items) <= limit

    async def _nodes(self) -> list[Any]:
        """
        Every entity, materialized once and kept while it fits.

        ``get_all_nodes`` rebuilds an ``Entity`` per node on each call, so a
        client paging through a graph would otherwise pay O(n) per page. The
        list is dropped whenever the graph is written to, and not kept at all
        above ``graph_cache_max_items``.
        """
        if self._node_cache is not None:
            return self._node_cache
        graph, _, _ = self._require_loaded()
        nodes = await graph.index.graph_backend.get_all_nodes()
        if self._worth_caching(nodes):
            self._node_cache = nodes
        return nodes

    async def _edges(self) -> list[Any]:
        """
        Every relation, materialized once and kept while it fits. See :meth:`_nodes`.
        """
        if self._edge_cache is not None:
            return self._edge_cache
        graph, _, _ = self._require_loaded()
        edges = await graph.index.graph_backend.get_all_edges()
        if self._worth_caching(edges):
            self._edge_cache = edges
        return edges

    def _drop_graph_cache(self) -> None:
        self._node_cache = None
        self._edge_cache = None
        self._degree_cache = None

    async def graph_detail(self) -> dict[str, Any]:
        graph, _, embedder = self._require_loaded()
        index = graph.index
        communities = await index.community_kv_storage.all_keys()
        chunks = await index.chunks_kv_storage.get_by_ids(
            await index.chunks_kv_storage.all_keys()
        )
        documents = {
            chunk.get("doc_id") if isinstance(chunk, dict) else getattr(chunk, "doc_id", None)
            for chunk in chunks
            if chunk is not None
        }
        stats = self._stats or GraphStats()
        created_at, updated_at = _folder_timestamps(self.spec.storage_folder)
        return {
            "entities": stats.entities,
            "relations": stats.relations,
            "chunks": stats.chunks,
            "communities": len(communities),
            "community_summaries": stats.community_summaries,
            "documents": len({doc for doc in documents if doc}),
            "embedding_dim": getattr(embedder, "dim", None),
            "created_at": created_at,
            "updated_at": updated_at,
        }

    async def list_entities(
        self,
        *,
        limit: int,
        offset: int,
        entity_type: str | None = None,
        search: str | None = None,
        community_id: str | None = None,
        sort: str | None = None,
        order: str = "asc",
        ids: Sequence[str] | None = None,
    ) -> tuple[int, list[Any]]:
        self.require_idle()
        if ids is not None:
            found = await self._entities_by_id(ids)
            return len(found), found

        if sort not in (None, "degree", "name"):
            raise InvalidRequestError(
                f"Unknown sort '{sort}'. Expected 'degree' or 'name'."
            )

        nodes = await self._nodes()
        if not (entity_type or search or community_id is not None or sort):
            # Plain paging touches nothing, so it does not pay for a thread.
            return len(nodes), nodes[offset : offset + limit]

        degrees = await self._degrees() if sort == "degree" else None
        # Filtering and sorting walk the whole corpus: on a graph of tens of
        # thousands of entities that is a tenth of a second of pure CPU, and on
        # the event loop it is a tenth of a second in which the service answers
        # nothing at all — /health included. One hop off the loop, not one per
        # step, because each hop costs a context switch.
        nodes = await asyncio.to_thread(
            _select_entities, nodes, entity_type, search, community_id, sort, order, degrees
        )
        return len(nodes), nodes[offset : offset + limit]

    async def _entities_by_id(self, ids: Sequence[str]) -> list[Any]:
        """
        Exactly these entities, in the order asked for, skipping the unknown.
        """
        graph, _, _ = self._require_loaded()
        wanted = list(dict.fromkeys(ids))
        found = await graph.index.graph_backend.get_nodes(wanted)
        return [node for node in found if node is not None]

    async def _degrees(self) -> dict[str, int]:
        """
        How many relations touch each entity.

        Counted from the edge list rather than asked of the storage: the adapter
        contract exposes degree per *edge*, not per node, and the edges are
        materialized anyway for the relation routes. The count itself runs off
        the event loop — it is the single most expensive pass in this class.
        """
        if self._degree_cache is None:
            edges = await self._edges()
            self._degree_cache = await asyncio.to_thread(_count_degrees, edges)
        return self._degree_cache

    async def get_entity(self, entity_id: str) -> Any:
        self.require_idle()
        graph, _, _ = self._require_loaded()
        found = await graph.index.graph_backend.get_nodes([entity_id])
        if not found or found[0] is None:
            raise NotFoundError(f"No entity with id '{entity_id}' in this graph.")
        return found[0]

    async def list_chunks(
        self, *, limit: int, offset: int, ids: Sequence[str] | None = None
    ) -> tuple[int, list[Any]]:
        self.require_idle()
        graph, _, _ = self._require_loaded()
        if ids is not None:
            wanted = list(dict.fromkeys(ids))
            found = await graph.get_chunks(wanted)
            kept = [chunk for chunk in found if chunk is not None]
            return len(kept), kept

        keys = sorted(await graph.index.chunks_kv_storage.all_keys())
        page = keys[offset : offset + limit]
        found = await graph.get_chunks(page)
        return len(keys), [chunk for chunk in found if chunk is not None]

    async def list_relations(
        self, *, limit: int, offset: int, min_strength: float | None = None
    ) -> tuple[int, list[Any]]:
        self.require_idle()
        edges = await self._edges()
        if min_strength is not None:
            edges = [
                e for e in edges if float(getattr(e, "relation_strength", 1.0)) >= min_strength
            ]
        return len(edges), edges[offset : offset + limit]

    async def select_relations(
        self,
        *,
        entity_ids: Sequence[str],
        edge_scope: str = "induced",
        min_strength: float | None = None,
        limit: int,
        offset: int,
    ) -> tuple[int, list[Any]]:
        self.require_idle()
        edges = await self._edges()
        # A membership test per relation over the whole corpus: pure CPU, so it
        # runs off the event loop like the other whole-graph passes.
        kept = await asyncio.to_thread(
            _select_relations, edges, set(entity_ids), edge_scope, min_strength
        )
        return len(kept), kept[offset : offset + limit]

    async def neighbors(self, entity_id: str, depth: int, limit: int) -> dict[str, Any]:
        self.require_idle()
        graph, _, _ = self._require_loaded()
        backend = graph.index.graph_backend

        found = await backend.get_nodes([entity_id])
        if not found or found[0] is None:
            raise NotFoundError(f"No entity with id '{entity_id}' in this graph.")

        seen = {entity_id: found[0]}
        frontier = [entity_id]
        relations: dict[str, Any] = {}
        truncated = False

        for _ in range(max(depth, 0)):
            if not frontier:
                break
            grouped = await backend.get_all_edges_for_nodes(frontier)
            next_frontier: list[str] = []
            for edges in grouped:
                for edge in edges or []:
                    relations[edge.id] = edge
                    for side in (edge.subject_id, edge.object_id):
                        if side not in seen:
                            next_frontier.append(side)
            if not next_frontier:
                break
            if len(seen) + len(set(next_frontier)) > limit:
                truncated = True
                next_frontier = list(dict.fromkeys(next_frontier))[: limit - len(seen)]
            fetched = await backend.get_nodes(list(dict.fromkeys(next_frontier)))
            for node_id, node in zip(dict.fromkeys(next_frontier), fetched):
                if node is not None:
                    seen[node_id] = node
            frontier = [node_id for node_id in dict.fromkeys(next_frontier) if node_id in seen]
            if truncated:
                break

        return {
            "entities": list(seen.values()),
            "relations": list(relations.values()),
            "truncated": truncated,
        }

    async def list_communities(
        self,
        *,
        limit: int,
        offset: int,
        level: int | None = None,
        ids: Sequence[str] | None = None,
    ) -> tuple[int, list[Any]]:
        self.require_idle()
        graph, _, _ = self._require_loaded()
        if ids is not None:
            wanted = list(dict.fromkeys(ids))
            communities = [
                c for c in await graph.get_communities(wanted) if c is not None
            ]
            page = communities
        else:
            keys = sorted(await graph.index.community_kv_storage.all_keys())
            communities = [
                c for c in await graph.get_communities(keys) if c is not None
            ]
            if level is not None:
                communities = [c for c in communities if c.level == level]
            page = communities[offset : offset + limit]
        summaries = await graph.index.community_summary_kv_storage.get_by_ids(
            [c.id for c in page]
        )
        return len(communities), list(zip(page, summaries))

    async def get_community(self, community_id: str) -> tuple[Any, Any]:
        self.require_idle()
        graph, _, _ = self._require_loaded()
        found = await graph.get_communities([community_id])
        if not found or found[0] is None:
            raise NotFoundError(f"No community with id '{community_id}' in this graph.")
        summary = await graph.index.community_summary_kv_storage.get_by_id(community_id)
        return found[0], summary

    async def get_chunk(self, chunk_id: str) -> Any:
        self.require_idle()
        graph, _, _ = self._require_loaded()
        found = await graph.get_chunks([chunk_id])
        if not found or found[0] is None:
            raise NotFoundError(f"No chunk with id '{chunk_id}' in this graph.")
        return found[0]

    async def consistency(self) -> Any:
        self.require_idle()
        graph, _, _ = self._require_loaded()
        return await graph.index.check_consistency()

    async def reindex(self, kind: str) -> dict[str, Any]:
        """
        Rebuild communities, descriptions or the whole graph.

        Held under the same write lock as ingestion, and it invalidates the
        materialized node and edge lists.
        """
        graph, _, _ = self._require_loaded()
        operations = {
            "community": graph.reindex_community,
            "descriptions": graph.reindex_descriptions,
            "graph": graph.reindex_graph,
        }
        operation = operations.get(kind)
        if operation is None:
            raise InvalidRequestError(
                f"Unknown reindex '{kind}'. Expected one of {sorted(operations)}."
            )

        async with self._write_lock:
            self._building = True
            try:
                async with exclusive_settings():
                    if self.spec.settings_file:
                        Settings.load(self.spec.settings_file)
                    Settings.storage_folder = self.spec.storage_folder
                    Settings.language = self.language
                    await operation()
                self._drop_graph_cache()
                self._stats = await self._measure(graph)
            finally:
                self._building = False

        return {"reindex": kind, "stats": self._stats.to_response().model_dump()}

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

    def _require_loaded(self) -> tuple[KnowledgeGraph, LLM, Embedder]:
        """
        Return the graph and models built at startup.

        :raises ServiceNotReadyError: If the backend has not started.
        """
        if self.graph is None or self._llm is None or self._embedder is None:
            raise ServiceNotReadyError("Knowledge graph is not loaded yet.")
        return self.graph, self._llm, self._embedder

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

    def _require_llm(self) -> LLM:
        """
        Return the LLM built at startup.

        :raises ServiceNotReadyError: If the backend has not started.
        """
        if self._llm is None:
            raise ServiceNotReadyError("Knowledge graph is not loaded yet.", mode="mix")
        return self._llm
