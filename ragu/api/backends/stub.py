"""
In-memory backend for local development and tests.

Serves deterministic canned answers without a real graph, so the agent side can
be exercised end-to-end without building a knowledge graph.
"""

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

from ragu.api.backends.base import (
    GraphStats,
    SearchStreamEvent,
    RetrieveOutcome,
    SearchBackend,
    SearchCall,
    SearchOutcome,
)
from ragu.api.config import DEFAULT_GRAPH_ID, GraphSpec, ServiceSettings
from ragu.api.errors import InvalidRequestError, NotFoundError
from ragu.api.models import (
    ChildEngineReport,
    ChunkMeta,
    CommunityMeta,
    EngineReport,
    EntityMeta,
    SourceItem,
    SubqueryItem,
)
from ragu.api.mapping import _deduplicated
from ragu.common.logger import logger
from ragu.search_engine.global_search import GlobalSearchParams
from ragu.search_engine.local_search import LocalParams
from ragu.search_engine.naive_search import NaiveSearchParams

# How a simulated missing capability shows up in the graph sizes the base class
# reads. Driving the stub through the same GraphStats the real backend measures
# keeps both backends on one capability code path instead of two.
_EMPTY_STORE_FOR_CAPABILITY = {
    "community_summaries": "community_summaries",
    "entity_graph": "entities",
    "vector_index": "chunks",
}

_FULL_STATS = GraphStats(
    entities=1, relations=1, chunks=1, community_summaries=1
)


_CHILD_ENGINE = {
    "local": "LocalSearchEngine",
    "naive": "NaiveSearchEngine",
    "global": "GlobalSearchEngine",
}


def _stub_entity_source() -> SourceItem:
    return SourceItem(
        id="entity_1",
        type="entity",
        content="stub entity",
        meta=EntityMeta(
            name="Сенкевич",
            type="PERSON",
            communities=["0"],
            source_chunk_ids=["chunk_1"],
        ),
    )


def _as_child_call(call: SearchCall, child: str) -> SearchCall:
    """
    The call one mix child would have received on its own.

    Recursing through the same builder keeps the ensemble's evidence identical
    to what each child answers alone, which is what the real engines do.
    """
    params = {
        "local": call.local_params or LocalParams(),
        "naive": call.naive_params or NaiveSearchParams(),
        "global": call.global_params or GlobalSearchParams(),
    }[child]
    return replace(call, mode=child, params=params)


class StubBackend(SearchBackend):
    """Canned backend. Which modes fail is driven by
    ``RAGU_API_STUB_MISSING_CAPABILITIES``."""

    def __init__(
        self,
        settings: ServiceSettings | None = None,
        spec: GraphSpec | None = None,
    ):
        settings = settings or ServiceSettings(backend="stub")
        super().__init__(
            settings,
            graph_id=spec.id if spec else DEFAULT_GRAPH_ID,
            language=spec.language if spec else None,
        )
        self._spec = spec
        self._ingested = 0
        self._missing = self.settings.missing_capabilities()

    @property
    def accepts_documents(self) -> bool:
        return bool(self._spec and self._spec.build.enabled)

    async def build(self, documents: list[str]) -> dict[str, Any]:
        if not self.accepts_documents:
            return await super().build(documents)
        async with self._write_lock:
            self._building = True
            try:
                self._ingested += len(documents)
            finally:
                self._building = False
        return {"documents": len(documents), "total": self._ingested}

    async def startup(self) -> None:
        self._stats = self._simulated_stats()
        logger.warning(
            "StubBackend started: canned answers only, no real knowledge graph"
        )

    def _simulated_stats(self) -> GraphStats:
        """
        Build graph sizes that reproduce the configured missing capabilities.

        :return: Sizes where each simulated-missing store counts zero.
        """
        empty = {
            _EMPTY_STORE_FOR_CAPABILITY[capability]
            for capability in self._missing
            if capability in _EMPTY_STORE_FOR_CAPABILITY
        }
        return GraphStats(
            entities=0 if "entities" in empty else _FULL_STATS.entities,
            relations=_FULL_STATS.relations,
            chunks=0 if "chunks" in empty else _FULL_STATS.chunks,
            community_summaries=(
                0 if "community_summaries" in empty else _FULL_STATS.community_summaries
            ),
        )

    @staticmethod
    def _subqueries(query: str, use_query_plan: bool) -> list[SubqueryItem]:
        if not use_query_plan:
            return []
        return [SubqueryItem(query=query, answer=f"stub answer for '{query}'")]

    def _sources_for(self, call: SearchCall) -> list[SourceItem]:
        """
        Canned retrieval for one mode, shaped by the request parameters.
        """
        if call.mode == "global":
            params = self.bound_params(call.params or GlobalSearchParams())
            return [
                SourceItem(
                    id="community_1",
                    type="community_summary",
                    content=f"stub community summary (min_cluster_size={params.min_cluster_size})",
                    meta=CommunityMeta(title="Сенкевич и Польша"),
                )
            ]

        if call.mode == "local":
            params = self.bound_params(call.params or LocalParams())
            sources = [_stub_entity_source()]
            if params.use_chunks:
                sources.append(
                    SourceItem(
                        id="chunk_1",
                        type="chunk",
                        content="stub chunk",
                        score=0.87,
                        meta=ChunkMeta(doc_id="doc-1", chunk_order_idx=0),
                    )
                )
            if params.use_summary:
                sources.append(
                    SourceItem(
                        id="community_1",
                        type="community_summary",
                        content="stub community summary",
                        meta=CommunityMeta(
                            level=0, cluster_id=0, title="Сенкевич и Польша",
                            entity_count=2,
                        ),
                    )
                )
            return sources

        if call.mode == "mix":
            # Children overlap — local and naive both return chunks — and the
            # real mapping deduplicates the union, so this does too.
            sources: list[SourceItem] = []
            for child in call.mix_engines:
                sources.extend(self._sources_for(_as_child_call(call, child)))
            return _deduplicated(sources)

        params = self.bound_params(call.params or NaiveSearchParams())
        return self._chunks(params.top_k)

    @staticmethod
    def _chunks(top_k: int) -> list[SourceItem]:
        return [
            SourceItem(
                id=f"chunk_{i}",
                type="chunk",
                content=f"stub chunk {i}",
                score=round(0.9 - i / 100, 2),
                meta=ChunkMeta(doc_id="doc-1", chunk_order_idx=i - 1),
            )
            for i in range(1, top_k + 1)
        ]

    def _children_for(self, call: SearchCall) -> list[ChildEngineReport]:
        if call.mode != "mix":
            return []
        return [
            ChildEngineReport(engine=_CHILD_ENGINE[child], mode=child, ok=True)
            for child in call.mix_engines
        ]

    def _report(self, call: SearchCall, *, query_plan: bool) -> EngineReport:
        children = self._children_for(call)
        return EngineReport(
            requested=call.mode,
            used="StubBackend",
            query_plan=query_plan,
            degraded=any(not child.ok for child in children),
            children=children,
        )

    async def stream(self, call: SearchCall) -> AsyncIterator[SearchStreamEvent]:
        self.require_capability(call.mode, call.mix_engines)
        report = self._report(call, query_plan=call.use_query_plan)
        yield SearchStreamEvent(
            "meta",
            {
                "query": call.query,
                "mode": call.mode,
                "sources": [
                    source.model_dump() for source in self._sources_for(call)
                ],
                "engines": report.model_dump(),
            },
        )
        for word in f"[stub {call.mode}] {call.query}".split(" "):
            yield SearchStreamEvent("delta", {"text": word + " "})
        yield SearchStreamEvent("done", {"engines": report.model_dump()})

    async def search(self, call: SearchCall) -> list[SearchOutcome]:
        self.require_idle()
        self.require_capability(call.mode, call.mix_engines)
        report = self._report(call, query_plan=call.use_query_plan)
        return [
            SearchOutcome(
                answer=f"[stub {call.mode}] {query}",
                sources=self._sources_for(call),
                subqueries=self._subqueries(query, call.use_query_plan),
                engines=report,
            )
            for query in call.queries
        ]

    async def retrieve(self, call: SearchCall) -> list[RetrieveOutcome]:
        self.require_idle()
        self.require_capability(call.mode, call.mix_engines)
        report = self._report(call, query_plan=False)
        return [
            RetrieveOutcome(sources=self._sources_for(call), engines=report)
            for _ in call.queries
        ]

    # --- a canned graph surface, so a client can be built without a real one ---

    _ENTITIES = [
        {"id": "entity_1", "name": "Сенкевич", "type": "PERSON",
         "description": "stub entity", "communities": ["0"],
         "source_chunk_ids": ["chunk_1"]},
        {"id": "entity_2", "name": "Польша", "type": "COUNTRY",
         "description": "stub entity", "communities": ["0"],
         "source_chunk_ids": ["chunk_1", "chunk_2"]},
    ]
    _CHUNKS = [
        {"id": "chunk_1", "content": "stub chunk", "doc_id": "doc-1",
         "chunk_order_idx": 0, "num_tokens": 2},
        {"id": "chunk_2", "content": "stub chunk 2", "doc_id": "doc-1",
         "chunk_order_idx": 1, "num_tokens": 3},
    ]
    _COMMUNITY = {"id": "com-1", "level": 0, "cluster_id": 0,
                  "entity_count": 2, "relation_count": 1,
                  "entity_ids": ["entity_1", "entity_2"]}
    # Rendered the way RAGU renders a community report, so the title parser runs.
    _COMMUNITY_SUMMARY = (
        "Report title: Сенкевич и Польша\nReport summary: stub community summary"
    )
    _RELATIONS = [
        {
            "id": "relation_1",
            "subject_id": "entity_1",
            "object_id": "entity_2",
            "subject_name": "Сенкевич",
            "object_name": "Польша",
            "type": "born_in",
            "description": "stub relation",
            "strength": 1.0,
            "source_chunk_ids": ["chunk_1"],
        }
    ]

    async def graph_detail(self) -> dict[str, Any]:
        stats = self._stats or GraphStats()
        return {
            "entities": stats.entities,
            "relations": stats.relations,
            "chunks": stats.chunks,
            "communities": 1,
            "community_summaries": stats.community_summaries,
            "documents": 1,
            "embedding_dim": 8,
        }

    async def list_entities(
        self,
        *,
        limit,
        offset,
        entity_type=None,
        search=None,
        community_id=None,
        sort=None,
        order="asc",
        ids=None,
    ) -> tuple[int, list[Any]]:
        if ids is not None:
            by_id = {entity["id"]: entity for entity in self._ENTITIES}
            found = [by_id[key] for key in dict.fromkeys(ids) if key in by_id]
            return len(found), found

        items = self._ENTITIES
        if entity_type:
            items = [e for e in items if e["type"].casefold() == entity_type.casefold()]
        if search:
            items = [e for e in items if search.casefold() in e["name"].casefold()]
        if community_id is not None:
            items = [e for e in items if community_id in e.get("communities", [])]
        if sort == "name":
            items = sorted(items, key=lambda e: e["name"].casefold(), reverse=order == "desc")
        elif sort == "degree":
            items = sorted(items, key=self._degree, reverse=order == "desc")
        elif sort:
            raise InvalidRequestError(
                f"Unknown sort '{sort}'. Expected 'degree' or 'name'."
            )
        return len(items), items[offset : offset + limit]

    def _degree(self, entity: dict[str, Any]) -> int:
        return sum(
            1
            for relation in self._RELATIONS
            for side in (relation["subject_id"], relation["object_id"])
            if side == entity["id"]
        )

    async def get_entity(self, entity_id: str) -> Any:
        for entity in self._ENTITIES:
            if entity["id"] == entity_id:
                return entity
        raise NotFoundError(f"No entity with id '{entity_id}' in this graph.")

    async def list_chunks(self, *, limit, offset, ids=None) -> tuple[int, list[Any]]:
        if ids is not None:
            by_id = {chunk["id"]: chunk for chunk in self._CHUNKS}
            found = [by_id[key] for key in dict.fromkeys(ids) if key in by_id]
            return len(found), found
        return len(self._CHUNKS), self._CHUNKS[offset : offset + limit]

    async def list_relations(
        self, *, limit, offset, min_strength=None
    ) -> tuple[int, list[Any]]:
        items = self._RELATIONS
        if min_strength is not None:
            items = [r for r in items if r["strength"] >= min_strength]
        return len(items), items[offset : offset + limit]

    async def select_relations(
        self, *, entity_ids, edge_scope="induced", min_strength=None, limit, offset
    ) -> tuple[int, list[Any]]:
        wanted = set(entity_ids)
        both = edge_scope == "induced"
        kept = []
        for relation in self._RELATIONS:
            inside = (relation["subject_id"] in wanted, relation["object_id"] in wanted)
            if not (all(inside) if both else any(inside)):
                continue
            if min_strength is not None and relation["strength"] < min_strength:
                continue
            kept.append(relation)
        return len(kept), kept[offset : offset + limit]

    async def neighbors(self, entity_id: str, depth: int, limit: int) -> dict[str, Any]:
        if entity_id not in {e["id"] for e in self._ENTITIES}:
            raise NotFoundError(f"No entity with id '{entity_id}' in this graph.")
        return {
            "entities": self._ENTITIES,
            "relations": self._RELATIONS,
            "truncated": False,
        }

    async def list_communities(
        self, *, limit, offset, level=None, ids=None
    ) -> tuple[int, list[Any]]:
        items = [(dict(self._COMMUNITY), self._COMMUNITY_SUMMARY)]
        if ids is not None:
            wanted = set(ids)
            found = [item for item in items if item[0]["id"] in wanted]
            return len(found), found
        if level is not None:
            items = [item for item in items if item[0]["level"] == level]
        return len(items), items[offset : offset + limit]

    async def get_community(self, community_id: str) -> tuple[Any, Any]:
        if community_id != self._COMMUNITY["id"]:
            raise NotFoundError(f"No community with id '{community_id}' in this graph.")
        return dict(self._COMMUNITY), self._COMMUNITY_SUMMARY

    async def get_chunk(self, chunk_id: str) -> Any:
        for chunk in self._CHUNKS:
            if chunk["id"] == chunk_id:
                return chunk
        raise NotFoundError(f"No chunk with id '{chunk_id}' in this graph.")

    async def consistency(self) -> Any:
        return None

    async def reindex(self, kind: str) -> dict[str, Any]:
        if kind not in ("community", "descriptions", "graph"):
            raise InvalidRequestError(
                f"Unknown reindex '{kind}'. Expected one of "
                "['community', 'descriptions', 'graph']."
            )
        return {"reindex": kind}
