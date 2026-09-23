"""
Reading a loaded graph: entities, relations, communities and chunks.

The storages answer point lookups, but everything a client pages, filters or
sorts over needs the whole entity or relation list. ``GraphView`` builds that
list once, keeps it while it fits, and runs every whole-graph pass off the event
loop.
"""

import asyncio
from collections.abc import Sequence
from typing import Any

from ragu.api.errors import InvalidRequestError, NotFoundError


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


def _cluster_ids(entity: Any) -> set[str]:
    """
    The community ids an entity belongs to, as the wire spells them.
    """
    return {
        str(cluster.get("cluster_id"))
        for cluster in (getattr(entity, "clusters", None) or [])
        if cluster.get("cluster_id") is not None
    }


class GraphView:
    """
    Read-only access to one loaded graph, with the lists it pages through.

    Owns the materialized node, edge and degree lists: built on first use, kept
    while they fit under ``cache_max_items``, and dropped by whoever writes to
    the graph. It knows nothing of builds or locks — the backend refuses a read
    while a build runs before it ever asks.
    """

    def __init__(self, graph: Any, cache_max_items: int):
        """
        :param graph: The loaded ``KnowledgeGraph``.
        :param cache_max_items: Longest list worth keeping; ``0`` keeps none.
        """
        self.graph = graph
        self.cache_max_items = cache_max_items
        self._node_cache: list[Any] | None = None
        self._edge_cache: list[Any] | None = None
        self._degree_cache: dict[str, int] | None = None

    def drop_cache(self) -> None:
        """
        Forget the materialized lists; the next read rebuilds them.
        """
        self._node_cache = None
        self._edge_cache = None
        self._degree_cache = None

    def _worth_caching(self, items: list[Any]) -> bool:
        """
        Whether a materialized list is small enough to keep.

        The cache trades memory for paging speed, and on a graph of a few hundred
        thousand relations that trade is a permanent floor under the process for
        the benefit of one endpoint. Above the ceiling the list is rebuilt per
        page instead — slower, but the service survives the request.
        """
        limit = self.cache_max_items
        return limit > 0 and len(items) <= limit

    async def nodes(self) -> list[Any]:
        """
        Every entity, materialized once and kept while it fits.

        ``get_all_nodes`` rebuilds an ``Entity`` per node on each call, so a
        client paging through a graph would otherwise pay O(n) per page. The
        list is dropped whenever the graph is written to, and not kept at all
        above ``cache_max_items``.
        """
        if self._node_cache is not None:
            return self._node_cache
        nodes = await self.graph.index.graph_backend.get_all_nodes()
        if self._worth_caching(nodes):
            self._node_cache = nodes
        return nodes

    async def edges(self) -> list[Any]:
        """
        Every relation, materialized once and kept while it fits. See :meth:`nodes`.
        """
        if self._edge_cache is not None:
            return self._edge_cache
        edges = await self.graph.index.graph_backend.get_all_edges()
        if self._worth_caching(edges):
            self._edge_cache = edges
        return edges

    async def degrees(self) -> dict[str, int]:
        """
        How many relations touch each entity.

        Counted from the edge list rather than asked of the storage: the adapter
        contract exposes degree per *edge*, not per node, and the edges are
        materialized anyway for the relation routes. The count itself runs off
        the event loop — it is the single most expensive pass in this class.
        """
        if self._degree_cache is None:
            edges = await self.edges()
            self._degree_cache = await asyncio.to_thread(_count_degrees, edges)
        return self._degree_cache

    async def counts(self) -> dict[str, int]:
        """
        The sizes the startup measurement does not cover.

        :return: Numbers of communities and of distinct source documents.
        """
        index = self.graph.index
        communities = await index.community_kv_storage.all_keys()
        chunks = await index.chunks_kv_storage.get_by_ids(
            await index.chunks_kv_storage.all_keys()
        )
        documents = {
            chunk.get("doc_id") if isinstance(chunk, dict) else getattr(chunk, "doc_id", None)
            for chunk in chunks
            if chunk is not None
        }
        return {
            "communities": len(communities),
            "documents": len({doc for doc in documents if doc}),
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
        if ids is not None:
            found = await self._entities_by_id(ids)
            return len(found), found

        if sort not in (None, "degree", "name"):
            raise InvalidRequestError(
                f"Unknown sort '{sort}'. Expected 'degree' or 'name'."
            )

        nodes = await self.nodes()
        if not (entity_type or search or community_id is not None or sort):
            # Plain paging touches nothing, so it does not pay for a thread.
            return len(nodes), nodes[offset : offset + limit]

        degrees = await self.degrees() if sort == "degree" else None
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
        wanted = list(dict.fromkeys(ids))
        found = await self.graph.index.graph_backend.get_nodes(wanted)
        return [node for node in found if node is not None]

    async def get_entity(self, entity_id: str) -> Any:
        found = await self.graph.index.graph_backend.get_nodes([entity_id])
        if not found or found[0] is None:
            raise NotFoundError(f"No entity with id '{entity_id}' in this graph.")
        return found[0]

    async def list_chunks(
        self, *, limit: int, offset: int, ids: Sequence[str] | None = None
    ) -> tuple[int, list[Any]]:
        graph = self.graph
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
        edges = await self.edges()
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
        edges = await self.edges()
        # A membership test per relation over the whole corpus: pure CPU, so it
        # runs off the event loop like the other whole-graph passes.
        kept = await asyncio.to_thread(
            _select_relations, edges, set(entity_ids), edge_scope, min_strength
        )
        return len(kept), kept[offset : offset + limit]

    async def neighbors(self, entity_id: str, depth: int, limit: int) -> dict[str, Any]:
        backend = self.graph.index.graph_backend

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
        graph = self.graph
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
        graph = self.graph
        found = await graph.get_communities([community_id])
        if not found or found[0] is None:
            raise NotFoundError(f"No community with id '{community_id}' in this graph.")
        summary = await graph.index.community_summary_kv_storage.get_by_id(community_id)
        return found[0], summary

    async def get_chunk(self, chunk_id: str) -> Any:
        found = await self.graph.get_chunks([chunk_id])
        if not found or found[0] is None:
            raise NotFoundError(f"No chunk with id '{chunk_id}' in this graph.")
        return found[0]

    async def consistency(self) -> Any:
        return await self.graph.index.check_consistency()
