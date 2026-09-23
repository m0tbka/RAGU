"""
Reading a graph's contents: entities, relations, communities and chunks.

Reads rather than searches: a client that draws the graph or traces an answer
back to its source needs the structure, not an answer about it.
"""

from typing import Any, Literal

from fastapi import APIRouter, Depends, Query

from ragu.api.backends.base import SearchBackend
from ragu.api.errors import InvalidRequestError
from ragu.api.models import (
    ChunkItem,
    ChunkPage,
    CommunityDetail,
    CommunityItem,
    CommunityPage,
    EntityItem,
    EntityPage,
    MAX_PAGE_LIMIT,
    Neighborhood,
    PageInfo,
    RelationItem,
    RelationPage,
    RelationSelectRequest,
)
from ragu.api.routes.deps import GRAPH_RESPONSES, get_backend
from ragu.api.search.mapping import split_report_title

router = APIRouter()


# A community can hold thousands of entities; the list is for highlighting one
# on a canvas, not for paging through it. Above this the caller asks /entities.
COMMUNITY_MEMBER_LIMIT = 1000


# A by-id selection travels in the query string, and an entity id is 36
# characters: five hundred of them already make a URL of roughly 20 KB, past
# what most servers accept on the request line. The page ceiling rose to 5000;
# this one deliberately did not follow it.
MAX_IDS_IN_QUERY = 500


def _require_id_count(ids: list[str] | None, limit: int) -> None:
    """
    Hold a by-id selection to what a URL can actually carry.

    :raises InvalidRequestError: If more ids were asked for than fit.
    """
    ceiling = min(limit, MAX_IDS_IN_QUERY)
    if ids is not None and len(ids) > ceiling:
        raise InvalidRequestError(
            f"Asked for {len(ids)} ids, which is over the limit of {ceiling}. "
            "Split the request, or POST the set to a selection route."
        )


def _chunk_item(chunk: Any) -> ChunkItem:
    """
    Render a chunk, from either a real ``Chunk`` or the stub's mapping.
    """
    if isinstance(chunk, dict):
        return ChunkItem(**chunk)
    return ChunkItem(
        id=chunk.id,
        content=chunk.content,
        doc_id=getattr(chunk, "doc_id", None),
        chunk_order_idx=getattr(chunk, "chunk_order_idx", None),
        num_tokens=getattr(chunk, "num_tokens", None),
    )


def _entity_item(entity: Any) -> EntityItem:
    """
    Render an entity, from either a real ``Entity`` or the stub's mapping.
    """
    if isinstance(entity, dict):
        return EntityItem(**entity)
    return EntityItem(
        id=entity.id,
        name=entity.entity_name,
        type=entity.entity_type,
        description=entity.description or "",
        communities=[
            str(cluster.get("cluster_id"))
            for cluster in (getattr(entity, "clusters", None) or [])
            if cluster.get("cluster_id") is not None
        ],
        source_chunk_ids=list(getattr(entity, "source_chunk_id", None) or []),
    )


def _relation_item(relation: Any) -> RelationItem:
    if isinstance(relation, dict):
        return RelationItem(**relation)
    return RelationItem(
        id=relation.id,
        subject_id=relation.subject_id,
        object_id=relation.object_id,
        subject_name=relation.subject_name,
        object_name=relation.object_name,
        type=relation.relation_type,
        description=relation.description or "",
        strength=float(getattr(relation, "relation_strength", 1.0)),
        source_chunk_ids=list(getattr(relation, "source_chunk_id", None) or []),
    )


def _community_item(community: Any, summary: Any) -> CommunityItem:
    title, body = split_report_title(_summary_text(summary))
    if isinstance(community, dict):
        return CommunityItem(**community, title=title, summary=body)

    entities = community.entities or []
    members = [entity.id for entity in entities][:COMMUNITY_MEMBER_LIMIT]
    return CommunityItem(
        id=community.id,
        level=community.level,
        cluster_id=community.cluster_id,
        entity_count=len(entities),
        relation_count=len(community.relations or []),
        title=title,
        summary=body,
        entity_ids=members,
        truncated=len(entities) > len(members),
    )


def _summary_text(summary: Any) -> str | None:
    if summary is None or isinstance(summary, str):
        return summary
    if isinstance(summary, dict):
        return summary.get("summary")
    return getattr(summary, "summary", None)


@router.get(
    "/v1/graphs/{graph_id}/entities",
    response_model=EntityPage,
    responses=GRAPH_RESPONSES,
    tags=["graphs"],
)
async def list_entities(
    graph_id: str,
    backend: SearchBackend = Depends(get_backend),
    limit: int = Query(default=50, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(default=0, ge=0),
    type: str | None = Query(default=None, description="Exact entity type"),
    search: str | None = Query(default=None, description="Substring of the name"),
    community_id: str | None = Query(
        default=None, description="Only entities in this community"
    ),
    sort: Literal["degree", "name"] | None = Query(
        default=None, description="Sort key; storage order when omitted"
    ),
    order: Literal["asc", "desc"] = Query(default="asc"),
    ids: list[str] | None = Query(
        default=None,
        description="Fetch exactly these entities. Every other filter and the "
        "paging are ignored, and an unknown id is skipped rather than failing "
        "the selection.",
    ),
) -> EntityPage:
    _require_id_count(ids, limit)
    total, entities = await backend.list_entities(
        limit=limit,
        offset=offset,
        entity_type=type,
        search=search,
        community_id=community_id,
        sort=sort,
        order=order,
        ids=ids,
    )
    return EntityPage(
        page=PageInfo(total=total, limit=limit, offset=0 if ids else offset),
        entities=[_entity_item(entity) for entity in entities],
    )


@router.get(
    "/v1/graphs/{graph_id}/entities/{entity_id}",
    response_model=EntityItem,
    responses=GRAPH_RESPONSES,
    tags=["graphs"],
)
async def get_entity(
    graph_id: str,
    entity_id: str,
    backend: SearchBackend = Depends(get_backend),
) -> EntityItem:
    """
    One entity, without having to ask for its neighbourhood to find it.
    """
    return _entity_item(await backend.get_entity(entity_id))


@router.get(
    "/v1/graphs/{graph_id}/chunks",
    response_model=ChunkPage,
    responses=GRAPH_RESPONSES,
    tags=["graphs"],
)
async def list_chunks(
    graph_id: str,
    backend: SearchBackend = Depends(get_backend),
    limit: int = Query(default=50, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(default=0, ge=0),
    ids: list[str] | None = Query(
        default=None, description="Fetch exactly these chunks, skipping the unknown"
    ),
) -> ChunkPage:
    """
    Source chunks, for resolving the ids a search answer cites in one call.
    """
    _require_id_count(ids, limit)
    total, chunks = await backend.list_chunks(limit=limit, offset=offset, ids=ids)
    return ChunkPage(
        page=PageInfo(total=total, limit=limit, offset=0 if ids else offset),
        chunks=[_chunk_item(chunk) for chunk in chunks],
    )


@router.get(
    "/v1/graphs/{graph_id}/relations",
    response_model=RelationPage,
    responses=GRAPH_RESPONSES,
    tags=["graphs"],
)
async def list_relations(
    graph_id: str,
    backend: SearchBackend = Depends(get_backend),
    limit: int = Query(default=50, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(default=0, ge=0),
    min_strength: float | None = Query(default=None),
) -> RelationPage:
    total, relations = await backend.list_relations(
        limit=limit, offset=offset, min_strength=min_strength
    )
    return RelationPage(
        page=PageInfo(total=total, limit=limit, offset=offset),
        relations=[_relation_item(relation) for relation in relations],
    )


@router.post(
    "/v1/graphs/{graph_id}/relations/select",
    response_model=RelationPage,
    responses=GRAPH_RESPONSES,
    tags=["graphs"],
)
async def select_relations(
    graph_id: str,
    payload: RelationSelectRequest,
    backend: SearchBackend = Depends(get_backend),
) -> RelationPage:
    """
    Relations restricted to a set of entities.

    A canvas showing N entities needs the relations *between* them — the induced
    subgraph. Anything else draws edges running off to nodes that are not on
    screen. The set travels in the body because five hundred 36-character ids
    make a URL no server will accept.
    """
    total, relations = await backend.select_relations(
        entity_ids=payload.entity_ids,
        edge_scope=payload.edge_scope,
        min_strength=payload.min_strength,
        limit=payload.limit,
        offset=payload.offset,
    )
    return RelationPage(
        page=PageInfo(total=total, limit=payload.limit, offset=payload.offset),
        relations=[_relation_item(relation) for relation in relations],
    )


@router.get(
    "/v1/graphs/{graph_id}/entities/{entity_id}/neighbors",
    response_model=Neighborhood,
    responses=GRAPH_RESPONSES,
    tags=["graphs"],
)
async def entity_neighbors(
    graph_id: str,
    entity_id: str,
    backend: SearchBackend = Depends(get_backend),
    depth: int = Query(default=1, ge=1, le=4),
    limit: int = Query(default=200, ge=1, le=2000),
) -> Neighborhood:
    """
    Everything within ``depth`` hops, for drawing a piece of the graph.

    Coordinates are not returned: the client knows its own viewport and lays the
    graph out itself.
    """
    found = await backend.neighbors(entity_id, depth, limit)
    return Neighborhood(
        root=entity_id,
        depth=depth,
        entities=[_entity_item(entity) for entity in found["entities"]],
        relations=[_relation_item(relation) for relation in found["relations"]],
        truncated=found["truncated"],
    )


@router.get(
    "/v1/graphs/{graph_id}/communities",
    response_model=CommunityPage,
    responses=GRAPH_RESPONSES,
    tags=["graphs"],
)
async def list_communities(
    graph_id: str,
    backend: SearchBackend = Depends(get_backend),
    limit: int = Query(default=50, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(default=0, ge=0),
    level: int | None = Query(default=None, description="Leiden level"),
    ids: list[str] | None = Query(
        default=None,
        description="Fetch exactly these communities, skipping the unknown",
    ),
) -> CommunityPage:
    _require_id_count(ids, limit)
    total, rows = await backend.list_communities(
        limit=limit, offset=offset, level=level, ids=ids
    )
    return CommunityPage(
        page=PageInfo(total=total, limit=limit, offset=0 if ids else offset),
        communities=[_community_item(community, summary) for community, summary in rows],
    )


@router.get(
    "/v1/graphs/{graph_id}/communities/{community_id}",
    response_model=CommunityDetail,
    responses=GRAPH_RESPONSES,
    tags=["graphs"],
)
async def get_community(
    graph_id: str,
    community_id: str,
    backend: SearchBackend = Depends(get_backend),
) -> CommunityDetail:
    community, summary = await backend.get_community(community_id)
    item = _community_item(community, summary)
    members = (
        {"entities": [], "relations": []}
        if isinstance(community, dict)
        else {"entities": community.entities or [], "relations": community.relations or []}
    )
    return CommunityDetail(
        **item.model_dump(),
        entities=[_entity_item(entity) for entity in members["entities"]],
        relations=[_relation_item(relation) for relation in members["relations"]],
    )


@router.get(
    "/v1/graphs/{graph_id}/chunks/{chunk_id}",
    response_model=ChunkItem,
    responses=GRAPH_RESPONSES,
    tags=["graphs"],
)
async def get_chunk(
    graph_id: str,
    chunk_id: str,
    backend: SearchBackend = Depends(get_backend),
) -> ChunkItem:
    """
    One source chunk, for tracing an answer back to the corpus.
    """
    return _chunk_item(await backend.get_chunk(chunk_id))
