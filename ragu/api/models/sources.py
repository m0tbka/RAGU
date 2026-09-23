"""
The evidence an answer rests on: one item per source, with the typed fields
of its kind.
"""

from typing import Literal

from pydantic import BaseModel, Field


class EntityMeta(BaseModel):
    """
    The typed fields behind an entity source.
    """

    kind: Literal["entity"] = "entity"
    name: str
    type: str
    degree: int | None = Field(
        default=None, description="Relations touching this entity, when counted"
    )
    communities: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(
        default_factory=list, description="Chunks this entity was extracted from"
    )


class RelationMeta(BaseModel):
    """
    The typed fields behind a relation source.
    """

    kind: Literal["relation"] = "relation"
    subject_id: str
    object_id: str
    subject_name: str
    object_name: str
    type: str
    strength: float = 1.0
    source_chunk_ids: list[str] = Field(
        default_factory=list, description="Chunks this relation was extracted from"
    )


class ChunkMeta(BaseModel):
    """
    The typed fields behind a chunk source.
    """

    kind: Literal["chunk"] = "chunk"
    doc_id: str | None = None
    chunk_order_idx: int | None = None


class CommunityMeta(BaseModel):
    """
    The typed fields behind a community-summary source.

    ``level``, ``cluster_id`` and ``entity_count`` are filled only when the
    source carries a real community id. Global search reports insights the LLM
    wrote *about* communities and does not say which one each came from, so its
    sources carry the title alone.
    """

    kind: Literal["community_summary"] = "community_summary"
    level: int | None = None
    cluster_id: int | None = None
    title: str | None = None
    entity_count: int | None = None


SourceMeta = EntityMeta | RelationMeta | ChunkMeta | CommunityMeta


class SourceItem(BaseModel):
    id: str = Field(
        description="Stable source identifier, e.g. chunk_42 or community_3"
    )
    type: str = Field(
        description="Source kind: chunk, entity, relation, community_summary"
    )
    content: str = Field(default="", description="Source text")
    score: float | None = Field(
        default=None, description="Retrieval score when the engine provides one"
    )
    meta: SourceMeta | None = Field(
        default=None,
        discriminator="kind",
        description="Typed fields for this source kind, so a client need not "
        "fetch each source again to learn what it is",
    )
