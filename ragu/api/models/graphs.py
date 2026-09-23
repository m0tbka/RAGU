"""
The graph catalogue, and read-only views of what one graph holds.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ragu.api.models.common import Capability, SearchMode


class GraphStatsResponse(BaseModel):
    """
    Sizes of the stores each search mode reads, counted once at startup.
    """

    entities: int = Field(description="Vectorized entities backing local search")
    relations: int = Field(description="Vectorized relations backing local search")
    chunks: int = Field(description="Vectorized chunks backing naive search")
    community_summaries: int = Field(
        description="Community summaries backing global search"
    )


class ModeAvailability(BaseModel):
    """
    Whether one search mode can run against a graph, and why not when it cannot.

    A client has to know which modes to offer before it offers them; without
    this it can only try one and read the 409.
    """

    mode: SearchMode
    available: bool
    missing_capability: Capability | None = None
    reason: str | None = Field(
        default=None, description="What the graph would need, when the mode is off"
    )


class GraphInfo(BaseModel):
    """
    One graph in the catalogue.
    """

    id: str
    loaded: bool
    language: str = Field(description="Default answer language for this graph")
    stats: GraphStatsResponse | None = None
    modes: list[ModeAvailability] = Field(default_factory=list)
    error: str | None = Field(
        default=None, description="Why this graph is not loaded, when it is not"
    )


class GraphListResponse(BaseModel):
    default: str = Field(description="Graph served by the paths that name none")
    graphs: list[GraphInfo] = Field(default_factory=list)


class GraphDetail(BaseModel):
    """
    Everything a client needs to describe a corpus in an interface.
    """

    id: str
    loaded: bool
    language: str
    entities: int = 0
    relations: int = 0
    chunks: int = 0
    communities: int = 0
    community_summaries: int = 0
    documents: int = 0
    embedding_dim: int | None = None
    accepts_documents: bool = False
    modes: list[ModeAvailability] = Field(default_factory=list)
    updated_at: datetime | None = Field(
        default=None,
        description="When the graph's data was last written: the latest "
        "modification time among the files in its storage folder. For a graph "
        "built offline and mounted as is, this is its build time",
    )
    created_at: datetime | None = Field(
        default=None,
        description="When the storage folder was created, where the filesystem "
        "records that. Linux does not expose it, so on a typical container "
        "deployment this is null — null, and not a placeholder date",
    )


# The largest page any listing will return. A consumer that has to export a
# corpus pages through tens of thousands of rows, and 500 at a time turns that
# into a hundred round trips.
MAX_PAGE_LIMIT = 5000


class PageInfo(BaseModel):
    """
    Where a listing sits in the whole.
    """

    total: int = Field(description="Items matching the filter, before paging")
    limit: int
    offset: int


class EntityItem(BaseModel):
    id: str
    name: str
    type: str
    description: str = ""
    degree: int | None = Field(
        default=None, description="Relations touching this entity, when counted"
    )
    communities: list[str] = Field(
        default_factory=list, description="Community ids this entity belongs to"
    )
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunks this entity was extracted from. The only way back "
        "from an entity to the text that produced it: there is no reverse index",
    )


class RelationItem(BaseModel):
    id: str
    subject_id: str
    object_id: str
    subject_name: str
    object_name: str
    type: str
    description: str = ""
    strength: float = 1.0
    source_chunk_ids: list[str] = Field(
        default_factory=list, description="Chunks this relation was extracted from"
    )


class EntityPage(BaseModel):
    page: PageInfo
    entities: list[EntityItem] = Field(default_factory=list)


# How many entities one selection may name. Ten thousand ids is about 400 KB of
# body, well inside the default 32 MiB ceiling.
MAX_SELECT_IDS = 10_000


class RelationSelectRequest(BaseModel):
    """
    Relations restricted to a set of entities.

    A POST rather than a query string: an entity id is 36 characters, so five
    hundred of them make a URL of roughly 24 KB — past the 8 KB request line
    most servers accept. The set is the body instead.
    """

    model_config = ConfigDict(extra="forbid")

    entity_ids: list[str] = Field(
        min_length=1,
        max_length=MAX_SELECT_IDS,
        description="Entities the selection is restricted to. An id the graph "
        "does not hold is skipped rather than failing the request.",
    )
    edge_scope: Literal["induced", "incident"] = Field(
        default="induced",
        description="'induced' keeps a relation only when BOTH ends are in the "
        "set — the induced subgraph a canvas draws. 'incident' keeps it when "
        "either end is, which is what expanding a neighbourhood needs.",
    )
    min_strength: float | None = Field(
        default=None, description="Keep only relations at least this strong"
    )
    limit: int = Field(default=500, ge=1, le=MAX_PAGE_LIMIT)
    offset: int = Field(default=0, ge=0)


class RelationPage(BaseModel):
    page: PageInfo
    relations: list[RelationItem] = Field(default_factory=list)


class Neighborhood(BaseModel):
    """
    One entity and everything within ``depth`` hops of it.

    Positions are not returned: a client laying the graph out knows its own
    viewport and does the layout itself.
    """

    root: str
    depth: int
    entities: list[EntityItem] = Field(default_factory=list)
    relations: list[RelationItem] = Field(default_factory=list)
    truncated: bool = Field(
        default=False,
        description="The neighbourhood hit the node ceiling and was cut short",
    )


class CommunityItem(BaseModel):
    id: str
    level: int
    cluster_id: int
    entity_count: int = 0
    relation_count: int = 0
    title: str | None = Field(
        default=None,
        description="Report title, lifted out of the summary text RAGU renders",
    )
    summary: str | None = Field(
        default=None, description="Report body, without the title line"
    )
    entity_ids: list[str] = Field(
        default_factory=list,
        description="Members, for highlighting the community without fetching it",
    )
    truncated: bool = Field(
        default=False, description="entity_ids hit the ceiling and was cut short"
    )


class CommunityPage(BaseModel):
    page: PageInfo
    communities: list[CommunityItem] = Field(default_factory=list)


class CommunityDetail(CommunityItem):
    entities: list[EntityItem] = Field(default_factory=list)
    relations: list[RelationItem] = Field(default_factory=list)


class ChunkItem(BaseModel):
    """
    One source chunk, for tracing an answer back to the corpus.
    """

    id: str
    content: str
    doc_id: str | None = None
    chunk_order_idx: int | None = None
    num_tokens: int | None = None


class ChunkPage(BaseModel):
    page: PageInfo
    chunks: list[ChunkItem] = Field(default_factory=list)


class ConsistencyIssueItem(BaseModel):
    check: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ConsistencyReportModel(BaseModel):
    consistent: bool
    issues: list[ConsistencyIssueItem] = Field(default_factory=list)
