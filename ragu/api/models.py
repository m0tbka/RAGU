"""
Request and response schemas of the search API.
"""

from datetime import datetime
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field
from ragu.search_engine.global_search import GlobalSearchParams
from ragu.search_engine.local_search import LocalParams
from ragu.search_engine.mix_search import MixQueryParams
from ragu.search_engine.naive_search import NaiveSearchParams

SearchMode = Literal["global", "local", "naive", "mix"]

# What a search mode needs the graph to hold. Named here rather than in the
# backends so that the configuration layer can validate against the same set.
Capability = Literal["entity_graph", "community_summaries", "vector_index"]

CAPABILITIES: frozenset[str] = frozenset(get_args(Capability))

# The value is interpolated into the answer prompt ("Provide the answer in the
# following language: {{ language }}"), so it is a prompt-injection surface: it
# is constrained to a plain language name rather than accepted as free text.

RerankField = Field(
    default=True,
    description="Use the configured reranker. A no-op when the deployment has "
    "none; a failing one degrades to the un-reranked order rather than a 500.",
)

LanguageField = Field(
    default=None,
    min_length=2,
    max_length=32,
    pattern=r"^[A-Za-z][A-Za-z \-]*$",
    description="Answer language, e.g. 'russian'. Defaults to RAGU_API_LANGUAGE.",
)

# Which leaf engines the mix ensemble runs. Local and naive by default: they are
# the cheap pair and the one every graph with a vector index can serve. Adding
# global is a deliberate choice, because global costs one LLM call per surviving
# community and needs community summaries the graph may not have.
MixEngine = Literal["local", "naive", "global"]

DEFAULT_MIX_ENGINES: tuple[str, ...] = ("local", "naive")

MixEnginesField = Field(
    default_factory=lambda: list(DEFAULT_MIX_ENGINES),
    min_length=1,
    max_length=3,
    description="Leaf engines to ensemble, in order. Adding 'global' requires "
    "community summaries and costs one LLM call per surviving community.",
)



class GlobalSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Search query")
    params: GlobalSearchParams = Field(
        default_factory=GlobalSearchParams,
        description="GlobalSearchEngine retrieval parameters: min_cluster_size",
    )
    language: str | None = LanguageField


class LocalSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Search query")
    use_query_plan: bool = Field(
        default=True, description="Decompose the query into subqueries"
    )
    params: LocalParams = Field(
        default_factory=LocalParams,
        description="LocalSearchEngine retrieval parameters: top_k, rerank_top_k, "
        "use_summary, use_chunks",
    )
    language: str | None = LanguageField
    rerank: bool = RerankField


class NaiveSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Search query")
    use_query_plan: bool = Field(
        default=True, description="Decompose the query into subqueries"
    )
    params: NaiveSearchParams = Field(
        default_factory=NaiveSearchParams,
        description="NaiveSearchEngine retrieval parameters: top_k, rerank_top_k",
    )
    language: str | None = LanguageField
    rerank: bool = RerankField


class MixSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Search query")
    use_query_plan: bool = Field(
        default=True, description="Decompose the query into subqueries"
    )
    params: MixQueryParams = Field(
        default_factory=MixQueryParams,
        description="MixSearchEngine parameters: ensemble_responses",
    )
    engines: list[MixEngine] = MixEnginesField
    local_params: LocalParams = Field(
        default_factory=LocalParams,
        description="Parameters for the local child engine. MixSearchEngine reads "
        "child parameters from its constructor, not from the request-time params, "
        "so they are named separately here.",
    )
    naive_params: NaiveSearchParams = Field(
        default_factory=NaiveSearchParams,
        description="Parameters for the naive child engine",
    )
    global_params: GlobalSearchParams = Field(
        default_factory=GlobalSearchParams,
        description="Parameters for the global child engine, when it is selected",
    )
    language: str | None = LanguageField
    rerank: bool = RerankField


class ChildEngineReport(BaseModel):
    """
    What one child engine of an ensemble actually did.
    """

    engine: str = Field(description="Child engine class name")
    mode: str | None = Field(
        default=None, description="Search mode the child corresponds to, when known"
    )
    ok: bool = Field(description="Whether the child produced a result")
    error: str | None = Field(
        default=None, description="Why the child failed, when it did"
    )


class EngineReport(BaseModel):
    """
    What ran, as opposed to what was asked for.

    ``MixSearchEngine`` tolerates child failures by design, so a request that
    asked for graph plus chunks can silently be answered from chunks alone.
    This reports the difference instead of hiding it.
    """

    requested: SearchMode = Field(description="Mode the client asked for")
    used: str = Field(description="Engine class that produced the answer")
    query_plan: bool = Field(
        default=False, description="Whether the query was decomposed first"
    )
    degraded: bool = Field(
        default=False, description="True when some child engine did not contribute"
    )
    children: list[ChildEngineReport] = Field(
        default_factory=list, description="Per-child outcome, for ensemble engines"
    )
    reranked: bool = Field(
        default=False, description="Whether a reranker actually reordered the results"
    )
    rerank_error: str | None = Field(
        default=None,
        description="Why reranking was skipped; the answer is the un-reranked one",
    )


class StageUsageModel(BaseModel):
    """
    What one stage of a request cost.
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieval_ms: float | None = Field(
        default=None, description="Wall time spent retrieving, in milliseconds"
    )
    generation_ms: float | None = Field(
        default=None, description="Wall time spent in the LLM, in milliseconds"
    )


class UsageModel(BaseModel):
    """
    What a request cost, by stage.

    Token counts are measured with the tokenizer rather than read from the
    provider — the LLM clients return the parsed answer, not the raw response —
    so they are close, not exact. Price from your provider's bill, not from here.
    """

    estimated: bool = Field(
        default=True, description="Token counts are measured locally, not billed"
    )
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    stages: dict[str, StageUsageModel] = Field(default_factory=dict)


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


class SubqueryItem(BaseModel):
    query: str = Field(description="Subquery produced by the query plan")
    answer: str = Field(default="", description="Intermediate answer to the subquery")


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    used_query_plan: bool
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    subqueries: list[SubqueryItem] = Field(default_factory=list)
    engines: EngineReport = Field(
        description="What actually ran, including child-engine failures"
    )
    usage: UsageModel | None = Field(
        default=None, description="What this request cost, by stage"
    )


class GlobalRetrieveRequest(BaseModel):
    """
    Retrieval-only request.

    No ``use_query_plan``: ``QueryPlanEngine.batch_search`` delegates straight to
    the wrapped engine and does no planning, so accepting the flag would promise
    a decomposition that never happens.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Search query")
    params: GlobalSearchParams = Field(default_factory=GlobalSearchParams)


class LocalRetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Search query")
    params: LocalParams = Field(default_factory=LocalParams)
    rerank: bool = RerankField


class NaiveRetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Search query")
    params: NaiveSearchParams = Field(default_factory=NaiveSearchParams)
    rerank: bool = RerankField


class MixRetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Search query")
    params: MixQueryParams = Field(default_factory=MixQueryParams)
    engines: list[MixEngine] = MixEnginesField
    local_params: LocalParams = Field(default_factory=LocalParams)
    naive_params: NaiveSearchParams = Field(default_factory=NaiveSearchParams)
    global_params: GlobalSearchParams = Field(default_factory=GlobalSearchParams)
    rerank: bool = RerankField


class RetrieveResponse(BaseModel):
    """
    Context gathered for a query, with no answer generated for it.
    """

    query: str
    mode: SearchMode
    sources: list[SourceItem] = Field(default_factory=list)
    engines: EngineReport = Field(
        description="What actually ran, including child-engine failures"
    )
    usage: UsageModel | None = Field(
        default=None, description="What this request cost, by stage"
    )


class BatchQueries(BaseModel):
    """
    Shared shape of the batch requests: many queries, one mode, one parameter set.

    One mode and one parameter set per batch is what makes the batch worth
    making: the engines share retrieval across the whole list, and
    ``QueryPlanEngine`` merges independent subqueries from different top-level
    queries into the same child batch.
    """

    model_config = ConfigDict(extra="forbid")

    queries: list[str] = Field(
        min_length=1, description="Queries to answer in one pass"
    )


class GlobalBatchRequest(BatchQueries):
    params: GlobalSearchParams = Field(default_factory=GlobalSearchParams)
    language: str | None = LanguageField


class LocalBatchRequest(BatchQueries):
    use_query_plan: bool = Field(default=True)
    params: LocalParams = Field(default_factory=LocalParams)
    language: str | None = LanguageField
    rerank: bool = RerankField


class NaiveBatchRequest(BatchQueries):
    use_query_plan: bool = Field(default=True)
    params: NaiveSearchParams = Field(default_factory=NaiveSearchParams)
    language: str | None = LanguageField
    rerank: bool = RerankField


class MixBatchRequest(BatchQueries):
    use_query_plan: bool = Field(default=True)
    params: MixQueryParams = Field(default_factory=MixQueryParams)
    engines: list[MixEngine] = MixEnginesField
    local_params: LocalParams = Field(default_factory=LocalParams)
    naive_params: NaiveSearchParams = Field(default_factory=NaiveSearchParams)
    global_params: GlobalSearchParams = Field(default_factory=GlobalSearchParams)
    language: str | None = LanguageField
    rerank: bool = RerankField


class BatchSearchItem(BaseModel):
    """
    One query's result inside a batch.

    A query that retrieved nothing carries ``error`` instead of an answer: one
    empty query must not fail the whole batch.
    """

    query: str
    answer: str = ""
    sources: list[SourceItem] = Field(default_factory=list)
    subqueries: list[SubqueryItem] = Field(default_factory=list)
    error: "ErrorBody | None" = None


class BatchSearchResponse(BaseModel):
    mode: SearchMode
    used_query_plan: bool
    engines: EngineReport
    results: list[BatchSearchItem]
    usage: UsageModel | None = Field(
        default=None, description="What the whole batch cost, by stage"
    )


class ErrorBody(BaseModel):
    code: str
    mode: str | None = None
    missing_capability: str | None = None
    message: str
    request_id: str | None = Field(
        default=None,
        description="Correlates this response with the service log; echoed in "
        "the X-Request-ID header",
    )


class ErrorResponse(BaseModel):
    error: ErrorBody


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


class BuildRequest(BaseModel):
    """
    Documents to add to a graph.
    """

    model_config = ConfigDict(extra="forbid")

    documents: list[str] = Field(
        min_length=1, description="Raw document texts to ingest"
    )


class JobResponse(BaseModel):
    """
    A long-running operation and what became of it.
    """

    id: str
    kind: str
    graph_id: str
    state: str = Field(
        description="queued, running, succeeded, failed or cancelled"
    )
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


class JobListResponse(BaseModel):
    jobs: list[JobResponse] = Field(default_factory=list)


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
    modes: list["ModeAvailability"] = Field(default_factory=list)


class ConsistencyIssueItem(BaseModel):
    check: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ConsistencyReportModel(BaseModel):
    consistent: bool
    issues: list[ConsistencyIssueItem] = Field(default_factory=list)


class OntologyResponse(BaseModel):
    """
    The entity and relation types the extractors work with.
    """

    entity_types: list[str] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)


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


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when searches can be served, else 'degraded'")
    graph_loaded: bool
    stats: GraphStatsResponse | None = Field(
        default=None, description="Graph sizes; absent until the graph is loaded"
    )
    error: str | None = Field(
        default=None, description="Why the backend is not ready, when it is not"
    )


BatchSearchItem.model_rebuild()


GraphDetail.model_rebuild()
