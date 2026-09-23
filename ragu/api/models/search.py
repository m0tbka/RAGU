"""
Search, retrieval and batch requests, and what they answer with.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ragu.api.models.common import ErrorBody, SearchMode
from ragu.api.models.sources import SourceItem

# From the parameter module, not the engine modules: the schemas are what a
# client imports, and the engines drag the whole library in behind them.
from ragu.search_engine.params import (
    GlobalSearchParams,
    LocalParams,
    MixQueryParams,
    NaiveSearchParams,
)


RerankField = Field(
    default=True,
    description="Use the configured reranker. A no-op when the deployment has "
    "none; a failing one degrades to the un-reranked order rather than a 500.",
)


# The value is interpolated into the answer prompt ("Provide the answer in the
# following language: {{ language }}"), so it is a prompt-injection surface: it
# is constrained to a plain language name rather than accepted as free text.
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
    rerank_ms: float | None = Field(
        default=None,
        description="Wall time spent in the reranker, in milliseconds. Set on the "
        "'rerank' stage, whose calls are not LLM calls and so are left out of the "
        "request's call total",
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
    error: ErrorBody | None = None


class BatchSearchResponse(BaseModel):
    mode: SearchMode
    used_query_plan: bool
    engines: EngineReport
    results: list[BatchSearchItem]
    usage: UsageModel | None = Field(
        default=None, description="What the whole batch cost, by stage"
    )
