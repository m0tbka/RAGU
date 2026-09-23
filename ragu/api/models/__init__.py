"""
Request and response schemas of the search API.

Split by resource, like the routes. Import from this package: its names are
the public surface, while the module a schema lives in is an internal detail
that may move.
"""

from ragu.api.models.common import (
    CAPABILITIES,
    Capability,
    ErrorBody,
    ErrorResponse,
    SearchMode,
)
from ragu.api.models.sources import (
    ChunkMeta,
    CommunityMeta,
    EntityMeta,
    RelationMeta,
    SourceItem,
    SourceMeta,
)
from ragu.api.models.search import (
    BatchQueries,
    BatchSearchItem,
    BatchSearchResponse,
    ChildEngineReport,
    DEFAULT_MIX_ENGINES,
    EngineReport,
    GlobalBatchRequest,
    GlobalRetrieveRequest,
    GlobalSearchRequest,
    LanguageField,
    LocalBatchRequest,
    LocalRetrieveRequest,
    LocalSearchRequest,
    MixBatchRequest,
    MixEngine,
    MixEnginesField,
    MixRetrieveRequest,
    MixSearchRequest,
    NaiveBatchRequest,
    NaiveRetrieveRequest,
    NaiveSearchRequest,
    RerankField,
    RetrieveResponse,
    SearchResponse,
    StageUsageModel,
    SubqueryItem,
    UsageModel,
)
from ragu.api.models.graphs import (
    ChunkItem,
    ChunkPage,
    CommunityDetail,
    CommunityItem,
    CommunityPage,
    ConsistencyIssueItem,
    ConsistencyReportModel,
    EntityItem,
    EntityPage,
    GraphDetail,
    GraphInfo,
    GraphListResponse,
    GraphStatsResponse,
    MAX_PAGE_LIMIT,
    MAX_SELECT_IDS,
    ModeAvailability,
    Neighborhood,
    PageInfo,
    RelationItem,
    RelationPage,
    RelationSelectRequest,
)
from ragu.api.models.jobs import (
    BuildRequest,
    JobListResponse,
    JobResponse,
)
from ragu.api.models.service import (
    HealthResponse,
    OntologyResponse,
)

# The engine parameter dataclasses the request schemas embed, so that a client
# builds a request from one import.
from ragu.search_engine.params import (
    GlobalSearchParams,
    LocalParams,
    MixQueryParams,
    NaiveSearchParams,
)

__all__ = [
    # common
    "SearchMode",
    "Capability",
    "CAPABILITIES",
    "ErrorBody",
    "ErrorResponse",
    # sources
    "EntityMeta",
    "RelationMeta",
    "ChunkMeta",
    "CommunityMeta",
    "SourceMeta",
    "SourceItem",
    # search
    "RerankField",
    "LanguageField",
    "MixEngine",
    "DEFAULT_MIX_ENGINES",
    "MixEnginesField",
    "GlobalSearchRequest",
    "LocalSearchRequest",
    "NaiveSearchRequest",
    "MixSearchRequest",
    "ChildEngineReport",
    "EngineReport",
    "StageUsageModel",
    "UsageModel",
    "SubqueryItem",
    "SearchResponse",
    "GlobalRetrieveRequest",
    "LocalRetrieveRequest",
    "NaiveRetrieveRequest",
    "MixRetrieveRequest",
    "RetrieveResponse",
    "BatchQueries",
    "GlobalBatchRequest",
    "LocalBatchRequest",
    "NaiveBatchRequest",
    "MixBatchRequest",
    "BatchSearchItem",
    "BatchSearchResponse",
    # graphs
    "GraphStatsResponse",
    "ModeAvailability",
    "GraphInfo",
    "GraphListResponse",
    "GraphDetail",
    "MAX_PAGE_LIMIT",
    "PageInfo",
    "EntityItem",
    "RelationItem",
    "EntityPage",
    "MAX_SELECT_IDS",
    "RelationSelectRequest",
    "RelationPage",
    "Neighborhood",
    "CommunityItem",
    "CommunityPage",
    "CommunityDetail",
    "ChunkItem",
    "ChunkPage",
    "ConsistencyIssueItem",
    "ConsistencyReportModel",
    # jobs
    "BuildRequest",
    "JobResponse",
    "JobListResponse",
    # service
    "HealthResponse",
    "OntologyResponse",
    # engine parameters
    "GlobalSearchParams",
    "LocalParams",
    "MixQueryParams",
    "NaiveSearchParams",
]
