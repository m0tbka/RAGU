"""
The service's own endpoints: health, and the ontology the extractors use.
"""

from pydantic import BaseModel, Field

from ragu.api.models.graphs import GraphStatsResponse


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when searches can be served, else 'degraded'")
    graph_loaded: bool
    stats: GraphStatsResponse | None = Field(
        default=None, description="Graph sizes; absent until the graph is loaded"
    )
    error: str | None = Field(
        default=None, description="Why the backend is not ready, when it is not"
    )


class OntologyResponse(BaseModel):
    """
    The entity and relation types the extractors work with.
    """

    entity_types: list[str] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)
