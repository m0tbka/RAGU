"""
Long-running operations: what starts one, and how it is reported.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
