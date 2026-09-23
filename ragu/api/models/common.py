"""
What every other schema module shares: the search modes, the capabilities a
graph may have, and the error envelope.
"""

from typing import Literal, get_args

from pydantic import BaseModel, Field


SearchMode = Literal["global", "local", "naive", "mix"]


# What a search mode needs the graph to hold. Named here rather than in the
# backends so that the configuration layer can validate against the same set.
Capability = Literal["entity_graph", "community_summaries", "vector_index"]

CAPABILITIES: frozenset[str] = frozenset(get_args(Capability))


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
