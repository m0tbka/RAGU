"""
Search backends.
"""

from ragu.api.backends.base import GraphStats, SearchBackend, SearchOutcome
from ragu.api.backends.ragu_backend import RaguBackend
from ragu.api.backends.stub import StubBackend
from ragu.api.config import GraphSpec, ServiceSettings
from ragu.models.scorer import Scorer


def build_backend(
    settings: ServiceSettings,
    spec: GraphSpec,
    reranker: "Scorer | None" = None,
) -> SearchBackend:
    """
    Instantiate the backend selected by ``RAGU_API_BACKEND`` for one graph.

    :param settings: Service settings.
    :param spec: The graph this backend serves.
    :param reranker: Reranker to hand the engines, when the deployment has one.
    :return: The configured backend, not yet started.
    """
    if settings.backend == "stub":
        return StubBackend(settings, spec)
    return RaguBackend(settings, spec, reranker=reranker)


__all__ = [
    "GraphStats",
    "RaguBackend",
    "SearchBackend",
    "SearchOutcome",
    "StubBackend",
    "build_backend",
]
