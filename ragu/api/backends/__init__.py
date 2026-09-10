"""
Search backends.
"""

from ragu.api.backends.base import GraphStats, SearchBackend, SearchOutcome
from ragu.api.backends.ragu_backend import RaguBackend
from ragu.api.backends.stub import StubBackend
from ragu.api.config import GraphSpec, ServiceSettings


def build_backend(settings: ServiceSettings, spec: GraphSpec) -> SearchBackend:
    """
    Instantiate the backend selected by ``RAGU_API_BACKEND`` for one graph.

    :param settings: Service settings.
    :param spec: The graph this backend serves.
    :return: The configured backend, not yet started.
    """
    if settings.backend == "stub":
        return StubBackend(settings, spec)
    return RaguBackend(settings, spec)


__all__ = [
    "GraphStats",
    "RaguBackend",
    "SearchBackend",
    "SearchOutcome",
    "StubBackend",
    "build_backend",
]
