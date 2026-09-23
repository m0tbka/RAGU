"""
What more than one route module needs: the graph catalogue and the backend a
request addresses.
"""

from fastapi import Request

from ragu.api.backends.base import SearchBackend
from ragu.api.errors import GraphNotFoundError, ServiceNotReadyError
from ragu.api.models import ErrorResponse

GRAPH_RESPONSES = {404: {"model": ErrorResponse, "description": "No such graph"}}


def registry_of(request: Request):
    """
    The graph catalogue, or ``None`` before the lifespan has run.
    """
    return getattr(request.app.state, "registry", None)


def configured_backend(request: Request, graph_id: str) -> SearchBackend:
    """
    The backend of a configured graph, loaded or not.

    :raises GraphNotFoundError: If no graph by that name is configured.
    """
    registry = registry_of(request)
    backend = registry.backend(graph_id) if registry is not None else None
    if backend is None:
        known = ", ".join(registry.ids) if registry is not None else "none"
        raise GraphNotFoundError(
            f"No graph named '{graph_id}'. Configured graphs: {known or 'none'}."
        )
    return backend


def get_backend(request: Request) -> SearchBackend:
    """
    Resolve the graph this request addresses.

    :raises GraphNotFoundError: If the path names a graph that is not configured.
    :raises ServiceNotReadyError: If that graph is not loaded.
    """
    registry = registry_of(request)
    if registry is None:
        raise ServiceNotReadyError("The service is still starting up.")
    return registry.resolve(request.path_params.get("graph_id"))
