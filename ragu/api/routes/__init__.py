"""
HTTP routes, one module per resource.

``service`` health and metrics; ``graphs`` the catalogue; ``browse`` a graph's
contents; ``jobs`` ingestion and reindexing; ``search`` the four modes. This
package only assembles them into the one router the application mounts.
"""

from fastapi import APIRouter

from ragu.api.routes import browse, graphs, jobs, search, service

router = APIRouter()
router.include_router(service.router)
router.include_router(graphs.router)
router.include_router(jobs.router)
router.include_router(browse.router)

# The catalogue path is canonical; the flat one is kept for clients written
# before the service served more than one graph.
router.include_router(
    search.search_router, prefix="/v1/graphs/{graph_id}", tags=["search"]
)
router.include_router(
    search.search_router, prefix="/v1", tags=["search"], deprecated=True
)

__all__ = ["router"]
