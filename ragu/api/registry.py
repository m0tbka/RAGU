"""
The catalogue of graphs this service serves.

One process can hold several graphs even though ``Settings`` is a process-wide
singleton: every per-graph value is read inside the constructors that run while
that graph is being built — ``Index`` reads the storage folder, the embedder its
token limit, the engines their tokenizer — and nothing re-reads the singleton
afterwards. The registry therefore builds graphs one at a time and rolls the
singleton back after each, and the backend captures whatever it still needs.
"""

import asyncio
from collections.abc import Callable

from ragu.api.backends.base import SearchBackend
from ragu.api.config import GraphSpec, ServiceSettings
from ragu.api.errors import GraphNotFoundError, ServiceNotReadyError
from ragu.common.logger import logger

from ragu.models.scorer import Scorer

BackendFactory = Callable[[ServiceSettings, GraphSpec], SearchBackend]


class GraphRegistry:
    """
    Holds one backend per configured graph, and which of them failed to load.
    """

    def __init__(
        self,
        settings: ServiceSettings,
        factory: BackendFactory | None = None,
        reranker: Scorer | None = None,
    ):
        """
        :param settings: Service settings; its ``resolved_graphs`` names the graphs.
        :param factory: Builds a backend for one spec. Injected by tests.
        :param reranker: Reranker shared by every graph. Supplied by the caller
            because the model runs outside this process.
        """
        from ragu.api.backends import build_backend

        self._settings = settings
        self._factory = factory or (
            lambda service_settings, spec: build_backend(
                service_settings, spec, reranker=reranker
            )
        )
        self._specs: dict[str, GraphSpec] = {
            spec.id: spec for spec in settings.resolved_graphs()
        }
        self._backends: dict[str, SearchBackend] = {}
        self._errors: dict[str, str] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def of(cls, settings: ServiceSettings, backend: SearchBackend) -> "GraphRegistry":
        """
        Wrap one already-built backend, for tests and in-process embedding.

        :param settings: Service settings.
        :param backend: The backend to serve.
        :return: A registry holding just that backend.
        """
        registry = cls(settings)
        registry._specs = {}
        registry._backends = {getattr(backend, "graph_id", "default"): backend}
        return registry

    @property
    def ids(self) -> list[str]:
        """
        Every configured graph id, in configuration order.
        """
        return list(self._specs) or list(self._backends)

    @property
    def default_id(self) -> str:
        """
        The graph served by the paths that name no graph.
        """
        return self.ids[0]

    def error(self, graph_id: str) -> str | None:
        """
        Why this graph is not loaded, when it is not.
        """
        return self._errors.get(graph_id)

    def backend(self, graph_id: str) -> SearchBackend | None:
        """
        The backend for a graph, loaded or not, or ``None`` if there is no such graph.
        """
        return self._backends.get(graph_id)

    @property
    def any_loaded(self) -> bool:
        """
        Whether at least one graph can answer searches.
        """
        return any(backend.graph_loaded for backend in self._backends.values())

    def resolve(self, graph_id: str | None) -> SearchBackend:
        """
        Resolve the backend for a request.

        :param graph_id: Graph named in the path, or ``None`` for the default.
        :return: A backend that can serve searches.
        :raises GraphNotFoundError: If no such graph is configured.
        :raises ServiceNotReadyError: If the graph is configured but not loaded.
        """
        name = graph_id or self.default_id
        backend = self._backends.get(name)
        if backend is None:
            raise GraphNotFoundError(
                f"No graph named '{name}'. Configured graphs: {', '.join(self.ids) or 'none'}."
            )
        if not backend.graph_loaded:
            reason = self._errors.get(name)
            raise ServiceNotReadyError(
                f"Graph '{name}' is not loaded: {reason}"
                if reason
                else f"Graph '{name}' is not loaded yet."
            )
        return backend

    async def startup(self) -> None:
        """
        Build every configured graph, one at a time.

        A graph that fails to load is recorded and skipped rather than taking
        the service down: the others stay servable and ``/v1/graphs`` reports
        why this one is missing.
        """
        async with self._lock:
            for spec in self._specs.values():
                self._backends[spec.id] = self._factory(self._settings, spec)
            # Sequential, not gathered: the graphs are built through a global
            # settings singleton, one at a time.
            for graph_id, backend in self._backends.items():
                try:
                    await backend.startup()
                except Exception as exc:
                    self._errors[graph_id] = str(exc)
                    logger.opt(exception=True).error(
                        "Graph '{}' failed to load: {}", graph_id, exc
                    )

    async def shutdown(self) -> None:
        """
        Release every backend, whether or not it finished loading.
        """
        for graph_id, backend in self._backends.items():
            try:
                await backend.shutdown()
            except Exception:
                logger.exception("Graph '{}' failed to shut down", graph_id)
