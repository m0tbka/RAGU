"""
Writing to a graph: building it from documents, and reindexing it in place.
"""

import asyncio
from typing import Any

from ragu import ArtifactsExtractorLLM, BuilderArguments, Settings, SimpleChunker
from ragu.api.backends.capabilities import GraphStats
from ragu.api.backends.ragu_backend.settings import exclusive_settings
from ragu.api.config import GraphSpec
from ragu.api.errors import InvalidRequestError
from ragu.common.logger import logger
from ragu.models.embedder import Embedder
from ragu.models.llm import LLM


class Ingestion:
    """
    The writing part of ``RaguBackend``, kept in its own module.

    Both writes hold the backend's write lock, and reads on this graph are
    refused for as long as one runs: the file-backed stores tolerate no
    concurrent access. Afterwards the materialized lists are dropped and the
    stores re-measured, so capabilities follow what the graph now holds.
    """

    # Provided by RaguBackend.
    spec: GraphSpec
    graph_id: str
    language: str
    _stats: GraphStats | None
    _write_lock: asyncio.Lock
    _building: bool

    def _pipeline(self, llm: LLM, embedder: Embedder) -> dict[str, Any]:
        """
        The chunker, extractor and builder settings this graph is built with.

        A graph that only serves gets none of them: constructing an extractor
        costs nothing at rest, but it is configuration the deployment has not
        asked for.

        :param llm: LLM the extractor uses.
        :param embedder: Embedder the extractor may use for ICL examples.
        :return: Keyword arguments for ``KnowledgeGraph``.
        """
        build = self.spec.build
        if not build.enabled:
            return {}

        chunker = (
            SimpleChunker(
                max_chunk_size=build.chunk_size, overlap=build.chunk_overlap
            )
            if build.chunker == "simple"
            else None
        )
        extractor = (
            None if build.vector_only else ArtifactsExtractorLLM(llm, embedder=embedder)
        )
        return {
            "chunker": chunker,
            "artifact_extractor": extractor,
            "builder_settings": BuilderArguments(
                build_only_vector_context=build.vector_only,
                make_community_summary=build.make_community_summary,
            ),
        }

    @property
    def accepts_documents(self) -> bool:
        return self.spec.build.enabled

    async def build(self, documents: list[str]) -> dict[str, Any]:
        """
        Add documents to this graph and re-measure it.

        Searches on this graph are refused while the build runs: it writes into
        the same stores they read, and the file-backed ones tolerate no
        concurrent access.

        :param documents: Raw document texts.
        :return: The graph sizes after the build.
        :raises InvalidRequestError: If this graph does not accept documents.
        """
        if not self.accepts_documents:
            return await super().build(documents)

        graph, _, _ = self._require_loaded()
        async with self._write_lock:
            self._building = True
            try:
                async with exclusive_settings():
                    if self.spec.settings_file:
                        Settings.load(self.spec.settings_file)
                    Settings.storage_folder = self.spec.storage_folder
                    Settings.language = self.language
                    await graph.build_from_docs(documents)
                self._drop_graph_cache()
                self._stats = await self._measure(graph)
            finally:
                self._building = False

        logger.info(
            "Graph '{}' rebuilt from {} document(s): {}",
            self.graph_id,
            len(documents),
            self._stats,
        )
        return {"documents": len(documents), "stats": self._stats.to_response().model_dump()}

    async def reindex(self, kind: str) -> dict[str, Any]:
        """
        Rebuild communities, descriptions or the whole graph.

        Held under the same write lock as ingestion, and it invalidates the
        materialized node and edge lists.
        """
        graph, _, _ = self._require_loaded()
        operations = {
            "community": graph.reindex_community,
            "descriptions": graph.reindex_descriptions,
            "graph": graph.reindex_graph,
        }
        operation = operations.get(kind)
        if operation is None:
            raise InvalidRequestError(
                f"Unknown reindex '{kind}'. Expected one of {sorted(operations)}."
            )

        async with self._write_lock:
            self._building = True
            try:
                async with exclusive_settings():
                    if self.spec.settings_file:
                        Settings.load(self.spec.settings_file)
                    Settings.storage_folder = self.spec.storage_folder
                    Settings.language = self.language
                    await operation()
                self._drop_graph_cache()
                self._stats = await self._measure(graph)
            finally:
                self._building = False

        return {"reindex": kind, "stats": self._stats.to_response().model_dump()}
