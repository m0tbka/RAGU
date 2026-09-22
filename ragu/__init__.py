__version__ = "0.0.5"

# The public names below are resolved on first use rather than imported here.
# Importing them eagerly made every ``import ragu.<anything>`` pay for the whole
# library — chunkers, the graph, every engine, and through them fastembed,
# scikit-learn, pandas and nltk — even when the caller wanted one schema. The
# HTTP client in ``ragu.api.client`` took 3 seconds and 3 000 modules to import
# because of it. ``from ragu import KnowledgeGraph`` and ``ragu.KnowledgeGraph``
# still work exactly as before; they just load on the first touch.

from typing import TYPE_CHECKING

_EXPORTS: dict[str, str] = {
    # Default chunkers
    "SimpleChunker": "ragu.chunker",
    "SmartSemanticChunker": "ragu.chunker",
    # Knowledge Graph and builders
    "KnowledgeGraph": "ragu.graph.knowledge_graph",
    "InMemoryGraphBuilder": "ragu.graph.graph_builder_pipeline",
    "BuilderArguments": "ragu.graph.graph_builder_pipeline",
    "GraphRetriever": "ragu.graph.graph_retrieve_backend",
    "StorageArguments": "ragu.graph.index",
    # Global settings
    "Env": "ragu.common.env",
    "Settings": "ragu.common.global_parameters",
    # Model clients
    "CachedAsyncOpenAI": "ragu.models",
    "EmbedderOpenAI": "ragu.models",
    "LLMOpenAI": "ragu.models",
    # Search engines
    "LocalSearchEngine": "ragu.search_engine",
    "GlobalSearchEngine": "ragu.search_engine",
    "MixSearchEngine": "ragu.search_engine",
    "NaiveSearchEngine": "ragu.search_engine",
    "QueryPlanEngine": "ragu.search_engine",
    "SearchEngineStreamEvent": "ragu.search_engine",
    # Default extractors
    "ArtifactsExtractorLLM": "ragu.triplet",
    "TwoStageArtifactsExtractorLLM": "ragu.triplet",
    "RaguLmArtifactExtractor": "ragu.triplet",
}


def __getattr__(name: str):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'ragu' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


if TYPE_CHECKING:
    from ragu.chunker import SimpleChunker, SmartSemanticChunker
    from ragu.common.env import Env
    from ragu.common.global_parameters import Settings
    from ragu.graph.graph_builder_pipeline import BuilderArguments, InMemoryGraphBuilder
    from ragu.graph.graph_retrieve_backend import GraphRetriever
    from ragu.graph.index import StorageArguments
    from ragu.graph.knowledge_graph import KnowledgeGraph
    from ragu.models import CachedAsyncOpenAI, EmbedderOpenAI, LLMOpenAI
    from ragu.search_engine import (
        GlobalSearchEngine,
        LocalSearchEngine,
        MixSearchEngine,
        NaiveSearchEngine,
        QueryPlanEngine,
        SearchEngineStreamEvent,
    )
    from ragu.triplet import (
        ArtifactsExtractorLLM,
        RaguLmArtifactExtractor,
        TwoStageArtifactsExtractorLLM,
    )


__all__ = [
    "__version__",
    "KnowledgeGraph",
    "InMemoryGraphBuilder",
    "BuilderArguments",
    "GraphRetriever",
    "StorageArguments",
    "LocalSearchEngine",
    "GlobalSearchEngine",
    "MixSearchEngine",
    "NaiveSearchEngine",
    "QueryPlanEngine",
    "SearchEngineStreamEvent",
    "ArtifactsExtractorLLM",
    "TwoStageArtifactsExtractorLLM",
    "RaguLmArtifactExtractor",
    "Env",
    "Settings",
    "CachedAsyncOpenAI",
    "EmbedderOpenAI",
    "LLMOpenAI",
    "SimpleChunker",
    "SmartSemanticChunker",
]
