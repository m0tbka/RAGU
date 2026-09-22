"""
Parameter objects of the search engines, and nothing else.

Kept apart from the engines so that a schema or a client which only needs the
parameters does not import what the engines run on — the graph, the storages,
the model clients, and through them fastembed, scikit-learn, pandas and nltk.
The engine modules re-export these names, so importing them from there keeps
working.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EngineParams:
    """
    Base class for engine parameters.

    Carries no fields of its own; concrete engines subclass it to declare only
    the options they actually use (e.g. ``top_k``, reranking limits, generation
    flags). Shared by ``search`` / ``batch_search`` and ``query`` /
    ``batch_query``.
    """


@dataclass
class GlobalSearchParams(EngineParams):
    """
    Per-query parameters for :class:`GlobalSearchEngine`.

    :param min_cluster_size: Minimum number of entities a community must
        contain for its summary to be evaluated. When ``1`` (the default),
        every stored community takes part in retrieval.
    """
    min_cluster_size: int = 1


@dataclass
class LocalParams(EngineParams):
    """
    Parameters for :class:`LocalSearchEngine`.

    :param top_k: Maximum number of entities to retrieve. (Retrieval-time.)
    :param rerank_top_k: After reranking the retrieved entities, keep only this
        many most-relevant entities before deriving relations, summaries and
        chunks. ``None`` keeps all entities. (Retrieval-time.)
    :param use_summary: Whether community summaries are included in the generated
        context. (Generation-time; ignored by :meth:`batch_search`.)
    :param use_chunks: Whether source chunks are included in the generated
        context. (Generation-time; ignored by :meth:`batch_search`.)
    """
    top_k: int = 20
    rerank_top_k: int | None = None
    use_summary: bool = False
    use_chunks: bool = True


@dataclass
class NaiveSearchParams(EngineParams):
    """
    Retrieval/query parameters for :class:`NaiveSearchEngine`.

    :param top_k: Number of chunks to retrieve.
    :param rerank_top_k: Number of chunks to keep after reranking. ``None`` keeps
        all reranked chunks. Used only when a reranker is configured.
    """
    top_k: int = 20
    rerank_top_k: Optional[int] = None


@dataclass
class MixQueryParams(EngineParams):
    """
    Query parameters for :class:`MixSearchEngine`.

    :param ensemble_responses: When ``True``, ensemble child-engine *answers*
        (via their ``batch_query``) instead of child-engine retrieval contexts.
    """
    ensemble_responses: bool = False
