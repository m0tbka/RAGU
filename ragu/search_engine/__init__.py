# Resolved on first use, like the top-level package: importing one engine's
# parameters (``ragu.search_engine.params``) must not import every engine.

from typing import TYPE_CHECKING

_EXPORTS: dict[str, str] = {
    "GlobalSearchEngine": "ragu.search_engine.global_search",
    "GlobalSearchRetrieve": "ragu.search_engine.global_search",
    "SearchEngineStreamEvent": "ragu.search_engine.base_engine",
    "LocalSearchEngine": "ragu.search_engine.local_search",
    "LocalSearchRetrieve": "ragu.search_engine.local_search",
    "MixSearchEngine": "ragu.search_engine.mix_search",
    "MixSearchRetrieve": "ragu.search_engine.mix_search",
    "NaiveSearchEngine": "ragu.search_engine.naive_search",
    "NaiveSearchRetrieve": "ragu.search_engine.naive_search",
    "QueryPlanEngine": "ragu.search_engine.query_plan",
}


def __getattr__(name: str):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'ragu.search_engine' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


if TYPE_CHECKING:
    from ragu.search_engine.base_engine import SearchEngineStreamEvent
    from ragu.search_engine.global_search import GlobalSearchEngine, GlobalSearchRetrieve
    from ragu.search_engine.local_search import LocalSearchEngine, LocalSearchRetrieve
    from ragu.search_engine.mix_search import MixSearchEngine, MixSearchRetrieve
    from ragu.search_engine.naive_search import NaiveSearchEngine, NaiveSearchRetrieve
    from ragu.search_engine.query_plan import QueryPlanEngine


# Spelled out rather than derived from ``_EXPORTS``: linters and IDEs read
# ``__all__`` statically, and a computed one hides every name above from them.
__all__ = [
    "GlobalSearchEngine",
    "GlobalSearchRetrieve",
    "SearchEngineStreamEvent",
    "LocalSearchEngine",
    "LocalSearchRetrieve",
    "MixSearchEngine",
    "MixSearchRetrieve",
    "NaiveSearchEngine",
    "NaiveSearchRetrieve",
    "QueryPlanEngine",
]
