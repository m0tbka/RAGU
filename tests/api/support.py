"""
Helpers shared by the service tests: clients, backends and test data.

Not a test module, and it imports FastAPI unconditionally: every test module
calls ``pytest.importorskip("fastapi")`` before importing from here, so without
the ``api`` extra they skip before this file is ever loaded.
"""

from __future__ import annotations

import asyncio
import time

from fastapi.testclient import TestClient

from ragu.api.app import create_app
from ragu.api.config import ServiceSettings
from ragu.chunker.types import Chunk
from ragu.graph.types import Entity
from ragu.search_engine.naive_search import (
    NaiveSearchResult,
    NaiveSearchRetrieve,
)


def build_client(missing: str = "", **overrides) -> TestClient:
    settings = ServiceSettings(
        backend="stub", stub_missing_capabilities=missing, **overrides
    )
    return TestClient(create_app(settings))


def make_chunk(content: str, index: int = 0) -> Chunk:
    return Chunk(content=content, chunk_order_idx=index, doc_id="doc-1")


def make_entity(name: str) -> Entity:
    return Entity(
        entity_name=name,
        entity_type="Person",
        description=f"{name} description",
        source_chunk_id=["c1"],
    )


def make_backend(**overrides):
    """A RaguBackend for the single graph the flat settings describe."""
    from ragu.api.backends.ragu_backend import RaguBackend

    settings = ServiceSettings(backend="ragu", **overrides)
    return RaguBackend(settings, settings.resolved_graphs()[0])


def make_retrieval(*contents: str) -> NaiveSearchRetrieve:
    return NaiveSearchRetrieve(
        query="sub",
        result=NaiveSearchResult(chunks=[make_chunk(c) for c in contents]),
    )


def graphs_client(specs, **overrides) -> TestClient:
    settings = ServiceSettings(backend="stub", graphs=specs, **overrides)
    return TestClient(create_app(settings))


def ingest_client(**build) -> TestClient:
    settings = ServiceSettings(
        backend="stub",
        graphs=[
            {"id": "corpus", "storage_folder": "a", "build": {"enabled": True, **build}},
            {"id": "readonly", "storage_folder": "b"},
        ],
    )
    return TestClient(create_app(settings))


class LifespanRunner:
    """Runs an app's lifespan around a block, without a TestClient."""

    def __init__(self, app):
        self.app = app

    async def __aenter__(self):
        from contextlib import AsyncExitStack

        self._stack = AsyncExitStack()
        await self._stack.enter_async_context(
            self.app.router.lifespan_context(self.app)
        )
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()


def asgi_client(app, **kwargs):
    """A RaguClient driving an app in-process, with no socket in between."""
    import httpx

    from ragu.api.client import RaguClient

    kwargs.setdefault("graph", "default")
    return RaguClient(
        "http://service",
        transport=httpx.ASGITransport(app=app),
        **kwargs,
    )


class SearchOutcomeStub:
    """Carries just the engine report, which is all _merged_report reads."""

    def __init__(self, engines):
        self.engines = engines


async def pause(seconds: float) -> None:
    """
    ``asyncio.sleep`` that lasts at least ``seconds`` of real time.

    The event loop wakes a timer as soon as it is due within the clock's
    resolution, and on Windows that resolution is 15.6 ms: a 20 ms sleep can
    return after five. The timing tests measure with ``perf_counter``, so they
    sleep until it agrees — still yielding to the loop, so concurrent pauses
    overlap exactly as the calls they stand for would.
    """
    deadline = time.perf_counter() + seconds
    while (left := deadline - time.perf_counter()) > 0:
        await asyncio.sleep(left)
