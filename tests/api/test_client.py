"""
RaguClient, driven against the service in process.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "fastapi", reason="install the 'api' extra to test the search service"
)

from ragu.api.app import create_app
from ragu.api.config import ServiceSettings
from tests.api.support import (
    LifespanRunner,
    asgi_client,
)


class TestClientLibrary:
    """Consumers should not each hand-roll the same calls and the same parsing."""

    async def test_the_client_covers_the_search_shapes(self):
        app = create_app(ServiceSettings(backend="stub"))
        async with LifespanRunner(app):
            async with asgi_client(app) as client:
                answer = await client.search("naive", "q", params={"top_k": 2})
                context = await client.retrieve("naive", "q")
                batch = await client.batch("naive", ["a", "b"])

        assert answer.answer.startswith("[stub naive]")
        assert answer.engines.requested == "naive"
        assert answer.usage is not None
        assert len(context.sources) > 0
        assert [item.query for item in batch.results] == ["a", "b"]

    async def test_the_client_streams(self):
        app = create_app(ServiceSettings(backend="stub"))
        async with LifespanRunner(app):
            async with asgi_client(app) as client:
                events = [name async for name, _ in client.stream("naive", "hello")]

        assert events[0] == "meta"
        assert events[-1] == "done"

    async def test_the_client_reads_the_catalogue_and_the_graph(self):
        app = create_app(ServiceSettings(backend="stub"))
        async with LifespanRunner(app):
            async with asgi_client(app) as client:
                catalogue = await client.graphs()
                stats = await client.stats()
                modes = await client.capabilities()
                entities = await client.entities(limit=1)
                chunk = await client.chunk("chunk_1")
                ontology = await client.ontology()

        assert catalogue.default == "default"
        assert stats.embedding_dim == 8
        assert all(mode.available for mode in modes)
        assert entities.page.total == 2
        assert chunk.content == "stub chunk"
        assert ontology.entity_types

    async def test_an_error_envelope_becomes_a_typed_exception(self):
        # A caller branches on `code`, not on the text of a message.
        from ragu.api.client import RaguApiError

        settings = ServiceSettings(
            backend="stub", stub_missing_capabilities="entity_graph"
        )
        app = create_app(settings)
        async with LifespanRunner(app):
            async with asgi_client(app) as client:
                with pytest.raises(RaguApiError) as failure:
                    await client.search("local", "q")

        assert failure.value.status_code == 409
        assert failure.value.code == "CAPABILITY_UNAVAILABLE"
        assert failure.value.missing_capability == "entity_graph"
        assert failure.value.request_id

    async def test_the_client_sends_its_key(self):
        from ragu.api.client import RaguApiError

        app = create_app(ServiceSettings(backend="stub", api_keys="secret"))
        async with LifespanRunner(app):
            async with asgi_client(app) as anonymous:
                with pytest.raises(RaguApiError) as failure:
                    await anonymous.search("naive", "q")
            async with asgi_client(app, api_key="secret") as authorised:
                answer = await authorised.search("naive", "q")

        assert failure.value.status_code == 401
        assert answer.answer

    async def test_the_client_drives_ingestion_and_jobs(self):
        app = create_app(
            ServiceSettings(
                backend="stub",
                graphs=[
                    {"id": "corpus", "storage_folder": "a", "build": {"enabled": True}}
                ],
            )
        )
        async with LifespanRunner(app):
            async with asgi_client(app, graph="corpus") as client:
                job = await client.add_documents(["a"], idempotency_key="k1")
                again = await client.add_documents(["a"], idempotency_key="k1")
                listed = await client.jobs()
                fetched = await client.job(job.id)

        assert job.id == again.id
        assert len(listed.jobs) == 1
        assert fetched.kind == "build"


class TestClientCaughtUp:
    """
    The client covers every read the service offers.

    A consumer that has to reach into ``_get`` and ``_post`` to call a route is
    maintaining a second client, and it drifts from this one on the next change.
    """

    @staticmethod
    def _client():
        import httpx

        from ragu.api.client import RaguClient

        app = create_app(ServiceSettings(backend="stub"))
        return app, RaguClient(
            "http://ragu", graph="default", transport=httpx.ASGITransport(app=app)
        )

    async def test_one_entity(self):
        app, client = self._client()
        async with LifespanRunner(app), client:
            entity = await client.entity("entity_1")
        assert entity.name == "Сенкевич"

    async def test_entities_accept_sorting_filtering_and_ids(self):
        app, client = self._client()
        async with LifespanRunner(app), client:
            by_degree = await client.entities(sort="degree", order="desc")
            in_community = await client.entities(community_id="0")
            by_id = await client.entities(ids=["entity_2", "entity_1"])
        assert len(by_degree.entities) == 2
        assert in_community.entities
        assert [e.id for e in by_id.entities] == ["entity_2", "entity_1"]

    async def test_chunks_collection(self):
        app, client = self._client()
        async with LifespanRunner(app), client:
            page = await client.chunks(ids=["chunk_2", "nope"])
        assert [c.id for c in page.chunks] == ["chunk_2"]

    async def test_communities_by_id(self):
        app, client = self._client()
        async with LifespanRunner(app), client:
            page = await client.communities(ids=["com-1"])
        assert [c.id for c in page.communities] == ["com-1"]

    async def test_select_relations(self):
        app, client = self._client()
        async with LifespanRunner(app), client:
            induced = await client.select_relations(["entity_1", "entity_2"])
            incident = await client.select_relations(["entity_1"], edge_scope="incident")
        assert [r.id for r in induced.relations] == ["relation_1"]
        assert [r.id for r in incident.relations] == ["relation_1"]

    def test_every_graph_read_route_has_a_client_method(self):
        # The thing that actually went wrong: routes were added and the client
        # was not. This fails the next time that happens.
        from ragu.api.client import RaguClient

        expected = {
            "entities", "entity", "relations", "select_relations", "neighbors",
            "communities", "community", "chunks", "chunk", "consistency",
        }
        assert expected <= set(dir(RaguClient))
