"""
GraphView: the whole-graph passes behind the browse routes, and the lists
it keeps.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "fastapi", reason="install the 'api' extra to test the search service"
)

from tests.api.support import (
    make_backend,
)


class TestEntitySelection:
    """The pure function behind the entity listing, tested without a loop."""

    @staticmethod
    def _nodes():
        from ragu.graph.types import Entity

        return [
            Entity(entity_name="Борис", entity_type="PERSON", description="d",
                   source_chunk_id=[], clusters=[{"cluster_id": 1, "level": 0}]),
            Entity(entity_name="Анна", entity_type="PERSON", description="d",
                   source_chunk_id=[], clusters=[{"cluster_id": 2, "level": 0}]),
            Entity(entity_name="Ватикан", entity_type="ORG", description="d",
                   source_chunk_id=[], clusters=[]),
        ]

    def test_filters_compose(self):
        from ragu.api.backends.ragu_backend.graph_view import _select_entities

        nodes = self._nodes()
        assert [n.entity_name for n in _select_entities(
            nodes, "PERSON", None, None, None, "asc", None)] == ["Борис", "Анна"]
        assert [n.entity_name for n in _select_entities(
            nodes, None, "ан", None, None, "asc", None)] == ["Анна", "Ватикан"]
        assert [n.entity_name for n in _select_entities(
            nodes, None, None, "1", None, "asc", None)] == ["Борис"]

    def test_sorting_by_name_is_case_folded(self):
        from ragu.api.backends.ragu_backend.graph_view import _select_entities

        names = [n.entity_name for n in _select_entities(
            self._nodes(), None, None, None, "name", "asc", None)]
        assert names == ["Анна", "Борис", "Ватикан"]

    def test_sorting_by_degree_puts_the_busiest_first(self):
        from ragu.api.backends.ragu_backend.graph_view import _select_entities

        nodes = self._nodes()
        degrees = {nodes[0].id: 1, nodes[1].id: 9, nodes[2].id: 4}
        ordered = _select_entities(nodes, None, None, None, "degree", "desc", degrees)
        assert [n.entity_name for n in ordered] == ["Анна", "Ватикан", "Борис"]

    def test_counting_degrees_counts_both_ends(self):
        from ragu.api.backends.ragu_backend.graph_view import _count_degrees
        from ragu.graph.types import Relation

        edge = Relation(subject_id="a", object_id="b", subject_name="a",
                        object_name="b", relation_type="r", description="d",
                        relation_strength=1.0, source_chunk_id=[])
        assert _count_degrees([edge, edge]) == {"a": 2, "b": 2}


class TestRelationSelection:
    """
    Relations restricted to a set of entities.

    A canvas draws the induced subgraph of what it shows; edges to invisible
    nodes are a rendering bug, and paging every relation to filter on the client
    is a copy of the graph the client then has to keep fresh.
    """

    @staticmethod
    def _backend(entities=10, relations=30):
        from ragu.graph.types import Entity, Relation

        nodes = [
            Entity(entity_name=f"E{i}", entity_type="PERSON", description="d",
                   source_chunk_id=[], clusters=[])
            for i in range(entities)
        ]
        edges = [
            Relation(
                subject_id=nodes[i % entities].id,
                object_id=nodes[(i * 3 + 1) % entities].id,
                subject_name="a", object_name="b",
                # Distinct per index: the id is derived from the two ends plus
                # the type, and a repeated triple is the same relation.
                relation_type=f"rel-{i}",
                description="d", relation_strength=float(i % 5),
                source_chunk_id=[],
            )
            for i in range(relations)
        ]
        backend = make_backend()
        backend.graph = object()
        backend._llm = object()
        backend._embedder = object()
        backend.view._node_cache = nodes
        backend.view._edge_cache = edges
        return backend, nodes, edges

    async def test_induced_keeps_only_relations_with_both_ends_inside(self):
        backend, nodes, _ = self._backend()
        chosen = {nodes[0].id, nodes[1].id, nodes[2].id}

        total, page = await backend.select_relations(
            entity_ids=list(chosen), limit=500, offset=0
        )
        assert total == len(page)
        assert page, "the fixture should contain at least one induced relation"
        for relation in page:
            assert relation.subject_id in chosen and relation.object_id in chosen

    async def test_incident_is_a_superset_of_induced(self):
        backend, nodes, _ = self._backend()
        chosen = [nodes[0].id, nodes[1].id, nodes[2].id]

        _, induced = await backend.select_relations(
            entity_ids=chosen, edge_scope="induced", limit=500, offset=0
        )
        _, incident = await backend.select_relations(
            entity_ids=chosen, edge_scope="incident", limit=500, offset=0
        )
        assert {r.id for r in induced} <= {r.id for r in incident}
        assert len(incident) > len(induced)

    async def test_incident_keeps_a_relation_with_one_end_inside(self):
        backend, nodes, _ = self._backend()
        total, page = await backend.select_relations(
            entity_ids=[nodes[0].id], edge_scope="incident", limit=500, offset=0
        )
        assert total > 0
        for relation in page:
            assert nodes[0].id in (relation.subject_id, relation.object_id)

    async def test_paging_covers_the_filtered_set_exactly_once(self):
        # The acceptance the consumer asked for: pages sum to the whole result,
        # with no duplicate and no gap.
        backend, nodes, _ = self._backend()
        chosen = [n.id for n in nodes[:6]]

        total, whole = await backend.select_relations(
            entity_ids=chosen, edge_scope="incident", limit=500, offset=0
        )
        collected = []
        for offset in range(0, total, 4):
            _, page = await backend.select_relations(
                entity_ids=chosen, edge_scope="incident", limit=4, offset=offset
            )
            collected.extend(r.id for r in page)

        assert collected == [r.id for r in whole]
        assert len(collected) == len(set(collected)) == total

    async def test_an_unknown_id_is_skipped_not_fatal(self):
        backend, nodes, _ = self._backend()
        total, _ = await backend.select_relations(
            entity_ids=[nodes[0].id, nodes[1].id, "ent-нет-такого"],
            edge_scope="incident", limit=500, offset=0,
        )
        assert total > 0

    async def test_nothing_matching_is_an_empty_page_not_an_error(self):
        backend, _, _ = self._backend()
        total, page = await backend.select_relations(
            entity_ids=["ent-нет-такого"], limit=500, offset=0
        )
        assert (total, page) == (0, [])

    async def test_min_strength_applies_to_the_selection(self):
        backend, nodes, _ = self._backend()
        chosen = [n.id for n in nodes]
        _, strong = await backend.select_relations(
            entity_ids=chosen, min_strength=4.0, limit=500, offset=0
        )
        assert strong
        for relation in strong:
            assert relation.relation_strength >= 4.0

    async def test_the_membership_pass_leaves_the_event_loop(self, monkeypatch):
        import threading

        import ragu.api.backends.ragu_backend.graph_view as module

        backend, nodes, _ = self._backend()
        loop_thread = threading.get_ident()
        ran_on = []
        original = module._select_relations

        def spy(*args):
            ran_on.append(threading.get_ident())
            return original(*args)

        monkeypatch.setattr(module, "_select_relations", spy)
        await backend.select_relations(entity_ids=[nodes[0].id], limit=10, offset=0)

        assert ran_on and loop_thread not in ran_on


class TestGraphCacheInvalidation:
    """A build changes the graph; the materialized lists must not survive it."""

    async def test_building_drops_the_materialized_lists(self):

        backend = make_backend()
        backend.graph = object()
        backend._llm = object()
        backend._embedder = object()
        view = backend.view
        view._node_cache = ["stale"]
        view._edge_cache = ["stale"]

        backend._drop_graph_cache()

        assert view._node_cache is None
        assert view._edge_cache is None

    async def test_entities_are_materialized_once(self):
        calls = []
        backend = make_backend()

        class FakeGraphBackend:
            async def get_all_nodes(self):
                calls.append(1)
                return []

        class FakeIndex:
            graph_backend = FakeGraphBackend()

        class FakeGraph:
            index = FakeIndex()

        backend.graph = FakeGraph()
        backend._llm = object()
        backend._embedder = object()

        await backend.view.nodes()
        await backend.view.nodes()

        assert calls == [1]


class TestGraphCacheCeiling:
    """Paging must not pin an arbitrarily large graph in memory."""

    async def test_a_graph_over_the_ceiling_is_not_kept(self):
        backend = make_backend(graph_cache_max_items=2)
        calls = []

        class FakeGraphBackend:
            async def get_all_nodes(self):
                calls.append(1)
                return ["a", "b", "c"]

        class FakeIndex:
            graph_backend = FakeGraphBackend()

        class FakeGraph:
            index = FakeIndex()

        backend.graph = FakeGraph()
        backend._llm = object()
        backend._embedder = object()

        await backend.view.nodes()
        await backend.view.nodes()

        assert calls == [1, 1]
        assert backend.view._node_cache is None

    async def test_a_graph_within_the_ceiling_is_kept(self):
        backend = make_backend(graph_cache_max_items=10)
        calls = []

        class FakeGraphBackend:
            async def get_all_nodes(self):
                calls.append(1)
                return ["a", "b"]

        class FakeIndex:
            graph_backend = FakeGraphBackend()

        class FakeGraph:
            index = FakeIndex()

        backend.graph = FakeGraph()
        backend._llm = object()
        backend._embedder = object()

        await backend.view.nodes()
        await backend.view.nodes()

        assert calls == [1]


class TestEventLoopIsNotBlocked:
    """
    Whole-corpus passes run off the event loop.

    The service is async so that it can keep answering while it waits on the
    models. Walking every entity and every relation is not waiting — it is pure
    CPU, and on the loop it is time in which nothing is served, /health least of
    all. These pin the work to a worker thread by identity, not by timing, so
    they do not flake on a loaded machine.
    """

    @staticmethod
    def _graph_backend(entities=8, relations=12):
        from ragu.graph.types import Entity, Relation

        nodes = [
            Entity(
                entity_name=f"Сущность {i}",
                entity_type="PERSON" if i % 2 else "ORG",
                description="d",
                source_chunk_id=[],
                clusters=[{"cluster_id": i % 3, "level": 0}],
            )
            for i in range(entities)
        ]
        edges = [
            Relation(
                subject_id=nodes[i % entities].id,
                object_id=nodes[(i * 3) % entities].id,
                subject_name="a",
                object_name="b",
                relation_type="rel",
                description="d",
                relation_strength=1.0,
                source_chunk_id=[],
            )
            for i in range(relations)
        ]
        backend = make_backend()
        backend.graph = object()
        backend._llm = object()
        backend._embedder = object()
        backend.view._node_cache = nodes
        backend.view._edge_cache = edges
        return backend, nodes, edges

    async def test_filtering_and_sorting_leave_the_loop(self, monkeypatch):
        import threading

        backend, _, _ = self._graph_backend()
        loop_thread = threading.get_ident()
        ran_on = []

        import ragu.api.backends.ragu_backend.graph_view as module

        original = module._select_entities

        def spy(*args, **kwargs):
            ran_on.append(threading.get_ident())
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "_select_entities", spy)
        await backend.list_entities(limit=5, offset=0, sort="name")

        assert ran_on, "_select_entities was not called at all"
        assert loop_thread not in ran_on, "the sort ran on the event loop"

    async def test_counting_degrees_leaves_the_loop(self, monkeypatch):
        import threading

        backend, _, _ = self._graph_backend()
        loop_thread = threading.get_ident()
        ran_on = []

        import ragu.api.backends.ragu_backend.graph_view as module

        original = module._count_degrees

        def spy(edges):
            ran_on.append(threading.get_ident())
            return original(edges)

        monkeypatch.setattr(module, "_count_degrees", spy)
        await backend.view.degrees()

        assert ran_on and loop_thread not in ran_on

    async def test_plain_paging_does_not_pay_for_a_thread(self, monkeypatch):
        # Nothing to filter and nothing to sort: the hop would cost a context
        # switch and buy nothing.
        backend, nodes, _ = self._graph_backend()
        called = []

        import ragu.api.backends.ragu_backend.graph_view as module

        monkeypatch.setattr(
            module, "_select_entities", lambda *a, **k: called.append(1) or []
        )
        total, page = await backend.list_entities(limit=3, offset=0)

        assert not called
        assert total == len(nodes)
        assert len(page) == 3

    async def test_the_degree_map_is_built_once(self):
        backend, _, edges = self._graph_backend()
        first = await backend.view.degrees()
        second = await backend.view.degrees()
        assert first is second

        backend._drop_graph_cache()
        assert backend.view._degree_cache is None
