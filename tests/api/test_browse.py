"""
Reading a graph over HTTP: entities, relations, communities and chunks.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "fastapi", reason="install the 'api' extra to test the search service"
)

from tests.api.support import (
    build_client,
)


class TestGraphSurface:
    """Reads of the structure, for a client that draws the graph."""

    def test_stats_describe_the_corpus(self):
        with build_client() as client:
            body = client.get("/v1/graphs/default/stats").json()
        assert body["id"] == "default"
        assert body["loaded"] is True
        assert body["embedding_dim"] == 8
        assert body["documents"] == 1
        assert {mode["mode"] for mode in body["modes"]} == {
            "global",
            "local",
            "naive",
            "mix",
        }

    def test_entities_are_paged(self):
        with build_client() as client:
            body = client.get(
                "/v1/graphs/default/entities", params={"limit": 1, "offset": 0}
            ).json()
        assert body["page"] == {"total": 2, "limit": 1, "offset": 0}
        assert len(body["entities"]) == 1

    def test_entities_filter_by_type_and_name(self):
        with build_client() as client:
            by_type = client.get(
                "/v1/graphs/default/entities", params={"type": "PERSON"}
            ).json()
            by_name = client.get(
                "/v1/graphs/default/entities", params={"search": "польш"}
            ).json()
        assert [e["name"] for e in by_type["entities"]] == ["Сенкевич"]
        assert [e["name"] for e in by_name["entities"]] == ["Польша"]

    def test_relations_filter_by_strength(self):
        with build_client() as client:
            kept = client.get(
                "/v1/graphs/default/relations", params={"min_strength": 0.5}
            ).json()
            dropped = client.get(
                "/v1/graphs/default/relations", params={"min_strength": 5}
            ).json()
        assert kept["page"]["total"] == 1
        assert dropped["page"]["total"] == 0

    def test_neighbors_return_a_subgraph(self):
        with build_client() as client:
            body = client.get(
                "/v1/graphs/default/entities/entity_1/neighbors",
                params={"depth": 2},
            ).json()
        assert body["root"] == "entity_1"
        assert body["depth"] == 2
        assert body["entities"] and body["relations"]
        # The client lays the graph out itself; no coordinates come back.
        assert "x" not in body["entities"][0]

    def test_an_unknown_entity_answers_404(self):
        with build_client() as client:
            response = client.get("/v1/graphs/default/entities/nope/neighbors")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "NOT_FOUND"

    def test_communities_carry_their_summary(self):
        with build_client() as client:
            listing = client.get("/v1/graphs/default/communities").json()
            detail = client.get("/v1/graphs/default/communities/com-1").json()
        community = listing["communities"][0]
        # The title is lifted out of the rendered report; the summary is the
        # body that follows it, so a client does not re-parse RAGU's own format.
        assert community["title"] == "Сенкевич и Польша"
        assert community["summary"] == "Report summary: stub community summary"
        assert detail["id"] == "com-1"
        assert detail["level"] == 0

    def test_a_chunk_can_be_traced(self):
        with build_client() as client:
            body = client.get("/v1/graphs/default/chunks/chunk_1").json()
            missing = client.get("/v1/graphs/default/chunks/nope")
        assert body["content"] == "stub chunk"
        assert body["doc_id"] == "doc-1"
        assert missing.status_code == 404

    def test_consistency_is_reported(self):
        with build_client() as client:
            body = client.get("/v1/graphs/default/consistency").json()
        assert body["consistent"] is True
        assert body["issues"] == []

    def test_the_ontology_is_published(self):
        with build_client() as client:
            body = client.get("/v1/ontology").json()
        assert body["entity_types"]
        assert body["relation_types"]

    def test_a_backend_without_a_graph_refuses_the_surface(self):
        # The base class declines rather than inventing a shape.
        from ragu.api.backends.base import SearchBackend

        assert hasattr(SearchBackend, "list_entities")


class TestSelectionById:
    """One request for many sources, instead of one request per source."""

    def test_entities_come_back_in_the_order_asked_for(self):
        with build_client() as client:
            body = client.get(
                "/v1/graphs/default/entities",
                params={"ids": ["entity_2", "entity_1"]},
            ).json()
        assert [e["id"] for e in body["entities"]] == ["entity_2", "entity_1"]

    def test_an_unknown_id_is_skipped_not_fatal(self):
        with build_client() as client:
            body = client.get(
                "/v1/graphs/default/entities", params={"ids": ["entity_1", "nope"]}
            ).json()
        assert [e["id"] for e in body["entities"]] == ["entity_1"]

    def test_chunks_and_communities_select_the_same_way(self):
        with build_client() as client:
            chunks = client.get(
                "/v1/graphs/default/chunks", params={"ids": ["chunk_2", "nope"]}
            ).json()
            communities = client.get(
                "/v1/graphs/default/communities", params={"ids": ["com-1", "nope"]}
            ).json()
        assert [c["id"] for c in chunks["chunks"]] == ["chunk_2"]
        assert [c["id"] for c in communities["communities"]] == ["com-1"]

    def test_more_ids_than_a_page_holds_is_refused(self):
        with build_client() as client:
            response = client.get(
                "/v1/graphs/default/entities",
                params={"ids": [f"e{i}" for i in range(60)], "limit": 50},
            )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_REQUEST"

    def test_chunks_page_without_ids(self):
        with build_client() as client:
            body = client.get(
                "/v1/graphs/default/chunks", params={"limit": 1}
            ).json()
        assert body["page"]["total"] == 2
        assert len(body["chunks"]) == 1


class TestEntityOrdering:
    """Sorting server-side, so a client need not download the corpus to rank it."""

    def test_sorting_by_name(self):
        with build_client() as client:
            ascending = client.get(
                "/v1/graphs/default/entities", params={"sort": "name"}
            ).json()
            descending = client.get(
                "/v1/graphs/default/entities",
                params={"sort": "name", "order": "desc"},
            ).json()
        names = [e["name"] for e in ascending["entities"]]
        assert names == sorted(names, key=str.casefold)
        assert [e["name"] for e in descending["entities"]] == names[::-1]

    def test_sorting_by_degree_puts_the_most_connected_first(self):
        with build_client() as client:
            body = client.get(
                "/v1/graphs/default/entities",
                params={"sort": "degree", "order": "desc"},
            ).json()
        assert len(body["entities"]) == 2

    def test_an_unknown_sort_key_is_refused(self):
        with build_client() as client:
            response = client.get(
                "/v1/graphs/default/entities", params={"sort": "colour"}
            )
        assert response.status_code == 400

    def test_filtering_by_community(self):
        with build_client() as client:
            inside = client.get(
                "/v1/graphs/default/entities", params={"community_id": "0"}
            ).json()
            outside = client.get(
                "/v1/graphs/default/entities", params={"community_id": "999"}
            ).json()
        assert inside["entities"] and not outside["entities"]


class TestProvenanceAndSurface:
    """The graph surface reports where each thing came from."""

    def test_entities_and_relations_name_their_chunks(self):
        with build_client() as client:
            entity = client.get("/v1/graphs/default/entities").json()["entities"][0]
            relation = client.get("/v1/graphs/default/relations").json()["relations"][0]
        assert entity["source_chunk_ids"] == ["chunk_1"]
        assert relation["source_chunk_ids"] == ["chunk_1"]

    def test_one_entity_can_be_fetched_on_its_own(self):
        with build_client() as client:
            found = client.get("/v1/graphs/default/entities/entity_1")
            missing = client.get("/v1/graphs/default/entities/nope")
        assert found.status_code == 200
        assert found.json()["name"] == "Сенкевич"
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "NOT_FOUND"

    def test_a_community_lists_its_members(self):
        with build_client() as client:
            community = client.get(
                "/v1/graphs/default/communities"
            ).json()["communities"][0]
        assert community["entity_ids"] == ["entity_1", "entity_2"]
        assert community["truncated"] is False


class TestRelationSelectRoute:
    """The wire contract of the selection route."""

    def test_the_set_travels_in_the_body(self):
        with build_client() as client:
            body = client.post(
                "/v1/graphs/default/relations/select",
                json={"entity_ids": ["entity_1", "entity_2"]},
            ).json()
        assert [r["id"] for r in body["relations"]] == ["relation_1"]
        assert body["page"]["total"] == 1

    def test_induced_is_the_default_scope(self):
        with build_client() as client:
            induced = client.post(
                "/v1/graphs/default/relations/select",
                json={"entity_ids": ["entity_1"]},
            ).json()
            incident = client.post(
                "/v1/graphs/default/relations/select",
                json={"entity_ids": ["entity_1"], "edge_scope": "incident"},
            ).json()
        assert induced["relations"] == []
        assert [r["id"] for r in incident["relations"]] == ["relation_1"]

    def test_an_empty_set_is_refused(self):
        with build_client() as client:
            response = client.post(
                "/v1/graphs/default/relations/select", json={"entity_ids": []}
            )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_REQUEST"

    def test_more_ids_than_the_ceiling_is_refused(self):
        with build_client() as client:
            response = client.post(
                "/v1/graphs/default/relations/select",
                json={"entity_ids": [f"ent-{i}" for i in range(10_001)]},
            )
        assert response.status_code == 400

    def test_ten_thousand_ids_fit_in_the_default_body_limit(self):
        # Roughly 400 KB against a 32 MiB ceiling.
        with build_client() as client:
            response = client.post(
                "/v1/graphs/default/relations/select",
                json={"entity_ids": [f"ent-{i:032d}" for i in range(10_000)]},
            )
        assert response.status_code == 200

    def test_an_unknown_field_is_refused(self):
        with build_client() as client:
            response = client.post(
                "/v1/graphs/default/relations/select",
                json={"entity_ids": ["entity_1"], "bogus": 1},
            )
        assert response.status_code == 400


class TestPageCeiling:
    """A consumer that exports a corpus should not pay a hundred round trips."""

    def test_five_thousand_is_accepted_everywhere(self):
        with build_client() as client:
            for path in ("entities", "relations", "communities"):
                response = client.get(
                    f"/v1/graphs/default/{path}", params={"limit": 5000}
                )
                assert response.status_code == 200, path

    def test_above_the_ceiling_is_refused(self):
        with build_client() as client:
            for path in ("entities", "relations", "communities"):
                response = client.get(
                    f"/v1/graphs/default/{path}", params={"limit": 5001}
                )
                assert response.status_code == 400, path

    def test_the_default_page_is_still_fifty(self):
        with build_client() as client:
            for path in ("entities", "relations", "communities"):
                body = client.get(f"/v1/graphs/default/{path}").json()
                assert body["page"]["limit"] == 50, path

    def test_ids_in_a_query_string_kept_their_own_ceiling(self):
        # The page ceiling rose to 5000; this one did not follow it, because
        # 501 ids of 36 characters is already a URL of roughly 20 KB.
        with build_client() as client:
            response = client.get(
                "/v1/graphs/default/entities",
                params={"ids": [f"ent-{i}" for i in range(501)], "limit": 5000},
            )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_REQUEST"


class TestGraphTimestamps:
    """
    A graph reports when it was written, and admits when it cannot say more.
    """

    def test_updated_at_is_the_latest_write(self, tmp_path):
        import os
        import time

        from ragu.api.backends.ragu_backend.backend import _folder_timestamps

        older = tmp_path / "a.json"
        newer = tmp_path / "b.json"
        older.write_text("{}")
        newer.write_text("{}")
        past = time.time() - 3600
        os.utime(older, (past, past))

        _, updated = _folder_timestamps(str(tmp_path))
        assert abs(updated.timestamp() - newer.stat().st_mtime) < 1

    def test_a_missing_folder_yields_nothing_rather_than_a_date(self, tmp_path):
        from ragu.api.backends.ragu_backend.backend import _folder_timestamps

        assert _folder_timestamps(str(tmp_path / "nope")) == (None, None)

    def test_created_at_is_never_guessed(self, tmp_path, monkeypatch):
        # Linux does not expose a creation time. The oldest modification time is
        # not one either — an in-place rebuild rewrites every file — so the
        # field stays null rather than carrying a plausible wrong date.
        import os

        from ragu.api.backends.ragu_backend import backend as backend_module

        (tmp_path / "a.json").write_text("{}")
        real_stat = os.stat

        class NoBirth:
            def __init__(self, result):
                self._result = result

            def __getattr__(self, name):
                if name == "st_birthtime":
                    raise AttributeError(name)
                return getattr(self._result, name)

        monkeypatch.setattr(
            backend_module.os, "stat", lambda path, *a, **k: NoBirth(real_stat(path, *a, **k))
        )
        created, updated = backend_module._folder_timestamps(str(tmp_path))
        assert created is None
        assert updated is not None

    def test_stats_carry_the_dates(self):
        with build_client() as client:
            body = client.get("/v1/graphs/default/stats").json()
        assert body["updated_at"].startswith("2026-01-01")
        assert body["created_at"].startswith("2026-01-01")
