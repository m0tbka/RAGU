"""
The graph catalogue: several graphs in one process, and what each serves.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "fastapi", reason="install the 'api' extra to test the search service"
)

from ragu.api.config import ServiceSettings
from tests.api.support import (
    build_client,
    graphs_client,
)


class TestGraphCatalogue:
    """Several graphs in one process, addressed by name."""

    SPECS = [
        {"id": "books", "storage_folder": "a", "language": "russian"},
        {"id": "papers", "storage_folder": "b", "language": "english"},
    ]

    def test_the_catalogue_lists_every_graph(self):
        with graphs_client(self.SPECS) as client:
            body = client.get("/v1/graphs").json()
        assert body["default"] == "books"
        assert [graph["id"] for graph in body["graphs"]] == ["books", "papers"]
        assert all(graph["loaded"] for graph in body["graphs"])

    def test_each_graph_keeps_its_own_language(self):
        with graphs_client(self.SPECS) as client:
            body = client.get("/v1/graphs").json()
        assert {g["id"]: g["language"] for g in body["graphs"]} == {
            "books": "russian",
            "papers": "english",
        }

    def test_a_graph_can_be_addressed_by_name(self):
        with graphs_client(self.SPECS) as client:
            body = client.post(
                "/v1/graphs/papers/search/naive", json={"query": "q"}
            ).json()
        assert body["mode"] == "naive"
        assert body["answer"].startswith("[stub naive]")

    def test_the_flat_path_serves_the_default_graph(self):
        # Kept for clients written before the service served more than one graph.
        with graphs_client(self.SPECS) as client:
            assert client.post("/v1/search/naive", json={"query": "q"}).status_code == 200

    def test_an_unknown_graph_answers_404(self):
        with graphs_client(self.SPECS) as client:
            for path in ("/v1/graphs/missing", "/v1/graphs/missing/search/naive"):
                response = (
                    client.get(path)
                    if path.endswith("missing")
                    else client.post(path, json={"query": "q"})
                )
                assert response.status_code == 404
                assert response.json()["error"]["code"] == "GRAPH_NOT_FOUND"

    def test_every_search_shape_is_addressable_per_graph(self):
        with graphs_client(self.SPECS) as client:
            assert client.post(
                "/v1/graphs/papers/search/naive/retrieve", json={"query": "q"}
            ).status_code == 200
            assert client.post(
                "/v1/graphs/papers/search/naive/batch", json={"queries": ["a"]}
            ).status_code == 200
            assert client.post(
                "/v1/graphs/papers/search/naive/stream", json={"query": "q"}
            ).status_code == 200

    def test_duplicate_ids_are_refused(self):
        with pytest.raises(ValueError):
            ServiceSettings(
                backend="stub",
                graphs=[
                    {"id": "a", "storage_folder": "x"},
                    {"id": "a", "storage_folder": "y"},
                ],
            )


class TestCapabilitiesEndpoint:
    """A client has to know which modes to offer before it offers them."""

    def test_capabilities_name_the_available_modes(self):
        with build_client() as client:
            modes = client.get("/v1/graphs/default/capabilities").json()
        assert {mode["mode"] for mode in modes} == {"global", "local", "naive", "mix"}
        assert all(mode["available"] for mode in modes)
        assert all(mode["reason"] is None for mode in modes)

    def test_an_unavailable_mode_says_what_it_needs(self):
        with build_client(missing="entity_graph") as client:
            modes = {m["mode"]: m for m in client.get("/v1/graphs/default/capabilities").json()}

        assert modes["naive"]["available"] is True
        assert modes["local"]["available"] is False
        assert modes["local"]["missing_capability"] == "entity_graph"
        assert "entity index" in modes["local"]["reason"]
        # mix ensembles local and naive, so it goes with local.
        assert modes["mix"]["available"] is False

    def test_the_same_view_is_on_the_graph_record(self):
        with build_client(missing="vector_index") as client:
            graph = client.get("/v1/graphs/default").json()
        naive = next(m for m in graph["modes"] if m["mode"] == "naive")
        assert naive["missing_capability"] == "vector_index"
