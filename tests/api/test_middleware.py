"""
What surrounds every request: correlation ids, authentication, size and
time limits, CORS and admission.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "fastapi", reason="install the 'api' extra to test the search service"
)

from fastapi.testclient import TestClient  # noqa: E402

from ragu.api.app import create_app
from ragu.api.config import ServiceSettings
from tests.api.support import (
    build_client,
)


class TestRequestCorrelation:
    """A 500 says 'see the log'; the id is how the log line is found."""

    def test_every_response_carries_a_request_id(self):
        with build_client() as client:
            response = client.get("/health")
        assert response.headers["X-Request-ID"]

    def test_a_client_supplied_id_is_kept(self):
        with build_client() as client:
            response = client.get("/health", headers={"X-Request-ID": "trace-42"})
        assert response.headers["X-Request-ID"] == "trace-42"

    def test_the_error_envelope_repeats_it(self):
        with build_client(missing="entity_graph") as client:
            response = client.post(
                "/v1/search/local",
                json={"query": "q"},
                headers={"X-Request-ID": "trace-99"},
            )
        assert response.json()["error"]["request_id"] == "trace-99"
        assert response.headers["X-Request-ID"] == "trace-99"


class TestAuth:
    """Every request costs LLM calls, so an open service is an open budget."""

    def keyed_client(self) -> TestClient:
        return TestClient(
            create_app(ServiceSettings(backend="stub", api_keys="secret-a,secret-b"))
        )

    def test_a_request_without_a_key_is_refused(self):
        with self.keyed_client() as client:
            response = client.post("/v1/search/naive", json={"query": "q"})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"
        assert response.headers["WWW-Authenticate"] == "Bearer"

    @pytest.mark.parametrize(
        "headers",
        [
            {"Authorization": "Bearer secret-a"},
            {"Authorization": "bearer secret-b"},
            {"X-API-Key": "secret-a"},
        ],
    )
    def test_an_accepted_key_gets_through(self, headers):
        with self.keyed_client() as client:
            response = client.post(
                "/v1/search/naive", json={"query": "q"}, headers=headers
            )
        assert response.status_code == 200

    def test_a_wrong_key_is_refused(self):
        with self.keyed_client() as client:
            response = client.post(
                "/v1/search/naive",
                json={"query": "q"},
                headers={"X-API-Key": "secret-c"},
            )
        assert response.status_code == 401

    def test_probes_stay_open(self):
        # An orchestrator has no key, and a client needs the schema to talk.
        with self.keyed_client() as client:
            for path in ("/health", "/health/live", "/health/ready", "/openapi.json"):
                assert client.get(path).status_code in (200, 503)

    def test_no_keys_configured_leaves_the_service_open(self):
        with build_client() as client:
            assert client.post("/v1/search/naive", json={"query": "q"}).status_code == 200


class TestServiceBounds:
    def test_an_oversized_body_is_refused(self):
        with build_client(max_body_bytes=200) as client:
            response = client.post(
                "/v1/search/naive", json={"query": "x" * 1000}
            )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"

    def test_a_body_within_the_limit_is_read(self):
        with build_client(max_body_bytes=32 * 1024) as client:
            assert client.post("/v1/search/naive", json={"query": "q"}).status_code == 200

    def test_a_request_that_overruns_is_abandoned(self):
        import asyncio

        class SlowBackend:
            graph_loaded = True
            stats = None
            graph_id = "default"
            language = "russian"

            async def startup(self):
                pass

            async def shutdown(self):
                pass

            def require_capability(self, mode, mix_engines=None):
                pass

            def require_evidence(self, mode, outcome):
                pass

            async def search(self, call):
                await asyncio.sleep(5)
                return []

        app = create_app(
            ServiceSettings(backend="stub", request_timeout=0.05), backend=SlowBackend()
        )
        with TestClient(app) as client:
            response = client.post("/v1/search/naive", json={"query": "q"})

        assert response.status_code == 504
        assert response.json()["error"]["code"] == "REQUEST_TIMEOUT"
        assert response.headers["Retry-After"] == "30"


class TestCors:
    def test_no_origins_means_no_cors_headers(self):
        with build_client() as client:
            response = client.get("/health", headers={"Origin": "https://ui.example"})
        assert "access-control-allow-origin" not in response.headers

    def test_a_configured_origin_is_allowed(self):
        settings = ServiceSettings(backend="stub", cors_origins="https://ui.example")
        with TestClient(create_app(settings)) as client:
            response = client.get("/health", headers={"Origin": "https://ui.example"})
        assert response.headers["access-control-allow-origin"] == "https://ui.example"
        assert "X-Request-ID" in response.headers["access-control-expose-headers"]


class TestAdmissionControl:
    """Refusing at the door is cheaper than queueing inside the process."""

    async def test_a_full_service_refuses_rather_than_queues(self):
        import asyncio

        from ragu.api.errors import TooManyRequestsError
        from ragu.api.runtime.middleware import Admission

        admission = Admission(1)
        held = asyncio.Event()
        release = asyncio.Event()

        async def occupy():
            async with admission.slot():
                held.set()
                await release.wait()

        task = asyncio.create_task(occupy())
        await held.wait()

        with pytest.raises(TooManyRequestsError) as failure:
            async with admission.slot():
                pass
        assert failure.value.status_code == 429
        assert failure.value.headers["Retry-After"] == "5"

        release.set()
        await task

    async def test_the_slot_is_released_afterwards(self):
        from ragu.api.runtime.middleware import Admission

        admission = Admission(1)
        async with admission.slot():
            pass
        async with admission.slot():
            pass

    async def test_no_limit_configured_admits_everything(self):
        from ragu.api.runtime.middleware import Admission

        admission = Admission(None)
        async with admission.slot():
            async with admission.slot():
                pass
