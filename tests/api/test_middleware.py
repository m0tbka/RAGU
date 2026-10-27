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

    @staticmethod
    def capture():
        """Collect the loguru records emitted while the block runs."""
        from contextlib import contextmanager

        from ragu.common.logger import logger

        @contextmanager
        def capturing():
            records = []
            sink = logger.add(lambda message: records.append(message.record), level="DEBUG")
            try:
                yield records
            finally:
                logger.remove(sink)

        return capturing()

    def test_a_500_carries_the_id_in_its_body_header_and_log_line(self):
        # The one case the id exists for: a 500 says only "see the log". It used
        # to be rendered outside every middleware, with no id anywhere.
        app = create_app(ServiceSettings(backend="stub"))

        @app.get("/boom")
        async def boom():
            raise RuntimeError("https://llm.internal/v1 rejected the request")

        with self.capture() as records:
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/boom", headers={"X-Request-ID": "trace-500"})

        assert response.status_code == 500
        assert response.json()["error"]["request_id"] == "trace-500"
        assert response.headers["X-Request-ID"] == "trace-500"
        logged = [r for r in records if r["message"] == "Unhandled service error"]
        assert [r["extra"].get("request_id") for r in logged] == ["trace-500"]
        assert "llm.internal" not in response.text

    def test_every_line_a_request_logs_carries_its_id(self):
        from ragu.common.logger import logger

        app = create_app(ServiceSettings(backend="stub"))

        @app.get("/noisy")
        async def noisy():
            logger.info("inside the handler")
            return {}

        with self.capture() as records:
            with TestClient(app) as client:
                client.get("/noisy", headers={"X-Request-ID": "trace-7"})

        inside = [r for r in records if r["message"] == "inside the handler"]
        assert [r["extra"].get("request_id") for r in inside] == ["trace-7"]

    async def test_the_servers_access_line_is_inside_the_binding(self):
        # uvicorn writes its access line from inside send(), as the response
        # starts. That is past the point where a BaseHTTPMiddleware has returned,
        # so the line an operator greps first used to carry no id.
        from ragu.common.logger import logger

        app = create_app(ServiceSettings(backend="stub"))
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/health/live",
            "raw_path": b"/health/live",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"x-request-id", b"trace-access")],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8020),
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            if message["type"] == "http.response.start":
                logger.info("GET /health/live 200")

        with self.capture() as records:
            await app(scope, receive, send)

        access = [r for r in records if r["message"] == "GET /health/live 200"]
        assert [r["extra"].get("request_id") for r in access] == ["trace-access"]

    def test_an_id_that_could_forge_a_log_line_is_replaced(self):
        # The id is written into every log line of its request; one that reads
        # as log text would let a client write lines of its own.
        with build_client() as client:
            response = client.get(
                "/health", headers={"X-Request-ID": "x | ERROR | disk full"}
            )

        issued = response.headers["X-Request-ID"]
        assert issued != "x | ERROR | disk full"
        assert len(issued) == 32 and issued.isalnum()


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
