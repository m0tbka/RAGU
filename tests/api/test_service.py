"""
The service's own endpoints and configuration: health, errors, settings,
logging and metrics.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "fastapi", reason="install the 'api' extra to test the search service"
)

from fastapi.testclient import TestClient  # noqa: E402

from ragu.api.app import UNHANDLED_ERROR_MESSAGE, create_app
from ragu.api.config import ServiceSettings
from tests.api.support import (
    build_client,
    ingest_client,
)


class TestHealth:
    def test_health_reports_loaded_graph(self):
        with build_client() as client:
            body = client.get("/health").json()
        assert (body["status"], body["graph_loaded"], body["error"]) == (
            "ok",
            True,
            None,
        )

    def test_health_reports_graph_sizes(self):
        with build_client() as client:
            stats = client.get("/health").json()["stats"]
        assert stats == {
            "entities": 1,
            "relations": 1,
            "chunks": 1,
            "community_summaries": 1,
        }

    def test_readiness_is_503_until_the_graph_is_loaded(self):
        # No lifespan run: the app has no backend yet.
        client = TestClient(create_app(ServiceSettings(backend="stub")))
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "30"
        assert response.json()["graph_loaded"] is False

    def test_readiness_is_200_once_loaded(self):
        with build_client() as client:
            assert client.get("/health/ready").status_code == 200

    def test_liveness_answers_while_degraded(self):
        # Liveness must not restart a service that is still loading a graph.
        client = TestClient(create_app(ServiceSettings(backend="stub")))
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"

    def test_search_answers_503_until_the_graph_is_loaded(self):
        client = TestClient(create_app(ServiceSettings(backend="stub")))
        response = client.post("/v1/search/naive", json={"query": "q"})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "SERVICE_NOT_READY"

    def test_a_startup_failure_is_reported_instead_of_being_swallowed(self):
        class ExplodingBackend:
            graph_loaded = False
            stats = None

            async def startup(self):
                raise RuntimeError("embedder endpoint unreachable")

            async def shutdown(self):
                pass

        with TestClient(
            create_app(ServiceSettings(backend="stub"), backend=ExplodingBackend())
        ) as client:
            health = client.get("/health").json()
            search = client.post("/v1/search/naive", json={"query": "q"})

        assert "embedder endpoint unreachable" in health["error"]
        assert "embedder endpoint unreachable" in search.json()["error"]["message"]


class TestErrorEnvelope:
    def test_an_unhandled_error_does_not_leak_its_message(self):
        # Engine and LLM-client errors quote the endpoint URL and parts of the
        # request; only the log may see them.
        class ExplodingBackend:
            graph_loaded = True
            stats = None

            async def startup(self):
                pass

            async def shutdown(self):
                pass

            async def search_naive(self, query, *, use_query_plan, params):
                raise RuntimeError("https://llm.internal/v1 rejected sk-secret")

        app = create_app(ServiceSettings(backend="stub"), backend=ExplodingBackend())
        with TestClient(app, raise_server_exceptions=False) as client:
            body = client.post("/v1/search/naive", json={"query": "q"}).json()

        assert body["error"]["message"] == UNHANDLED_ERROR_MESSAGE
        assert "sk-secret" not in str(body)

    def test_every_error_shares_one_envelope_shape(self):
        with build_client(missing="entity_graph") as client:
            errors = [
                client.post("/v1/search/local", json={"query": "q"}).json()["error"],
                client.post("/v1/search/naive", json={"query": ""}).json()["error"],
            ]
        for error in errors:
            assert set(error) == {
                "code",
                "mode",
                "missing_capability",
                "message",
                "request_id",
            }


class TestRetryAfter:
    def test_search_503_tells_the_client_when_to_come_back(self):
        client = TestClient(create_app(ServiceSettings(backend="stub")))
        response = client.post("/v1/search/naive", json={"query": "q"})
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "30"

    def test_other_errors_carry_no_retry_after(self):
        with build_client(missing="entity_graph") as client:
            response = client.post("/v1/search/local", json={"query": "q"})
        assert response.status_code == 409
        assert "Retry-After" not in response.headers


class TestServiceConfig:
    def test_bind_address_defaults_to_loopback(self):
        # Exposing the service is a deliberate act; the container image passes
        # --host 0.0.0.0 itself.
        assert ServiceSettings(backend="stub").host == "127.0.0.1"

    def test_a_misspelled_stub_capability_is_rejected(self):
        # Silently simulating nothing turns a config typo into a test that
        # passes for the wrong reason.
        with pytest.raises(ValueError) as failure:
            ServiceSettings(backend="stub", stub_missing_capabilities="entity_grap")
        assert "entity_grap" in str(failure.value)

    def test_known_capabilities_are_accepted(self):
        settings = ServiceSettings(
            backend="stub",
            stub_missing_capabilities=" entity_graph , vector_index ",
        )
        assert settings.missing_capabilities() == {"entity_graph", "vector_index"}

    def test_capability_names_match_the_mode_requirements(self):
        # config.py validates against models.CAPABILITIES while the backends
        # answer from MODE_REQUIREMENTS; they must not drift apart.
        from ragu.api.backends.capabilities import MODE_REQUIREMENTS
        from ragu.api.models import CAPABILITIES

        declared = {
            capability
            for requirement in MODE_REQUIREMENTS.values()
            for capability in requirement.requires
        }
        assert declared == CAPABILITIES


class TestLogging:
    """The service logs through loguru, like the rest of RAGU."""

    def test_the_api_package_does_not_use_stdlib_logging(self):
        import pathlib

        import ragu.api

        package = pathlib.Path(ragu.api.__file__).parent
        offenders = [
            path.name
            for path in package.rglob("*.py")
            # logging_setup.py is the bridge itself; it must import logging.
            if path.name != "logging_setup.py"
            and "logging.getLogger" in path.read_text(encoding="utf-8")
        ]
        assert offenders == []

    def test_stdlib_records_are_re_emitted_through_loguru(self):
        import logging

        from ragu.api.runtime.logging_setup import InterceptHandler
        from ragu.common.logger import logger

        captured = []
        sink_id = logger.add(lambda message: captured.append(message), level="DEBUG")
        try:
            record = logging.LogRecord(
                name="uvicorn.error",
                level=logging.WARNING,
                pathname=__file__,
                lineno=1,
                msg="listening on %s",
                args=("127.0.0.1:8020",),
                exc_info=None,
            )
            InterceptHandler().emit(record)
        finally:
            logger.remove(sink_id)

        assert len(captured) == 1
        assert "listening on 127.0.0.1:8020" in captured[0]
        assert captured[0].record["level"].name == "WARNING"

    def test_set_level_replaces_the_sink(self):
        from ragu.common.logger import DEFAULT_LEVEL, set_level

        try:
            set_level("debug")
            with pytest.raises(ValueError):
                set_level("not-a-level")
        finally:
            set_level(DEFAULT_LEVEL)

    def test_set_level_survives_a_bare_logger_remove(self):
        # logger.remove() with no argument is the usual way to reconfigure
        # loguru, and it invalidates the sink id set_level tracks.
        from ragu.common.logger import DEFAULT_LEVEL, logger, set_level

        try:
            logger.remove()
            set_level("warning")
            assert logger._core.handlers
        finally:
            set_level(DEFAULT_LEVEL)


class TestMetrics:
    """Every request costs money; the operator needs to see the shape of it."""

    def test_metrics_are_exposed_in_prometheus_format(self):
        with build_client() as client:
            client.post("/v1/search/naive", json={"query": "q"})
            body = client.get("/metrics").text
        assert "# TYPE ragu_api_requests_total counter" in body
        assert "# TYPE ragu_api_request_duration_seconds histogram" in body
        assert 'ragu_api_searches_total{mode="naive",outcome="ok"}' in body

    def test_the_route_template_is_the_label_not_the_url(self):
        # Otherwise every graph id and query would open a new time series.
        with build_client() as client:
            client.post("/v1/graphs/default/search/naive", json={"query": "q"})
            body = client.get("/metrics").text
        assert 'path="/v1/graphs/{graph_id}/search/naive"' in body

    def test_a_degraded_search_is_counted_apart(self):
        with build_client() as client:
            client.post("/v1/search/naive", json={"query": "q"})
            body = client.get("/metrics").text
        assert 'outcome="ok"' in body

    def test_graph_and_job_gauges_are_reported(self):
        with ingest_client() as client:
            client.post("/v1/graphs/corpus/documents", json={"documents": ["a"]})
            body = client.get("/metrics").text
        assert 'ragu_api_graphs{state="loaded"}' in body
        assert "ragu_api_jobs{" in body

    def test_metrics_stay_reachable_without_a_key(self):
        settings = ServiceSettings(backend="stub", api_keys="secret")
        with TestClient(create_app(settings)) as client:
            assert client.get("/metrics").status_code == 200

    def test_the_histogram_is_cumulative(self):
        from ragu.api.runtime.metrics import DURATION, Metrics

        registry = Metrics()
        registry.observe(DURATION, 0.3)
        registry.observe(DURATION, 7.0)
        lines = registry.render().splitlines()

        buckets = {
            line.split('le="')[1].split('"')[0]: int(line.rsplit(" ", 1)[1])
            for line in lines
            if "_bucket{" in line
        }
        assert buckets["0.25"] == 0
        assert buckets["0.5"] == 1
        assert buckets["10"] == 2
        assert buckets["+Inf"] == 2
        assert "ragu_api_request_duration_seconds_count 2" in lines

    def test_observations_are_not_accumulated_one_by_one(self):
        # A long-running service must not keep a float per request.
        from ragu.api.runtime.metrics import DURATION, Metrics

        registry = Metrics()
        for _ in range(10_000):
            registry.observe(DURATION, 0.1)
        series = registry._histograms[DURATION][()]
        assert series.count == 10_000
        assert len(series.buckets) == len(
            __import__("ragu.api.runtime.metrics", fromlist=["DURATION_BUCKETS"]).DURATION_BUCKETS
        )
