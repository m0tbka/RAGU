"""
A typed client for the RAGU service.

Written so that consumers do not each hand-roll the same HTTP calls and the same
envelope parsing, and then drift from the contract one fix at a time. The
request and response models are the service's own, so a change to the schema is
a type error here rather than a surprise at runtime.

``httpx`` is imported lazily: it belongs to the test extra, not the ``api`` one,
and the service itself does not need it.
"""

from collections.abc import AsyncIterator
from typing import Any

from ragu.api.models import (
    BatchSearchResponse,
    ChunkItem,
    CommunityDetail,
    CommunityPage,
    ConsistencyReportModel,
    EntityPage,
    GraphDetail,
    GraphInfo,
    GraphListResponse,
    HealthResponse,
    JobListResponse,
    JobResponse,
    ModeAvailability,
    Neighborhood,
    OntologyResponse,
    RelationPage,
    RetrieveResponse,
    SearchMode,
    SearchResponse,
)

DEFAULT_TIMEOUT = 300.0


class RaguApiError(RuntimeError):
    """
    The service answered with an error envelope.

    Carries the envelope's fields rather than only its text, so a caller can
    branch on ``code`` — switching search mode on ``CAPABILITY_UNAVAILABLE``,
    backing off on ``SERVICE_NOT_READY`` — without parsing strings.
    """

    def __init__(self, status_code: int, payload: dict[str, Any]):
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        self.status_code = status_code
        self.code: str = error.get("code", "UNKNOWN")
        self.mode: str | None = error.get("mode")
        self.missing_capability: str | None = error.get("missing_capability")
        self.request_id: str | None = error.get("request_id")
        self.message: str = error.get("message", "")
        super().__init__(f"{status_code} {self.code}: {self.message}")


def _httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "ragu.api.client needs httpx. Install it, or use the service over "
            "plain HTTP."
        ) from exc
    return httpx


class RaguClient:
    """
    Async client for one RAGU service.

    :param base_url: Where the service is, e.g. ``http://localhost:8020``.
    :param api_key: Sent as ``Authorization: Bearer``; omit when the service is open.
    :param graph: Graph addressed when a call names none.
    :param timeout: Seconds to wait. Generous by default: a global search is
        N+1 LLM calls.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        graph: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        transport: Any = None,
    ):
        httpx = _httpx()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )
        self.graph = graph

    async def __aenter__(self) -> "RaguClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """
        Close the underlying connection pool.
        """
        await self._client.aclose()

    # --- plumbing ------------------------------------------------------------

    def _prefix(self, graph: str | None) -> str:
        name = graph or self.graph
        return f"/v1/graphs/{name}" if name else "/v1"

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self._client.request(method, path, **kwargs)
        if response.status_code >= 400:
            try:
                payload = response.json()
            except Exception:
                payload = {"error": {"code": "UNKNOWN", "message": response.text}}
            raise RaguApiError(response.status_code, payload)
        return response.json()

    async def _get(self, path: str, **params: Any) -> Any:
        clean = {key: value for key, value in params.items() if value is not None}
        return await self._request("GET", path, params=clean)

    async def _post(self, path: str, body: dict[str, Any], **kwargs: Any) -> Any:
        return await self._request("POST", path, json=body, **kwargs)

    # --- service -------------------------------------------------------------

    async def health(self) -> HealthResponse:
        return HealthResponse.model_validate(await self._get("/health"))

    async def ready(self) -> bool:
        """
        Whether the service can serve searches right now.
        """
        try:
            payload = await self._get("/health/ready")
        except RaguApiError:
            return False
        return bool(payload.get("graph_loaded"))

    async def ontology(self) -> OntologyResponse:
        return OntologyResponse.model_validate(await self._get("/v1/ontology"))

    # --- catalogue -----------------------------------------------------------

    async def graphs(self) -> GraphListResponse:
        return GraphListResponse.model_validate(await self._get("/v1/graphs"))

    async def graph_info(self, graph: str | None = None) -> GraphInfo:
        name = graph or self.graph
        return GraphInfo.model_validate(await self._get(f"/v1/graphs/{name}"))

    async def capabilities(self, graph: str | None = None) -> list[ModeAvailability]:
        name = graph or self.graph
        payload = await self._get(f"/v1/graphs/{name}/capabilities")
        return [ModeAvailability.model_validate(item) for item in payload]

    async def stats(self, graph: str | None = None) -> GraphDetail:
        name = graph or self.graph
        return GraphDetail.model_validate(await self._get(f"/v1/graphs/{name}/stats"))

    # --- search --------------------------------------------------------------

    async def search(
        self, mode: SearchMode, query: str, *, graph: str | None = None, **body: Any
    ) -> SearchResponse:
        path = f"{self._prefix(graph)}/search/{mode}"
        return SearchResponse.model_validate(
            await self._post(path, {"query": query, **body})
        )

    async def retrieve(
        self, mode: SearchMode, query: str, *, graph: str | None = None, **body: Any
    ) -> RetrieveResponse:
        path = f"{self._prefix(graph)}/search/{mode}/retrieve"
        return RetrieveResponse.model_validate(
            await self._post(path, {"query": query, **body})
        )

    async def batch(
        self,
        mode: SearchMode,
        queries: list[str],
        *,
        graph: str | None = None,
        **body: Any,
    ) -> BatchSearchResponse:
        path = f"{self._prefix(graph)}/search/{mode}/batch"
        return BatchSearchResponse.model_validate(
            await self._post(path, {"queries": queries, **body})
        )

    async def stream(
        self, mode: SearchMode, query: str, *, graph: str | None = None, **body: Any
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """
        Stream an answer, yielding ``(event, data)`` pairs.

        :return: ``meta`` once, then ``delta`` per chunk, then ``done``.
        """
        import json

        path = f"{self._prefix(graph)}/search/{mode}/stream"
        async with self._client.stream(
            "POST", path, json={"query": query, **body}
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise RaguApiError(response.status_code, response.json())
            event = "message"
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    event = line[len("event: ") :]
                elif line.startswith("data: "):
                    yield event, json.loads(line[len("data: ") :])

    # --- the graph surface ---------------------------------------------------

    async def entities(
        self,
        *,
        graph: str | None = None,
        limit: int = 50,
        offset: int = 0,
        type: str | None = None,
        search: str | None = None,
    ) -> EntityPage:
        name = graph or self.graph
        payload = await self._get(
            f"/v1/graphs/{name}/entities",
            limit=limit,
            offset=offset,
            type=type,
            search=search,
        )
        return EntityPage.model_validate(payload)

    async def relations(
        self,
        *,
        graph: str | None = None,
        limit: int = 50,
        offset: int = 0,
        min_strength: float | None = None,
    ) -> RelationPage:
        name = graph or self.graph
        payload = await self._get(
            f"/v1/graphs/{name}/relations",
            limit=limit,
            offset=offset,
            min_strength=min_strength,
        )
        return RelationPage.model_validate(payload)

    async def neighbors(
        self,
        entity_id: str,
        *,
        graph: str | None = None,
        depth: int = 1,
        limit: int = 200,
    ) -> Neighborhood:
        name = graph or self.graph
        payload = await self._get(
            f"/v1/graphs/{name}/entities/{entity_id}/neighbors",
            depth=depth,
            limit=limit,
        )
        return Neighborhood.model_validate(payload)

    async def communities(
        self,
        *,
        graph: str | None = None,
        limit: int = 50,
        offset: int = 0,
        level: int | None = None,
    ) -> CommunityPage:
        name = graph or self.graph
        payload = await self._get(
            f"/v1/graphs/{name}/communities", limit=limit, offset=offset, level=level
        )
        return CommunityPage.model_validate(payload)

    async def community(
        self, community_id: str, *, graph: str | None = None
    ) -> CommunityDetail:
        name = graph or self.graph
        payload = await self._get(f"/v1/graphs/{name}/communities/{community_id}")
        return CommunityDetail.model_validate(payload)

    async def chunk(self, chunk_id: str, *, graph: str | None = None) -> ChunkItem:
        name = graph or self.graph
        return ChunkItem.model_validate(
            await self._get(f"/v1/graphs/{name}/chunks/{chunk_id}")
        )

    async def consistency(self, *, graph: str | None = None) -> ConsistencyReportModel:
        name = graph or self.graph
        return ConsistencyReportModel.model_validate(
            await self._get(f"/v1/graphs/{name}/consistency")
        )

    # --- ingestion and jobs --------------------------------------------------

    async def add_documents(
        self,
        documents: list[str],
        *,
        graph: str | None = None,
        idempotency_key: str | None = None,
    ) -> JobResponse:
        """
        Submit documents for ingestion and return the job to poll.
        """
        name = graph or self.graph
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return JobResponse.model_validate(
            await self._post(
                f"/v1/graphs/{name}/documents", {"documents": documents}, headers=headers
            )
        )

    async def reindex(
        self,
        kind: str,
        *,
        graph: str | None = None,
        idempotency_key: str | None = None,
    ) -> JobResponse:
        name = graph or self.graph
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return JobResponse.model_validate(
            await self._request(
                "POST", f"/v1/graphs/{name}/reindex/{kind}", headers=headers
            )
        )

    async def jobs(self, *, graph: str | None = None) -> JobListResponse:
        return JobListResponse.model_validate(
            await self._get("/v1/jobs", graph_id=graph or self.graph)
        )

    async def job(self, job_id: str) -> JobResponse:
        return JobResponse.model_validate(await self._get(f"/v1/jobs/{job_id}"))

    async def cancel_job(self, job_id: str) -> JobResponse:
        return JobResponse.model_validate(
            await self._request("DELETE", f"/v1/jobs/{job_id}")
        )
