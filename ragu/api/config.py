"""
Service configuration.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ragu.api.models import CAPABILITIES

# Identifier of the graph served when none are configured explicitly.
DEFAULT_GRAPH_ID = "default"


class BuildSpec(BaseModel):
    """
    How documents become graph, for the graphs that accept documents.

    Ingestion is off by default: the service's original job is to serve a
    prebuilt graph, and building one is a different, far more expensive
    workload with its own configuration.

    :param enabled: Whether this graph accepts documents at all.
    :param chunker: How to split documents; ``None`` treats each document as
        one chunk.
    :param chunk_size: Maximum chunk size in characters.
    :param chunk_overlap: Overlap between consecutive chunks, in characters.
    :param vector_only: Build chunk vectors only, skipping entity extraction.
        The only mode that works without an extractor.
    :param make_community_summary: Summarize detected communities, which is
        what global search reads.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    chunker: Literal["simple"] | None = "simple"
    chunk_size: int = Field(default=1200, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    vector_only: bool = False
    make_community_summary: bool = True


class GraphSpec(BaseModel):
    """
    One graph this service serves.

    :param id: Identifier used in the URL. Kept to a safe slug because it is a
        path segment and selects a storage folder.
    :param storage_folder: Folder holding the built graph.
    :param language: Default answer language for this graph; falls back to the
        service default.
    :param settings_file: ``Settings`` JSON saved at build time. Applied while
        the graph is constructed and rolled back afterwards.
    :param embedder_dim: Embedding dimension the graph was built with.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    storage_folder: str = Field(min_length=1)
    language: str | None = None
    settings_file: str | None = None
    embedder_dim: int | None = None
    build: BuildSpec = Field(default_factory=BuildSpec)


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGU_API_", env_file=".env", extra="ignore"
    )

    backend: Literal["ragu", "stub"] = Field(
        default="ragu",
        description="'ragu' loads a real knowledge graph, 'stub' serves canned answers for local development",
    )

    graphs: list[GraphSpec] = Field(
        default_factory=list,
        description="Graphs to serve, as JSON. When empty, one graph named "
        "'default' is built from RAGU_API_STORAGE_FOLDER / RAGU_API_LANGUAGE.",
    )

    host: str = Field(
        default="127.0.0.1",
        description="Bind address. Loopback by default; the container image passes "
        "--host 0.0.0.0 explicitly, so exposing the service is a deliberate act.",
    )
    port: int = Field(default=8020, description="Bind port")

    storage_folder: str = Field(
        default="ragu_working_dir",
        description="Folder with the built graph. Set explicitly: the default Settings.storage_folder is timestamped "
        "per run and would point at an empty directory.",
    )
    language: str = Field(
        default="russian", description="Graph language passed to Settings.language"
    )
    settings_file: str | None = Field(
        default=None,
        description="Optional Settings JSON saved at build time (Settings.save) to load instead of the defaults",
    )
    embedder_dim: int | None = Field(
        default=None,
        description="Embedding dimension. None auto-detects it with a probe request on the first call.",
    )
    rate_min_delay: float | None = Field(
        default=None, description="Minimum delay between LLM calls, seconds"
    )
    rate_max_simultaneous: int | None = Field(
        default=None, description="Maximum simultaneous LLM calls"
    )
    llm_cache: str | None = Field(
        default=None,
        description="Path to the LLM response cache; None disables caching",
    )

    max_top_k: int = Field(
        default=100,
        gt=0,
        description="Upper bound applied to a client-supplied top_k / rerank_top_k; "
        "requests above it are clamped",
    )
    max_llm_calls_per_request: int | None = Field(
        default=None,
        gt=0,
        description="LLM calls one request may make. Global search costs one per "
        "surviving community, so this is the cap on a single expensive question.",
    )
    max_tokens_per_request: int | None = Field(
        default=None,
        gt=0,
        description="Approximate tokens one request may spend; counted with the "
        "tokenizer, since the clients do not surface provider usage.",
    )
    max_concurrent_generations: int | None = Field(
        default=None,
        gt=0,
        description="Generations that may run at once. Beyond it requests answer "
        "429 instead of fanning out to the LLM.",
    )
    api_keys: str = Field(
        default="",
        description="Comma-separated API keys. Empty leaves the service open, "
        "which suits a local stub and nothing else.",
    )
    cors_origins: str = Field(
        default="",
        description="Comma-separated allowed origins for browser clients. Empty "
        "sends no CORS headers.",
    )
    max_body_bytes: int = Field(
        default=32 * 1024 * 1024,
        gt=0,
        description="Largest request body the service will read; ingestion takes "
        "whole documents, so this is generous rather than tight",
    )
    request_timeout: float | None = Field(
        default=300.0,
        gt=0,
        description="Seconds a single request may take before it is abandoned. "
        "Global search is N+1 LLM calls, so this is minutes, not seconds.",
    )
    rerank_timeout: float | None = Field(
        default=10.0,
        gt=0,
        description="Seconds to wait for the reranker before answering without it. "
        "The model runs outside this process, so it can be slow or gone.",
    )
    engine_cache_size: int = Field(
        default=32,
        gt=0,
        description="How many (mode, language) engines to keep built. Clients choose "
        "the language, so the cache is bounded.",
    )
    max_batch_size: int = Field(
        default=50,
        gt=0,
        description="Maximum number of queries accepted by a /batch route",
    )
    min_cluster_size_floor: int = Field(
        default=1,
        gt=0,
        description="Lower bound applied to a client-supplied global min_cluster_size. "
        "Global search rates every surviving community with its own LLM call, so this is "
        "the knob that caps the cost of one request; raise it on large graphs.",
    )

    stub_missing_capabilities: str = Field(
        default="",
        description="Stub backend only: comma-separated capabilities to report as missing "
        "(entity_graph, community_summaries, vector_index)",
    )


    @field_validator("graphs")
    @classmethod
    def _reject_duplicate_ids(cls, value: list[GraphSpec]) -> list[GraphSpec]:
        seen = [spec.id for spec in value]
        duplicates = sorted({name for name in seen if seen.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate graph ids {duplicates}")
        return value

    def resolved_graphs(self) -> list[GraphSpec]:
        """
        The graphs to serve, including the single-graph fallback.

        :return: One spec per graph, never empty.
        """
        if self.graphs:
            return list(self.graphs)
        return [
            GraphSpec(
                id=DEFAULT_GRAPH_ID,
                storage_folder=self.storage_folder,
                language=self.language,
                settings_file=self.settings_file,
                embedder_dim=self.embedder_dim,
            )
        ]

    @field_validator("stub_missing_capabilities")
    @classmethod
    def _reject_unknown_capabilities(cls, value: str) -> str:
        """
        Fail on a misspelled capability instead of silently simulating nothing.
        """
        unknown = sorted(cls._split(value) - CAPABILITIES)
        if unknown:
            raise ValueError(
                f"unknown capabilities {unknown}; expected any of {sorted(CAPABILITIES)}"
            )
        return value

    @staticmethod
    def _split(value: str) -> set[str]:
        return {item.strip() for item in value.split(",") if item.strip()}

    def api_keys_set(self) -> set[str]:
        """
        The keys this service accepts. Empty means it is open.
        """
        return self._split(self.api_keys)

    def cors_origins_list(self) -> list[str]:
        """
        Origins allowed to call this service from a browser.
        """
        return sorted(self._split(self.cors_origins))

    def missing_capabilities(self) -> set[str]:
        """
        Capabilities the stub backend should report as absent.
        """
        return self._split(self.stub_missing_capabilities)
