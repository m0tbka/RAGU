"""
Service configuration.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ragu.api.models import CAPABILITIES

# Identifier of the graph served when none are configured explicitly.
DEFAULT_GRAPH_ID = "default"


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

    def missing_capabilities(self) -> set[str]:
        """
        Capabilities the stub backend should report as absent.
        """
        return self._split(self.stub_missing_capabilities)
