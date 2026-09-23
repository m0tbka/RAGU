"""
What each search mode needs a graph to hold, and what the service says when
the graph does not hold it.

Measured, not inferred: ``GraphStats`` counts the stores at startup, so a mode
the graph cannot serve is refused before a generation call is paid for.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from ragu.api.models import DEFAULT_MIX_ENGINES, Capability, GraphStatsResponse, SearchMode

# The graph has no store this mode can read at all.
GLOBAL_MISSING_MESSAGE = (
    "This graph carries no community summaries, so global search cannot run. Rebuild it "
    "with make_community_summary enabled, or use local or naive search."
)
LOCAL_MISSING_MESSAGE = (
    "This graph carries no entity index, so local search cannot run. Rebuild it with "
    "entity extraction enabled, or use naive search."
)
NAIVE_MISSING_MESSAGE = (
    "This graph carries no chunk index, so naive search cannot run. Rebuild it with chunk "
    "vectorization enabled, or use local or global search."
)

# The store exists, but this query retrieved nothing from it.
GLOBAL_UNAVAILABLE_MESSAGE = (
    "Global search found no community summaries to answer from. The graph may have been "
    "built without them. Try local or naive search instead."
)
LOCAL_UNAVAILABLE_MESSAGE = (
    "Local search found no entities, relations or chunks for this query. The graph may have "
    "been built without entity extraction or your query was too unrelated to the text corpus."
)
NAIVE_UNAVAILABLE_MESSAGE = (
    "Naive search found no matching chunks. The corpus has nothing on this query, or the "
    "graph carries no chunk index."
)

MIX_MISSING_MESSAGE = (
    "Mixed search needs everything its child engines read, and this graph is missing one "
    "of those stores. Drop that engine from `engines`, or use whichever single mode this "
    "graph supports."
)
MIX_UNAVAILABLE_MESSAGE = (
    "Mixed search found nothing for this query in either the entity index or the chunk "
    "index."
)


@dataclass(frozen=True, slots=True)
class ModeRequirement:
    """
    What one search mode needs from the graph, and what to say when it is absent.

    :param requires: Capabilities the mode needs, all of them. The first one
        the graph lacks is reported as ``missing_capability``.
    :param missing_message: Explanation when the graph cannot serve the mode.
    :param no_evidence_message: Explanation when the graph can serve the mode
        but this query retrieved nothing.
    """

    requires: tuple[Capability, ...]
    missing_message: str
    no_evidence_message: str


MODE_REQUIREMENTS: dict[SearchMode, ModeRequirement] = {
    "global": ModeRequirement(
        requires=("community_summaries",),
        missing_message=GLOBAL_MISSING_MESSAGE,
        no_evidence_message=GLOBAL_UNAVAILABLE_MESSAGE,
    ),
    "local": ModeRequirement(
        requires=("entity_graph",),
        missing_message=LOCAL_MISSING_MESSAGE,
        no_evidence_message=LOCAL_UNAVAILABLE_MESSAGE,
    ),
    "naive": ModeRequirement(
        requires=("vector_index",),
        missing_message=NAIVE_MISSING_MESSAGE,
        no_evidence_message=NAIVE_UNAVAILABLE_MESSAGE,
    ),
    "mix": ModeRequirement(
        requires=("entity_graph", "vector_index"),
        missing_message=MIX_MISSING_MESSAGE,
        no_evidence_message=MIX_UNAVAILABLE_MESSAGE,
    ),
}


def required_capabilities(
    mode: SearchMode, mix_engines: Sequence[str] | None = None
) -> tuple[Capability, ...]:
    """
    What the graph must hold for this request to run.

    ``mix`` has no fixed answer: it needs whatever its children read, and the
    request chooses the children. Asking for the local and naive pair needs no
    community summaries, so a graph built without them still serves mix — which
    a fixed requirement would have refused.

    :param mode: Search mode about to run.
    :param mix_engines: For ``mix``, the leaf engines selected. ``None`` means
        the default ensemble.
    :return: The capabilities needed, in a stable order, without repeats.
    """
    if mode != "mix":
        return MODE_REQUIREMENTS[mode].requires

    needed: list[Capability] = []
    for engine in mix_engines or DEFAULT_MIX_ENGINES:
        for capability in MODE_REQUIREMENTS[engine].requires:
            if capability not in needed:
                needed.append(capability)
    return tuple(needed)


@dataclass(frozen=True, slots=True)
class GraphStats:
    """
    Sizes of the stores the search modes read, counted once at startup.

    Counting them up front is what lets the service refuse a mode the graph
    cannot serve *before* paying for a generation call, and what makes
    ``graph_loaded`` mean "can answer searches" rather than "an object was
    constructed".
    """

    entities: int = 0
    relations: int = 0
    chunks: int = 0
    community_summaries: int = 0

    @property
    def is_empty(self) -> bool:
        """
        Whether the graph has nothing any mode could read.
        """
        return not (self.entities or self.chunks or self.community_summaries)

    def has(self, capability: Capability) -> bool:
        """
        Whether the store behind one capability holds anything.

        :param capability: Capability to check.
        :return: ``True`` when the store is non-empty.
        """
        if capability == "community_summaries":
            return self.community_summaries > 0
        if capability == "entity_graph":
            return self.entities > 0
        return self.chunks > 0

    def supports(
        self, mode: SearchMode, mix_engines: Sequence[str] | None = None
    ) -> bool:
        """
        Whether the graph holds everything this mode reads.

        :param mode: Search mode to check.
        :param mix_engines: For ``mix``, the leaf engines the request selected.
        :return: ``True`` when the mode can be served.
        """
        return self.missing_for(mode, mix_engines) is None

    def missing_for(
        self, mode: SearchMode, mix_engines: Sequence[str] | None = None
    ) -> Capability | None:
        """
        The first capability this mode needs and the graph does not have.

        :param mode: Search mode to check.
        :param mix_engines: For ``mix``, the leaf engines the request selected.
            ``None`` means the default ensemble.
        :return: The missing capability, or ``None`` when the mode is supported.
        """
        for capability in required_capabilities(mode, mix_engines):
            if not self.has(capability):
                return capability
        return None

    def to_response(self) -> GraphStatsResponse:
        """
        Render the counts for the health endpoint.
        """
        return GraphStatsResponse(
            entities=self.entities,
            relations=self.relations,
            chunks=self.chunks,
            community_summaries=self.community_summaries,
        )
