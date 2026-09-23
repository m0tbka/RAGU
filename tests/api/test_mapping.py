"""
Engine results rendered in the wire schema: answers, sources and scores.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "fastapi", reason="install the 'api' extra to test the search service"
)

from ragu.api.search.mapping import (
    extract_sources,
    extract_subqueries,
    split_report_title,
    to_outcome,
)
from ragu.common.prompts.default_models import GlobalSearchContextModel
from ragu.graph.types import CommunitySummary, Entity, Relation
from ragu.search_engine.base_engine import SearchEngineResponse
from ragu.search_engine.global_search import (
    GlobalSearchResult,
    GlobalSearchRetrieve,
)
from ragu.search_engine.local_search import (
    LocalSearchResult,
    LocalSearchRetrieve,
)
from ragu.search_engine.naive_search import (
    NaiveSearchResult,
    NaiveSearchRetrieve,
)
from tests.api.support import (
    make_chunk,
    make_entity,
    make_retrieval,
)


class TestResponseConversion:
    """The adapter is pinned to the real engine result types."""

    def test_naive_chunks_keep_their_ids_and_scores(self):
        chunks = [make_chunk("first"), make_chunk("second", 1)]
        retrieval = NaiveSearchRetrieve(
            query="q",
            result=NaiveSearchResult(
                chunks=chunks, scores=[0.9, 0.7], documents_id=["doc-1"]
            ),
        )
        sources = extract_sources(retrieval)
        assert [source.type for source in sources] == ["chunk", "chunk"]
        assert [source.id for source in sources] == [chunk.id for chunk in chunks]
        assert [source.score for source in sources] == [0.9, 0.7]
        assert sources[0].content == "first"

    def test_local_result_yields_entities_relations_summaries_and_chunks(self):
        entity = make_entity("Сенкевич")
        relation = Relation(
            subject_id="a",
            object_id="b",
            subject_name="Сенкевич",
            object_name="Польша",
            relation_type="born_in",
            description="родился в",
        )
        retrieval = LocalSearchRetrieve(
            query="q",
            result=LocalSearchResult(
                entities=[entity],
                relations=[relation],
                summaries=[CommunitySummary(id="com-1", summary="сводка")],
                chunks=[make_chunk("chunk text")],
            ),
        )
        sources = extract_sources(retrieval)
        assert [source.type for source in sources] == [
            "entity",
            "relation",
            "community_summary",
            "chunk",
        ]
        assert sources[0].id == entity.id
        assert "Сенкевич" in sources[1].content
        assert sources[2].content == "сводка"

    def test_global_insights_become_community_summaries_with_ratings(self):
        insight = GlobalSearchContextModel(
            reasoning="r", response="общий вывод", rating=8.0
        )
        retrieval = GlobalSearchRetrieve(
            query="q", result=GlobalSearchResult(insights=[insight])
        )
        sources = extract_sources(retrieval)
        assert (sources[0].type, sources[0].content, sources[0].score) == (
            "community_summary",
            "общий вывод",
            8.0,
        )

    def test_empty_retrieval_yields_no_sources(self):
        retrieval = NaiveSearchRetrieve(query="q", result=NaiveSearchResult())
        assert extract_sources(retrieval) == []

    def test_an_unmodelled_result_keeps_its_rendered_context(self):
        class UnknownResult:
            pass

        class UnknownRetrieve:
            result = UnknownResult()

            def to_text(self):
                return "rendered context"

        sources = extract_sources(UnknownRetrieve())
        assert [(s.id, s.type, s.content) for s in sources] == [
            ("retrieval_context", "context", "rendered context")
        ]

    def test_query_plan_payload_becomes_subqueries(self):
        retrieval = make_retrieval("text")
        payload = {
            "sq-1": SearchEngineResponse(
                query="Кто написал?", response="Сенкевич", retrieval=retrieval
            ),
            "sq-2": SearchEngineResponse(
                query="Из какой страны?", response="Польша", retrieval=retrieval
            ),
        }
        assert [(item.query, item.answer) for item in extract_subqueries(payload)] == [
            ("Кто написал?", "Сенкевич"),
            ("Из какой страны?", "Польша"),
        ]

    def test_a_response_without_a_plan_has_no_subqueries(self):
        answer, sources, subqueries = to_outcome(
            SearchEngineResponse(
                query="q", response="ответ", retrieval=make_retrieval("text")
            ),
            used_query_plan=False,
        )
        assert answer == "ответ"
        assert subqueries == []
        assert len(sources) == 1

    def test_a_plan_keeps_the_evidence_of_every_subquery(self):
        # The plan answers the top-level query from the sink subquery alone, so
        # without this the evidence behind the other subqueries is dropped even
        # though their answers are returned.
        sink_retrieval = make_retrieval("sink evidence")
        payload = {
            "sq-1": SearchEngineResponse(
                query="Кто написал?",
                response="Сенкевич",
                retrieval=make_retrieval("branch evidence"),
            ),
            "sq-2": SearchEngineResponse(
                query="Итог?", response="ответ", retrieval=sink_retrieval
            ),
        }
        _, sources, subqueries = to_outcome(
            SearchEngineResponse(
                query="q", response="ответ", retrieval=sink_retrieval, payload=payload
            ),
            used_query_plan=True,
        )
        assert len(subqueries) == 2
        assert {source.content for source in sources} == {
            "sink evidence",
            "branch evidence",
        }

    def test_sources_shared_between_subqueries_appear_once(self):
        shared = make_retrieval("shared evidence")
        payload = {
            "sq-1": SearchEngineResponse(query="a", response="a", retrieval=shared),
            "sq-2": SearchEngineResponse(query="b", response="b", retrieval=shared),
        }
        _, sources, _ = to_outcome(
            SearchEngineResponse(
                query="q", response="ответ", retrieval=shared, payload=payload
            ),
            used_query_plan=True,
        )
        assert len(sources) == 1

    def test_an_empty_sink_still_reports_the_subquery_evidence(self):
        # An empty sink retrieval used to make the whole request a 409 even when
        # the branches had retrieved plenty.
        payload = {
            "sq-1": SearchEngineResponse(
                query="Кто написал?",
                response="Сенкевич",
                retrieval=make_retrieval("branch evidence"),
            )
        }
        _, sources, _ = to_outcome(
            SearchEngineResponse(
                query="q",
                response="ответ",
                retrieval=NaiveSearchRetrieve(query="q", result=NaiveSearchResult()),
                payload=payload,
            ),
            used_query_plan=True,
        )
        assert [source.content for source in sources] == ["branch evidence"]


class TestSourceMeta:
    """
    Sources carry their typed fields, so a trace does not need a fetch per source.

    Built from real engine result objects rather than the stub: this is the path
    that decides whether the fields reach the wire at all.
    """

    def test_a_chunk_source_carries_its_document(self):
        retrieval = NaiveSearchRetrieve(
            query="q",
            result=NaiveSearchResult(chunks=[make_chunk("text", 3)], scores=[0.5]),
        )
        source = extract_sources(retrieval)[0]
        assert source.meta.kind == "chunk"
        assert source.meta.doc_id == "doc-1"
        assert source.meta.chunk_order_idx == 3

    def test_an_entity_source_carries_its_name_type_and_provenance(self):
        entity = Entity(
            entity_name="Сенкевич",
            entity_type="PERSON",
            description="писатель",
            source_chunk_id=["chunk-1", "chunk-2"],
            clusters=[{"cluster_id": 7, "level": 0}],
        )
        retrieval = LocalSearchRetrieve(
            query="q", result=LocalSearchResult(entities=[entity])
        )
        meta = extract_sources(retrieval)[0].meta
        assert meta.kind == "entity"
        assert meta.name == "Сенкевич"
        assert meta.type == "PERSON"
        assert meta.communities == ["7"]
        assert meta.source_chunk_ids == ["chunk-1", "chunk-2"]

    def test_a_relation_source_carries_both_ends_and_its_strength(self):
        relation = Relation(
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="Сенкевич",
            object_name="Польша",
            relation_type="born_in",
            description="родился в",
            relation_strength=0.75,
            source_chunk_id=["chunk-1"],
        )
        retrieval = LocalSearchRetrieve(
            query="q", result=LocalSearchResult(relations=[relation])
        )
        meta = extract_sources(retrieval)[0].meta
        assert meta.kind == "relation"
        assert (meta.subject_id, meta.object_id) == ("ent-1", "ent-2")
        assert (meta.subject_name, meta.object_name) == ("Сенкевич", "Польша")
        assert meta.type == "born_in"
        assert meta.strength == 0.75
        assert meta.source_chunk_ids == ["chunk-1"]

    def test_a_community_source_splits_the_report_title_from_the_body(self):
        summary = CommunitySummary(
            id="com-1",
            summary="Report title: Сенкевич и Польша\nReport summary: текст",
        )
        retrieval = LocalSearchRetrieve(
            query="q", result=LocalSearchResult(summaries=[summary])
        )
        source = extract_sources(retrieval)[0]
        assert source.meta.kind == "community_summary"
        assert source.meta.title == "Сенкевич и Польша"
        assert source.content == "Report summary: текст"

    def test_a_global_insight_admits_it_has_no_community(self):
        # GlobalSearchResult carries what the LLM wrote about a community, not
        # which community it wrote about, so the fields stay empty rather than
        # being invented.
        retrieval = GlobalSearchRetrieve(
            query="q",
            result=GlobalSearchResult(
                insights=[
                    GlobalSearchContextModel(reasoning="r", response="a", rating=7.0)
                ]
            ),
        )
        meta = extract_sources(retrieval)[0].meta
        assert meta.kind == "community_summary"
        assert meta.level is None and meta.cluster_id is None

    def test_a_summary_in_another_shape_is_left_alone(self):
        assert split_report_title("just a summary") == (None, "just a summary")
        assert split_report_title(None) == (None, None)


class TestLocalSourceScores:
    """
    Local sources carry the relevance score the engine actually had.

    Only naive chunks and global insights used to carry one, so a trace showed
    0.0 against every entity and relation — a number, and a wrong one.
    """

    def test_an_entity_carries_the_vector_score_that_retrieved_it(self):
        entity = make_entity("Сенкевич")
        retrieval = LocalSearchRetrieve(
            query="q",
            result=LocalSearchResult(entities=[entity]),
            metrics={"entities": [{"id": entity.id, "relevance_score": 0.82}]},
        )
        assert extract_sources(retrieval)[0].score == 0.82

    def test_the_rerankers_score_wins_when_one_ran(self):
        entity = make_entity("Сенкевич")
        retrieval = LocalSearchRetrieve(
            query="q",
            result=LocalSearchResult(entities=[entity]),
            metrics={
                "entities": [{"id": entity.id, "relevance_score": 0.82}],
                "rerank_scores": {"entities": {entity.id: 0.97}},
            },
        )
        assert extract_sources(retrieval)[0].score == 0.97

    def test_a_relation_scored_by_the_reranker_carries_it(self):
        relation = Relation(
            subject_id="a", object_id="b", subject_name="A", object_name="B",
            relation_type="rel", description="d", relation_strength=1.0,
            source_chunk_id=[],
        )
        retrieval = LocalSearchRetrieve(
            query="q",
            result=LocalSearchResult(relations=[relation]),
            metrics={"rerank_scores": {"relations": {relation.id: 0.64}}},
        )
        assert extract_sources(retrieval)[0].score == 0.64

    def test_an_unscored_relation_is_null_not_zero(self):
        # Without a reranker a relation was selected through the graph and has
        # no relevance score. null says so; 0.0 would claim it is irrelevant.
        relation = Relation(
            subject_id="a", object_id="b", subject_name="A", object_name="B",
            relation_type="rel", description="d", relation_strength=1.0,
            source_chunk_id=[],
        )
        retrieval = LocalSearchRetrieve(
            query="q", result=LocalSearchResult(relations=[relation]), metrics={}
        )
        assert extract_sources(retrieval)[0].score is None

    def test_a_degraded_rerank_score_never_reaches_the_wire(self):
        import math

        relation = Relation(
            subject_id="a", object_id="b", subject_name="A", object_name="B",
            relation_type="rel", description="d", relation_strength=1.0,
            source_chunk_id=[],
        )
        retrieval = LocalSearchRetrieve(
            query="q",
            result=LocalSearchResult(relations=[relation]),
            metrics={"rerank_scores": {"relations": {relation.id: math.nan}}},
        )
        assert extract_sources(retrieval)[0].score is None

    async def test_the_engine_keeps_the_rerankers_scores(self):
        # The core used to throw these away: _rerank_items kept the order and
        # dropped the score.
        from ragu.search_engine.search_functional import _rerank_items_scored

        class Reversing:
            async def score(self, query, texts, **kwargs):
                return [(i, 1.0 - i / 10) for i in reversed(range(len(texts)))]

        scored = await _rerank_items_scored("q", ["a", "b", "c"], str, Reversing())
        assert scored == [("c", 0.8), ("b", 0.9), ("a", 1.0)]

    async def test_without_a_reranker_scores_are_absent(self):
        from ragu.search_engine.search_functional import _rerank_items_scored

        scored = await _rerank_items_scored("q", ["a", "b"], str, None)
        assert scored == [("a", None), ("b", None)]
