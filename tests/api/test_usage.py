"""
What a request costs, and how its time splits across stages.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "fastapi", reason="install the 'api' extra to test the search service"
)

from tests.api.support import (
    build_client,
    pause,
)


class TestUsageAccounting:
    """The client prices the request; the operator caps it. Same numbers."""

    def test_every_search_reports_usage(self):
        with build_client() as client:
            body = client.post("/v1/search/naive", json={"query": "q"}).json()
        assert body["usage"] is not None
        assert body["usage"]["estimated"] is True
        assert body["usage"]["calls"] == 0  # the stub calls no LLM

    def test_retrieve_and_batch_report_it_too(self):
        with build_client() as client:
            retrieve = client.post(
                "/v1/search/naive/retrieve", json={"query": "q"}
            ).json()
            batch = client.post(
                "/v1/search/naive/batch", json={"queries": ["a"]}
            ).json()
        assert retrieve["usage"] is not None
        assert batch["usage"] is not None

    async def test_calls_are_attributed_to_the_stage_that_made_them(self):
        from ragu.api.search import usage
        from ragu.api.search.usage import CountingLLM

        class Encoder:
            def encode(self, text):
                return text.split()

        class FakeLLM:
            async def chat_completion(self, conversation, *args, **kwargs):
                return "one two three"

            async def batch_chat_completion(self, conversations, *args, **kwargs):
                return ["one two"] * len(conversations)

        counted = CountingLLM(FakeLLM(), encoder=Encoder())
        record = usage.start()

        await counted.batch_chat_completion(
            [[{"role": "user", "content": "a b c d"}]] * 2,
            desc="QueryPlan decompose",
        )
        await counted.batch_chat_completion(
            [[{"role": "user", "content": "e f"}]],
            desc="NaiveSearch batch query",
        )

        assert record.calls == 3
        assert record.stages["QueryPlan decompose"].calls == 2
        assert record.stages["QueryPlan decompose"].prompt_tokens == 8
        assert record.stages["NaiveSearch batch query"].calls == 1
        assert record.total_tokens == record.prompt_tokens + record.completion_tokens

    async def test_a_call_budget_stops_a_runaway_request(self):
        from ragu.api.search import usage
        from ragu.api.errors import BudgetExceededError
        from ragu.api.search.usage import CountingLLM

        class FakeLLM:
            sent = 0

            async def batch_chat_completion(self, conversations, *args, **kwargs):
                self.sent += len(conversations)
                return [""] * len(conversations)

        llm = FakeLLM()
        counted = CountingLLM(llm)
        record = usage.start(max_calls=3)

        await counted.batch_chat_completion([[]] * 2, desc="GlobalSearch batch meta-eval")
        with pytest.raises(BudgetExceededError) as failure:
            await counted.batch_chat_completion(
                [[]] * 2, desc="GlobalSearch batch meta-eval"
            )

        assert failure.value.status_code == 429
        assert "min_cluster_size" in failure.value.message
        # Refused before it was sent: the two calls that fit are all that was paid.
        assert llm.sent == 2
        assert record.calls == 2

    async def test_a_token_budget_stops_one_too(self):
        from ragu.api.search import usage
        from ragu.api.errors import BudgetExceededError
        from ragu.api.search.usage import CountingLLM

        class Encoder:
            def encode(self, text):
                return text.split()

        class FakeLLM:
            async def batch_chat_completion(self, conversations, *args, **kwargs):
                return ["a b c d e"] * len(conversations)

        counted = CountingLLM(FakeLLM(), encoder=Encoder())
        usage.start(max_tokens=4)

        with pytest.raises(BudgetExceededError):
            await counted.batch_chat_completion([[]], desc="stage")



class TestBudgetBeforeSending:
    """
    A call that cannot fit the budget is refused before it leaves, not after.

    The budget used to be checked as answers came back, so a batch was paid for
    in full before anything could say it was over: global search sent every one
    of its 474 rating calls against a budget of 12 and then reported the overrun.
    """

    class FakeLLM:
        def __init__(self, answer=""):
            self.answer = answer
            self.sent = 0

        async def chat_completion(self, conversation, *args, **kwargs):
            self.sent += 1
            return self.answer

        async def batch_chat_completion(self, conversations, *args, **kwargs):
            import asyncio

            self.sent += len(conversations)
            # Yield, as a real call would, so concurrent batches interleave.
            await asyncio.sleep(0)
            return [self.answer] * len(conversations)

        async def stream_chat_completion(self, conversation, *args, **kwargs):
            self.sent += 1
            for word in ("a", "b", "c"):
                yield word

    class Words:
        def encode(self, text):
            return text.split()

    async def test_an_oversized_batch_is_refused_before_it_is_sent(self):
        from ragu.api.errors import BudgetExceededError
        from ragu.api.search import usage
        from ragu.api.search.usage import CountingLLM

        llm = self.FakeLLM()
        record = usage.start(max_calls=12)

        with pytest.raises(BudgetExceededError) as failure:
            await CountingLLM(llm).batch_chat_completion(
                [[]] * 474, desc="GlobalSearch batch meta-eval"
            )

        assert llm.sent == 0
        assert record.calls == 0
        assert "474" in failure.value.message
        assert "Refused before sending" in failure.value.message

    async def test_a_prompt_over_the_token_budget_is_not_sent(self):
        from ragu.api.errors import BudgetExceededError
        from ragu.api.search import usage
        from ragu.api.search.usage import CountingLLM

        llm = self.FakeLLM()
        usage.start(max_tokens=4)

        with pytest.raises(BudgetExceededError):
            await CountingLLM(llm, encoder=self.Words()).chat_completion(
                [{"role": "user", "content": "one two three four five"}], desc="stage"
            )

        assert llm.sent == 0

    async def test_concurrent_batches_cannot_overrun_it_together(self):
        # An ensemble runs its children side by side; each batch alone fits the
        # budget, and the two together must not.
        import asyncio

        from ragu.api.errors import BudgetExceededError
        from ragu.api.search import usage
        from ragu.api.search.usage import CountingLLM

        llm = self.FakeLLM()
        counted = CountingLLM(llm)
        record = usage.start(max_calls=3)

        outcomes = await asyncio.gather(
            counted.batch_chat_completion([[]] * 2, desc="local"),
            counted.batch_chat_completion([[]] * 2, desc="global"),
            return_exceptions=True,
        )

        assert sum(isinstance(outcome, BudgetExceededError) for outcome in outcomes) == 1
        assert llm.sent == 2
        assert record.calls == 2

    async def test_an_abandoned_stream_still_counts_its_call(self):
        from ragu.api.search import usage
        from ragu.api.search.usage import CountingLLM

        record = usage.start()
        stream = CountingLLM(self.FakeLLM()).stream_chat_completion(
            [{"role": "user", "content": "q"}], desc="stage"
        )

        assert await stream.__anext__() == "a"
        await stream.aclose()

        assert record.calls == 1

    async def test_a_delivered_stream_is_recorded_not_refused(self):
        # Its answer is already with the client; an overrun can only stop the
        # next call, which the check on dispatch then does.
        from ragu.api.search import usage
        from ragu.api.search.usage import CountingLLM

        record = usage.start(max_tokens=2)
        counted = CountingLLM(self.FakeLLM(), encoder=self.Words())

        words = [
            word
            async for word in counted.stream_chat_completion(
                [{"role": "user", "content": ""}], desc="stage"
            )
        ]

        assert words == ["a", "b", "c"]
        assert record.completion_tokens == 3

class TestStageTimings:
    """A dashboard splits retrieval from generation without measuring it itself."""

    async def test_generation_is_timed_where_it_happens(self):

        from ragu.api.search import usage as usage_module

        class SlowLLM:
            async def chat_completion(self, conversation, *args, **kwargs):
                await pause(0.02)
                return "answer"

        usage_module.start()
        counting = usage_module.CountingLLM(SlowLLM())
        await counting.chat_completion([{"content": "prompt"}], desc="local")

        stage = usage_module.current().stages["local"]
        assert stage.calls == 1
        assert stage.generation_ms >= 15

    async def test_retrieval_is_what_the_llm_did_not_take(self):

        from ragu.api.search import usage as usage_module

        class SlowLLM:
            async def chat_completion(self, conversation, *args, **kwargs):
                await pause(0.02)
                return "answer"

        usage_module.start()
        counting = usage_module.CountingLLM(SlowLLM())
        with usage_module.measure_retrieval("local"):
            await pause(0.02)
            await counting.chat_completion([{"content": "prompt"}], desc="local")

        stage = usage_module.current().stages["local"]
        # Roughly 40ms total, of which roughly 20 was the model.
        assert stage.generation_ms >= 15
        assert stage.retrieval_ms >= 10
        assert stage.retrieval_ms < stage.generation_ms + 40

    def test_the_timings_reach_the_wire(self):
        with build_client() as client:
            usage = client.post("/v1/search/naive", json={"query": "q"}).json()["usage"]
        # The stub makes no LLM calls, so there is nothing to attribute; the
        # fields exist and stay null rather than reporting a fabricated zero.
        assert "stages" in usage


class TestRerankStage:
    """
    The reranker's time is its own stage, and parallel calls are not double-counted.
    """

    async def test_reranking_is_recorded_on_its_own_stage(self):

        from ragu.api.search import usage as usage_module
        from ragu.api.search.reranking import ForgivingScorer
        from ragu.models.scorer import Scorer

        class Slow(Scorer):
            async def score(self, text_1, text_2, **kwargs):
                await pause(0.02)
                return [(i, 1.0) for i in range(len(text_2))]

        usage_module.start()
        await ForgivingScorer(Slow()).score("q", ["a", "b"])

        stage = usage_module.current().stages[usage_module.RERANK_STAGE]
        assert stage.calls == 1
        assert stage.rerank_ms >= 15

    async def test_rerank_calls_are_not_llm_calls(self):
        # The call total is what the LLM budget counts.
        from ragu.api.search import usage as usage_module
        from ragu.api.search.reranking import ForgivingScorer
        from ragu.models.scorer import Scorer

        class Fast(Scorer):
            async def score(self, text_1, text_2, **kwargs):
                return [(0, 1.0)]

        usage_module.start()
        await ForgivingScorer(Fast()).score("q", ["a"])
        await ForgivingScorer(Fast()).score("q", ["a"])

        assert usage_module.current().stages[usage_module.RERANK_STAGE].calls == 2
        assert usage_module.current().calls == 0

    async def test_parallel_calls_are_timed_once_not_summed(self):
        # The engines fan out through asyncio.gather. Four parallel 30 ms calls
        # took about 30 ms of the request, not 120.
        import asyncio

        from ragu.api.search import usage as usage_module
        from ragu.api.search.reranking import ForgivingScorer
        from ragu.models.scorer import Scorer

        class Slow(Scorer):
            async def score(self, text_1, text_2, **kwargs):
                await pause(0.03)
                return [(0, 1.0)]

        usage_module.start()
        scorer = ForgivingScorer(Slow())
        await asyncio.gather(*[scorer.score("q", ["a"]) for _ in range(4)])

        spent = usage_module.current().stages[usage_module.RERANK_STAGE].rerank_ms
        assert 25 <= spent < 80

    async def test_a_timed_out_reranker_still_counts_its_time(self):
        import asyncio

        from ragu.api.search import usage as usage_module
        from ragu.api.search.reranking import ForgivingScorer
        from ragu.models.scorer import Scorer

        class Hanging(Scorer):
            async def score(self, text_1, text_2, **kwargs):
                await asyncio.sleep(10)

        usage_module.start()
        # The timeout is asyncio's own timer, which can fire up to one clock
        # resolution (15.6 ms on Windows) early: the floor sits below it by more.
        await ForgivingScorer(Hanging(), timeout=0.05).score("q", ["a"])
        assert usage_module.current().stages[usage_module.RERANK_STAGE].rerank_ms >= 30

    async def test_retrieval_excludes_the_rerankers_time(self):

        from ragu.api.search import usage as usage_module
        from ragu.api.search.reranking import ForgivingScorer
        from ragu.models.scorer import Scorer

        class Slow(Scorer):
            async def score(self, text_1, text_2, **kwargs):
                await pause(0.04)
                return [(0, 1.0)]

        usage_module.start()
        with usage_module.measure_retrieval("local"):
            await pause(0.01)
            await ForgivingScorer(Slow()).score("q", ["a"])

        record = usage_module.current()
        assert record.stages["local"].retrieval_ms < 35
        assert record.stages[usage_module.RERANK_STAGE].rerank_ms >= 35

    async def test_parallel_generation_is_not_double_counted_either(self):
        import asyncio

        from ragu.api.search import usage as usage_module

        class SlowLLM:
            async def chat_completion(self, conversation, *args, **kwargs):
                await pause(0.03)
                return "answer"

        usage_module.start()
        counting = usage_module.CountingLLM(SlowLLM())
        await asyncio.gather(*[
            counting.chat_completion([{"content": "p"}], desc="local") for _ in range(4)
        ])
        stage = usage_module.current().stages["local"]
        assert stage.calls == 4
        assert 25 <= stage.generation_ms < 80

    def test_the_stage_reaches_the_wire(self):
        from ragu.api.models import StageUsageModel

        assert "rerank_ms" in StageUsageModel.model_fields
