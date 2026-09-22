"""
Reranking that degrades instead of failing.

The reranker is supplied from outside — on the consumer's deployment the model
runs in a separate container — so the service never constructs one. It only
wraps whatever it is given so that a slow or broken reranker costs ranking
quality rather than the whole answer.
"""

import asyncio
import math
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from typing_extensions import override

from ragu.api.usage import measure_rerank
from ragu.common.logger import logger
from ragu.models.scorer import Scorer


@dataclass
class _Report:
    """
    One request's reranking record.
    """

    failure: str | None = None


# Set per request. Engines are cached, so the scorer they hold is shared and
# cannot carry per-request state itself.
#
# The variable holds a mutable record rather than the message itself: every
# engine reaches the scorer through ``asyncio.gather`` (``Scorer.batch_score``,
# ``LocalSearchEngine._retrieve``), which runs each call in a child task with
# its own copy of the context. A ``set()`` there is discarded with the task,
# while a mutation of the shared record is visible to the request that owns it.
_report: ContextVar["_Report | None"] = ContextVar("rerank_report", default=None)


def reset_rerank_report() -> None:
    """
    Start a fresh reranking record for this request.
    """
    _report.set(_Report())


def rerank_failure() -> str | None:
    """
    Why reranking did not happen on this request, if it did not.
    """
    report = _report.get()
    return report.failure if report is not None else None


def reranker_from_env() -> Scorer | None:
    """
    Build a client for the reranker the environment names, if it names one.

    ``create_app`` takes a reranker as a parameter because the model runs in a
    container of its own; this is the composition that ``python -m ragu.api``
    does with it. It reads ``RERANKER_BASE_URL``, ``RERANKER_MODEL_NAME`` and
    ``RERANKER_API_KEY`` (falling back to ``LLM_API_KEY``), the same variables
    ``Env`` documents.

    Never fatal. Credentials the graph also needs are reported by the backend
    through ``/health``, which is where an operator looks; crashing here instead
    would turn that into a container that restarts without saying why. A
    reranker that is named but incomplete is logged as an error and left out —
    ``engines.reranked`` in every answer then says it did not run.

    :return: The reranker, or ``None`` when none is configured.
    """
    from pydantic import ValidationError

    from ragu.common.env import Env
    from ragu.models.openai import CachedAsyncOpenAI
    from ragu.models.scorer import ScorerOpenAI

    try:
        env = Env()
    except ValidationError:
        return None
    if not env.reranker_base_url:
        return None
    if not env.reranker_model_name:
        logger.error(
            "RERANKER_BASE_URL is set but RERANKER_MODEL_NAME is not; "
            "running without a reranker."
        )
        return None

    client = CachedAsyncOpenAI(
        base_url=env.reranker_base_url,
        api_key=env.reranker_api_key or env.llm_api_key,
    )
    logger.info(
        "Reranker {} at {}", env.reranker_model_name, env.reranker_base_url
    )
    return ScorerOpenAI(client=client, model_name=env.reranker_model_name)


class ForgivingScorer(Scorer):
    """
    Scorer proxy that answers with the original order when the real one fails.

    ``_rerank_items_scored`` lets an exception from ``score`` propagate, which would
    turn a reranker outage into a 500 for a request the engines could still
    answer. The failure is recorded for the response instead.
    """

    def __init__(self, scorer: Scorer, timeout: float | None = None):
        """
        :param scorer: The real reranker.
        :param timeout: Seconds to wait before giving up on it.
        """
        self.scorer = scorer
        self.timeout = timeout

    @override
    async def score(
        self,
        text_1: str,
        text_2: list[str],
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        # Timed as its own stage whether it succeeds or not: a reranker that
        # times out cost the request its whole timeout.
        with measure_rerank():
            try:
                call = self.scorer.score(text_1, text_2, **kwargs)
                if self.timeout is not None:
                    return await asyncio.wait_for(call, self.timeout)
                return await call
            except asyncio.TimeoutError:
                return self._degrade(f"reranker timed out after {self.timeout}s", text_2)
            except Exception as exc:
                return self._degrade(f"{type(exc).__name__}: {exc}", text_2)

    @staticmethod
    def _degrade(reason: str, text_2: list[str]) -> list[tuple[int, float]]:
        """
        Record the failure and keep the retrieval order untouched.
        """
        report = _report.get()
        if report is not None:
            report.failure = reason
        logger.warning("Reranking skipped: {}", reason)
        # NaN rather than 0.0: the Scorer contract wants a float, but a zero
        # is a real score and would reach a client as one. NaN reads as "not
        # scored" everywhere downstream and serializes as null.
        return [(index, math.nan) for index in range(len(text_2))]
