"""
Reranking that degrades instead of failing.

The reranker is supplied from outside — on the consumer's deployment the model
runs in a separate container — so the service never constructs one. It only
wraps whatever it is given so that a slow or broken reranker costs ranking
quality rather than the whole answer.
"""

import asyncio
from contextvars import ContextVar
from typing import Any

from typing_extensions import override

from ragu.common.logger import logger
from ragu.models.scorer import Scorer

# Set per request. Engines are cached, so the scorer they hold is shared and
# cannot carry per-request state itself.
_rerank_failure: ContextVar[str | None] = ContextVar("rerank_failure", default=None)


def reset_rerank_report() -> None:
    """
    Start a fresh reranking record for this request.
    """
    _rerank_failure.set(None)


def rerank_failure() -> str | None:
    """
    Why reranking did not happen on this request, if it did not.
    """
    return _rerank_failure.get()


class ForgivingScorer(Scorer):
    """
    Scorer proxy that answers with the original order when the real one fails.

    ``_rerank_items`` lets an exception from ``score`` propagate, which would
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
        _rerank_failure.set(reason)
        logger.warning("Reranking skipped: {}", reason)
        return [(index, 0.0) for index in range(len(text_2))]
