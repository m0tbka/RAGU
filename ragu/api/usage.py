"""
What one request spent, and the ceiling on it.

The client shows cost in its interface and the operator caps it, and both need
the same numbers, so they are collected once here.

Token counts are measured with the tokenizer, not read from the provider: the
LLM clients return the parsed answer, not the raw response, so ``usage`` never
reaches this layer. The counts are therefore close rather than exact — good
enough to price a request and to stop a runaway one, and labelled ``estimated``
so nobody bills from them.
"""

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from typing_extensions import override

from ragu.api.errors import BudgetExceededError
from ragu.models.llm import LLM

# The stage names the engines already pass as ``desc``.
UNKNOWN_STAGE = "unknown"


@dataclass
class StageUsage:
    """
    What one stage of a request cost.
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Usage:
    """
    What a whole request cost, broken down by stage.
    """

    stages: dict[str, StageUsage] = field(default_factory=dict)

    @property
    def calls(self) -> int:
        return sum(stage.calls for stage in self.stages.values())

    @property
    def prompt_tokens(self) -> int:
        return sum(stage.prompt_tokens for stage in self.stages.values())

    @property
    def completion_tokens(self) -> int:
        return sum(stage.completion_tokens for stage in self.stages.values())

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record(
        self, stage: str, calls: int, prompt_tokens: int, completion_tokens: int
    ) -> None:
        entry = self.stages.setdefault(stage, StageUsage())
        entry.calls += calls
        entry.prompt_tokens += prompt_tokens
        entry.completion_tokens += completion_tokens


_usage: ContextVar[Usage | None] = ContextVar("usage", default=None)
_budget: ContextVar[tuple[int | None, int | None]] = ContextVar(
    "budget", default=(None, None)
)


def start(max_calls: int | None = None, max_tokens: int | None = None) -> Usage:
    """
    Begin accounting for this request.

    :param max_calls: LLM calls this request may make, or ``None`` for no cap.
    :param max_tokens: Tokens this request may spend, or ``None`` for no cap.
    :return: The fresh usage record.
    """
    usage = Usage()
    _usage.set(usage)
    _budget.set((max_calls, max_tokens))
    return usage


def current() -> Usage | None:
    """
    The usage record for this request, if accounting has started.
    """
    return _usage.get()


def _charge(stage: str, calls: int, prompt_tokens: int, completion_tokens: int) -> None:
    """
    Add to the current request's usage and enforce its budget.

    :raises BudgetExceededError: If the request has spent its allowance.
    """
    usage = _usage.get()
    if usage is None:
        return
    usage.record(stage, calls, prompt_tokens, completion_tokens)

    max_calls, max_tokens = _budget.get()
    if max_calls is not None and usage.calls > max_calls:
        raise BudgetExceededError(
            f"This request made {usage.calls} LLM calls, over its budget of {max_calls}. "
            "Global search costs one call per community; raise min_cluster_size or the "
            "budget."
        )
    if max_tokens is not None and usage.total_tokens > max_tokens:
        raise BudgetExceededError(
            f"This request spent about {usage.total_tokens} tokens, over its budget of "
            f"{max_tokens}."
        )


class CountingLLM(LLM):
    """
    LLM proxy that records what passes through it.

    Both entry points are overridden rather than only the abstract one: the base
    ``batch_chat_completion`` calls its *own* ``chat_completion``, so wrapping
    one would miss the other. ``desc`` is the stage — the engines already label
    their calls with it.
    """

    def __init__(self, llm: LLM, encoder: Any = None):
        """
        :param llm: The real LLM.
        :param encoder: Tokenizer with ``encode``; token counts are skipped
            without one.
        """
        self.llm = llm
        self.encoder = encoder

    def __getattr__(self, name: str) -> Any:
        # Anything the engines reach for that is not counted here.
        return getattr(self.llm, name)

    def _count(self, text: str) -> int:
        if not text or self.encoder is None:
            return 0
        try:
            return len(self.encoder.encode(text))
        except Exception:
            return 0

    def _conversation_tokens(self, conversation: Any) -> int:
        if not isinstance(conversation, (list, tuple)):
            return 0
        total = 0
        for message in conversation:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                total += self._count(content)
        return total

    def _answer_tokens(self, answer: Any) -> int:
        if answer is None:
            return 0
        if isinstance(answer, str):
            return self._count(answer)
        dump = getattr(answer, "model_dump_json", None)
        return self._count(dump()) if callable(dump) else 0

    @override
    async def chat_completion(self, conversation: Any, *args: Any, **kwargs: Any) -> Any:
        stage = kwargs.get("desc") or UNKNOWN_STAGE
        answer = await self.llm.chat_completion(conversation, *args, **kwargs)
        _charge(
            stage,
            1,
            self._conversation_tokens(conversation),
            self._answer_tokens(answer),
        )
        return answer

    async def batch_chat_completion(
        self, conversations: list[Any], *args: Any, **kwargs: Any
    ) -> Any:
        stage = kwargs.get("desc") or UNKNOWN_STAGE
        answers = await self.llm.batch_chat_completion(conversations, *args, **kwargs)
        prompt = sum(
            self._conversation_tokens(conversation) for conversation in conversations
        )
        completion = sum(self._answer_tokens(answer) for answer in answers or [])
        _charge(stage, len(conversations), prompt, completion)
        return answers

    async def stream_chat_completion(
        self, conversation: Any, *args: Any, **kwargs: Any
    ) -> Any:
        stage = kwargs.get("desc") or UNKNOWN_STAGE
        # Counted as one call; the deltas are summed as they pass.
        completion = 0
        async for delta in self.llm.stream_chat_completion(
            conversation, *args, **kwargs
        ):
            completion += self._count(delta if isinstance(delta, str) else "")
            yield delta
        _charge(stage, 1, self._conversation_tokens(conversation), completion)
