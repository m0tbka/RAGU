"""
The identity of the request being handled.

Lives apart from the middleware that sets it because the error envelope reads
it, and the middleware raises the errors: a shared context variable is the
seam that keeps those two from importing each other.
"""

from contextvars import ContextVar

REQUEST_ID_HEADER = "X-Request-ID"

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def current_request_id() -> str | None:
    """
    The id of the request being handled, or ``None`` outside one.
    """
    return _request_id.get()


def set_request_id(request_id: str):
    """
    Bind an id to this request's context.

    :param request_id: The id to bind.
    :return: A token for :func:`reset_request_id`.
    """
    return _request_id.set(request_id)


def reset_request_id(token) -> None:
    """
    Unbind the id, restoring whatever was there before.
    """
    _request_id.reset(token)
