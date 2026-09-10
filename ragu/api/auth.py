"""
Who may call this service.

Every request costs LLM calls, so an open deployment is an open budget. The
check is a shared secret rather than an identity system: the service has no
users, and putting one behind it belongs to the gateway.
"""

import hmac

from fastapi import Request

from ragu.api.config import ServiceSettings
from ragu.api.errors import UnauthorizedError

BEARER_PREFIX = "bearer "


def presented_key(request: Request) -> str | None:
    """
    Read the key a client offered, from either accepted header.

    :param request: Incoming request.
    :return: The key, or ``None`` when none was offered.
    """
    header = request.headers.get("authorization")
    if header and header[: len(BEARER_PREFIX)].lower() == BEARER_PREFIX:
        return header[len(BEARER_PREFIX) :].strip() or None
    return request.headers.get("x-api-key")


def authorize(request: Request, settings: ServiceSettings) -> None:
    """
    Reject a request that carries no accepted key.

    Comparison is constant-time so a wrong key cannot be found one character at
    a time. With no keys configured the service is open, which is the right
    default for a local stub and the wrong one for anything else — the docs say
    so and the log says so at startup.

    :param request: Incoming request.
    :param settings: Service settings, holding the accepted keys.
    :raises UnauthorizedError: If the key is missing or not accepted.
    """
    accepted = settings.api_keys_set()
    if not accepted:
        return

    offered = presented_key(request)
    if offered is None:
        raise UnauthorizedError(
            "This service requires an API key. Send it as 'Authorization: Bearer "
            "<key>' or 'X-API-Key: <key>'."
        )
    if not any(hmac.compare_digest(offered, key) for key in accepted):
        raise UnauthorizedError("The API key presented is not accepted.")
