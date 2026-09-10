import logging
import sys

import openai

openai._utils._logs.logger.setLevel(logging.WARNING)
openai._utils._logs.httpx_logger.setLevel(logging.WARNING)

from loguru import logger

LOG_FORMAT = (
    "<cyan>{time:HH:mm:ss}</cyan> | <level>{level: <8}</level> | <level>{message}</level>"
)

DEFAULT_LEVEL = "INFO"

# loguru prints the value of every local variable in a traceback when this is
# on. That is a good default for a library being debugged interactively, and a
# poor one for a long-running service, whose logs would then carry API keys and
# request bodies. Services turn it off through :func:`set_level`.
DEFAULT_DIAGNOSE = True

logger.remove()
_handler_id = logger.add(
    sys.stdout,
    colorize=True,
    enqueue=True,
    level=DEFAULT_LEVEL,
    format=LOG_FORMAT,
    diagnose=DEFAULT_DIAGNOSE,
)


def set_level(level: str, diagnose: bool = DEFAULT_DIAGNOSE) -> None:
    """
    Replace the default stdout sink with one at a different level.

    A loguru sink cannot be re-levelled in place, so the existing one is
    removed and re-added with the same destination and format.

    :param level: Level name, e.g. ``"DEBUG"`` or ``"warning"`` (case-insensitive).
    :param diagnose: Whether tracebacks carry the value of every local variable.
    :raises ValueError: If the level name is not known to loguru.
    """
    global _handler_id

    level = level.upper()
    logger.level(level)  # Raises ValueError on an unknown name.

    try:
        logger.remove(_handler_id)
    except ValueError:
        # A bare logger.remove() elsewhere already dropped every sink, so the
        # tracked id is stale. Adding the new sink is still the right outcome.
        pass
    _handler_id = logger.add(
        sys.stdout,
        colorize=True,
        enqueue=True,
        level=level,
        format=LOG_FORMAT,
        diagnose=diagnose,
    )


__all__ = ["logger", "set_level", "LOG_FORMAT", "DEFAULT_LEVEL", "DEFAULT_DIAGNOSE"]
