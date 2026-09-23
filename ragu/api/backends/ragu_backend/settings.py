"""
Holding the process-wide ``Settings`` singleton for one graph at a time.

There is one ``Settings`` per process, but every graph has its own storage
folder and language, and the library's constructors read both off it. These
apply one graph's values for the length of a block and put the previous ones
back afterwards.
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import get_type_hints

from ragu import Settings

# The public, annotated fields of the settings singleton. Snapshotting them by
# name keeps the isolation below out of GlobalSettings' internals.
_SETTINGS_FIELDS = tuple(
    name for name in get_type_hints(type(Settings)) if not name.startswith("_")
)


@contextmanager
def isolated_settings() -> Iterator[None]:
    """
    Apply changes to the ``Settings`` singleton and roll them back afterwards.

    ``Settings`` is process-global, so one graph's storage folder, language and
    token limits would otherwise leak into the next graph constructed. Every
    per-graph value is read inside the constructors that run in this block —
    ``Index`` reads the storage folder, the embedder reads its token limit — so
    restoring afterwards is enough.
    """
    snapshot = {name: getattr(Settings, name) for name in _SETTINGS_FIELDS}
    storage_folder = Settings.storage_folder
    try:
        yield
    finally:
        for name, value in snapshot.items():
            setattr(Settings, name, value)
        Settings.storage_folder = storage_folder


# Snapshotting is only isolation if the blocks do not overlap, and there is one
# singleton for the whole process. The registry serializes startup, but a build
# and a reindex are background jobs against whatever graph a client names: two
# of those on different graphs would each snapshot the other's folder, build
# into it, and leave the singleton restored to the wrong one.
_settings_lock = asyncio.Lock()


@asynccontextmanager
async def exclusive_settings() -> AsyncIterator[None]:
    """
    Hold the ``Settings`` singleton for one graph, then roll it back.

    The async counterpart of :func:`isolated_settings`, and the one every path
    that can run concurrently with another graph's must use. Blocks take minutes
    to hours, so a waiting job waits that long — which is the point: they cannot
    safely run at the same time.
    """
    async with _settings_lock:
        with isolated_settings():
            yield
