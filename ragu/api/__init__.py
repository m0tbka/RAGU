"""
RAGU search service.

The names below load on first use. A client that imports ``ragu.api.client``
or ``ragu.api.models`` goes through this package, and must not pay for the
server — FastAPI, the backends and the engines behind them — to get there.
"""

from typing import TYPE_CHECKING

_EXPORTS: dict[str, str] = {
    "create_app": "ragu.api.app",
    "ServiceSettings": "ragu.api.config",
}


def __getattr__(name: str):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'ragu.api' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


if TYPE_CHECKING:
    from ragu.api.app import create_app
    from ragu.api.config import ServiceSettings


__all__ = ["create_app", "ServiceSettings"]
