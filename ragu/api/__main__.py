"""Entry point: ``python -m ragu.api``."""

import argparse
import pathlib

import uvicorn

from ragu.api.app import create_app, openapi_document
from ragu.api.config import ServiceSettings
from ragu.api.runtime.logging_setup import configure_logging
from ragu.api.search.reranking import reranker_from_env


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGU search service")
    parser.add_argument(
        "--host", default=None, help="Bind address (default from RAGU_API_HOST)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Bind port (default from RAGU_API_PORT)"
    )
    parser.add_argument(
        "--backend",
        choices=["ragu", "stub"],
        default=None,
        help="Search backend (default from RAGU_API_BACKEND)",
    )
    parser.add_argument(
        "--storage-folder",
        default=None,
        help="RAGU storage folder with the built graph (default from RAGU_API_STORAGE_FOLDER)",
    )
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument(
        "--dump-openapi",
        metavar="PATH",
        default=None,
        help="Write the OpenAPI schema to PATH and exit, instead of serving",
    )
    args = parser.parse_args()

    if args.dump_openapi:
        pathlib.Path(args.dump_openapi).write_text(
            openapi_document(), encoding="utf-8", newline="\n"
        )
        return

    configure_logging(args.log_level)

    overrides = {
        key: value
        for key, value in (
            ("host", args.host),
            ("port", args.port),
            ("backend", args.backend),
            ("storage_folder", args.storage_folder),
        )
        if value is not None
    }
    settings = ServiceSettings(**overrides)
    # The stub never reranks, so it does not need to know about a reranker.
    reranker = reranker_from_env() if settings.backend == "ragu" else None

    uvicorn.run(
        create_app(settings, reranker=reranker),
        host=settings.host,
        port=settings.port,
        log_level=args.log_level,
        # configure_logging already routed stdlib logging into loguru; uvicorn's
        # own dictConfig would install a second set of handlers on top.
        log_config=None,
    )


if __name__ == "__main__":
    main()
