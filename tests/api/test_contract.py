"""
The contract as a whole: the committed schema, and what importing it costs.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip(
    "fastapi", reason="install the 'api' extra to test the search service"
)


class TestCommittedSchema:
    """
    The OpenAPI schema lives in the repository, and cannot go stale.

    Generated from the routes and the models, so it never disagrees with the
    code — but while it exists only in a running process, a change to the
    contract is invisible in review and a consumer has to boot the service to
    generate a client. Committing it fixes both; this test is what keeps the
    committed copy honest.
    """

    @staticmethod
    def _committed() -> tuple[pathlib.Path, str]:
        import ragu

        path = pathlib.Path(ragu.__file__).resolve().parent.parent / "docs" / "openapi.json"
        return path, path.read_text(encoding="utf-8")

    def test_the_committed_schema_matches_the_code(self):
        from ragu.api.app import openapi_document

        path, committed = self._committed()
        current = openapi_document()

        assert committed == current, (
            f"{path} is out of date with the routes and models.\n"
            "Regenerate it:\n"
            "    python -m ragu.api --dump-openapi docs/openapi.json"
        )

    def test_the_schema_is_deterministic(self):
        # Two runs must agree, or every commit carries a spurious diff.
        from ragu.api.app import openapi_document

        assert openapi_document() == openapi_document()

    def test_the_schema_is_not_shaped_by_the_environment(self, monkeypatch):
        # Built against stub settings on purpose: a developer's own RAGU_API_*
        # must not change what lands in the file.
        from ragu.api.app import openapi_document

        before = openapi_document()
        monkeypatch.setenv("RAGU_API_BACKEND", "ragu")
        monkeypatch.setenv("RAGU_API_MAX_TOP_K", "7")
        monkeypatch.setenv("RAGU_API_LANGUAGE", "english")
        assert openapi_document() == before

    def test_it_describes_every_operation_the_app_serves(self):
        import json

        from ragu.api.app import openapi_document

        schema = json.loads(openapi_document())
        operations = [
            (path, method)
            for path, item in schema["paths"].items()
            for method in item
            if method in ("get", "post", "put", "delete", "patch")
        ]
        assert len(operations) == 55
        # The routes a consumer is most likely to generate a client for.
        assert ("/v1/search/local", "post") in operations
        assert ("/v1/graphs/{graph_id}/relations/select", "post") in operations
        assert schema["info"]["version"]


class TestLightweightImport:
    """
    Importing the client does not import the library behind the service.

    It used to take three seconds and three thousand modules, fastembed,
    scikit-learn, pandas and nltk among them, because every ``ragu.*`` import
    ran a package init that loaded the whole library.
    """

    def test_the_client_imports_without_the_engines_or_the_server(self):
        import subprocess
        import sys

        probe = (
            "import sys; import ragu.api.client; "
            "heavy = [m for m in ('fastembed', 'sklearn', 'pandas', 'nltk', "
            "'networkx', 'openai', 'fastapi', 'starlette', 'uvicorn') "
            "if m in sys.modules]; print(','.join(heavy))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == ""

    @pytest.mark.parametrize("package", ["ragu", "ragu.search_engine", "ragu.api"])
    def test_the_package_still_exposes_every_public_name(self, package):
        import importlib

        module = importlib.import_module(package)
        # ``__all__`` is written out so linters can read it; this keeps it from
        # drifting away from the table the lazy lookup actually serves.
        assert set(module.__all__) - {"__version__"} == set(module._EXPORTS)
        for name in module.__all__:
            assert getattr(module, name) is not None, name

    def test_an_unknown_name_is_still_an_attribute_error(self):
        import ragu

        with pytest.raises(AttributeError):
            ragu.DoesNotExist

    def test_engine_parameters_keep_their_old_import_path(self):
        from ragu.search_engine.local_search import LocalParams as old
        from ragu.search_engine.params import LocalParams as new

        assert old is new
