# `ragu.api` — the RAGU search service

HTTP service in front of a prebuilt knowledge graph. It owns the graph; clients
only search it.

**The API contract — routes, request and response shapes, configuration
variables, error codes, design decisions and current limits — lives in
[`docs/en/api.md`](../../docs/en/api.md) / [`docs/ru/api.md`](../../docs/ru/api.md).**
This file covers what is specific to running and extending the package:
deployment, and how the code is put together.

```bash
python -m ragu.api --port 8020 --backend stub   # no graph, no LLM
pytest tests/api                                       # needs the `api` extra
```

## Docker

```bash
docker build -t ragu-api .                          # from the repository root
docker compose up -d ragu-api
docker compose --profile qdrant up -d               # add a remote Qdrant
```

The build context is whitelisted in `.dockerignore` — the repository root holds
multi-gigabyte working directories that must stay out of it.

- Dependencies come from `uv.lock` in one step (`uv sync --frozen --no-dev
  --no-editable --extra api`), not from a fresh resolve: `openai>=2.32.0`
  resolves to 3.x, where `openai._utils._logs.httpx_logger` no longer exists and
  `ragu.common.logger` fails on import. The lock pins the version that works.
- `--build-arg RAGU_SYNC_EXTRAS="--extra local"` adds sentence-transformers and
  transformers for local rerankers and embedders; the default image stays around
  1 GB of site-packages.
- The tiktoken vocabularies are baked at build time; otherwise the first request
  downloads them and an offline deployment fails outright.
- The healthcheck polls `/health/ready`, which answers 503 until the graph is
  loaded, so `depends_on: service_healthy` waits for a service that can actually
  answer searches. Large graphs take minutes to load (a 200 MB GML plus 170k
  relation vectors is a few minutes on a warm page cache), hence the 10-minute
  start period.
- Set `RAGU_API_EMBEDDER_DIM` to the dimension the graph was built with. Without
  it the service sends one probe embedding at startup to detect the dimension,
  and startup fails if the embedder endpoint is unreachable. The dimension is
  stored in the graph: `head -c 40 <graph>/vdb_entity.json` shows
  `{"embedding_dim": N, ...`.
- The graph directory is mounted read-write at `/data/graph` and must be
  writable by uid 1000 (the `ragu` user in the image): the storages open missing
  files for writing at construction, so a read-only or root-owned mount fails
  the load with `PermissionError` rather than serving the graph.
- With the Docker **snap**, the daemon has a private `/tmp` and no access to it
  from the host, so a graph path under `/tmp` silently mounts as an empty
  directory. Keep graphs under `$HOME` (or use a named volume).
- The LLM cache lives in a named volume.

## Package layout

```
app.py       application factory: lifespan, exception handlers, error envelope
routes.py    per mode: search, /retrieve, /batch, /stream; plus /health,
             /health/live, /health/ready
models.py    request/response schemas; the request models embed the engines'
             own parameter dataclasses
mapping.py   engine results → wire schema, dispatched on the concrete result types
errors.py    RaguServiceError subclasses; every one renders through ErrorResponse
config.py    ServiceSettings and GraphSpec — the RAGU_API_* environment contract
client.py    typed httpx client; returns the models above, raises RaguApiError
registry.py  GraphRegistry: one backend per configured graph, built one at a time
jobs.py      JobManager over a JobStore interface, for work too long for a request
usage.py     CountingLLM: per-request LLM calls and tokens, by stage
reranking.py ForgivingScorer: a reranker outage costs ranking, not the answer
metrics.py   Prometheus text exposition, no dependency
middleware.py  request id, auth, body limit, metrics, admission
auth.py      API-key check
request_context.py  the request id, kept apart so errors and middleware do not
             import each other
logging_setup.py  intercepts stdlib logging (uvicorn, httpx) into loguru, so the
             process has one sink; only `python -m ragu.api` installs it
backends/
  base.py        SearchBackend ABC (search / retrieve / stream over a
                 SearchCall), GraphStats, the shared request bounds and the
                 capability rules both backends obey
  stub.py        canned answers; simulates a graph missing a capability by
                 reporting zero for that store, so it exercises the real path
  ragu_backend.py loads the graph, keeps one engine per mode, wraps in
                 QueryPlanEngine when use_query_plan
```

Three couplings worth knowing before changing anything here:

- **`Settings` is a process-global singleton, and the registry works around
  it.** Every per-graph value is read inside the constructors that run while
  that graph is built, so `GraphRegistry` builds graphs one at a time with
  `isolated_settings()` and rolls the singleton back after each. Engines built
  lazily afterwards would pick up whatever the singleton holds by then, so
  `RaguBackend` captures the context limit and tokenizer at load time and passes
  them explicitly.
- **`Index.__init__` calls `Settings.init_storage_folder()`**, which *creates*
  the folder when missing. `RaguBackend._require_storage_folder` refuses a
  missing or empty folder before that happens; do not bypass it, or a typo in
  `RAGU_API_STORAGE_FOLDER` will produce an empty directory and a service that
  looks healthy.
- **Request schemas embed the engines' parameter dataclasses.** An engine
  parameter added upstream reaches the wire with no change here; one renamed
  upstream breaks the wire contract.

## Adding a search mode

1. Add the mode to `SearchMode` (`models.py`) and a request model beside the
   existing ones.
2. Add its `ModeRequirement` to `MODE_REQUIREMENTS` (`backends/base.py`) — the
   capabilities it `requires` and both messages. `GraphStats.missing_for` reads
   them; nothing else needs teaching.
3. Register the engine's result type in `mapping.py` with
   `@_sources_from_result.register`. Without it the mode still works, but its
   retrieval degrades to a single `to_text()` source.
4. Add the request models and the four routes. The backend needs no new
   method: `search`, `retrieve` and `stream` all take a `SearchCall` carrying
   the mode.
