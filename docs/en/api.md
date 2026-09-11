# RAGU HTTP service

---
1. [What it is](#what-it-is)
2. [Running the service](#running-the-service)
3. [Configuration](#configuration)
4. [The graph catalogue](#the-graph-catalogue)
5. [The graph surface](#the-graph-surface)
6. [Ingestion](#ingestion)
7. [Search endpoints](#search-endpoints)
8. [Operations](#operations)
9. [Health and readiness](#health-and-readiness)
10. [Errors](#errors)
11. [Design decisions](#design-decisions)
12. [Logging](#logging)
13. [Current limits](#current-limits)

---

## What it is

`ragu.api` is a FastAPI service that puts a prebuilt knowledge graph behind
HTTP. It ships inside the package, so an installed RAGU can serve a graph
without a checkout, and clients in any language can query it instead of
importing RAGU as a Python library.

The service **serves** a graph; it never builds one. Build the graph first —
`KnowledgeGraph.build_from_docs`, or one of the scripts in `examples/` — then
point the service at that storage folder.

It owns one graph per process and exposes one route per search mode, so a
gateway can apply a different timeout and rate limit to each. That matters:
global search rates every community summary against the query with its own LLM
call (N+1 calls for N communities), while local and naive issue one generation
each.

Install it with the `api` extra:

```bash
pip install -e ".[api]"
```

## Running the service

```bash
# Canned answers, no graph and no LLM required — for developing clients
python -m ragu.api --backend stub

# Real graph
python -m ragu.api --backend ragu --storage-folder ragu_working_dir
```

`--host`, `--port`, `--backend` and `--storage-folder` override the
corresponding environment variables on the command line. Interactive API docs
are at `/docs`.

With Docker, `docker compose up -d ragu-api` builds the image from the
repository root and mounts a prebuilt graph at `/data/graph`. The deployment
details that bite — the whitelist `.dockerignore`, baked tiktoken vocabularies,
the healthcheck start period, and write access for uid 1000 — are documented in
[`ragu/api/README.md`](../../ragu/api/README.md).

## Configuration

Service settings are `RAGU_API_*` environment variables, read by
`ServiceSettings` (`ragu/api/config.py`), optionally from a `.env` file.

| Variable | Default | Meaning |
|---|---|---|
| `RAGU_API_GRAPHS` | — | Graphs to serve, as a JSON list. Unset means one graph from the flat variables |
| `RAGU_API_MAX_LLM_CALLS_PER_REQUEST` | — | LLM calls one request may make |
| `RAGU_API_MAX_TOKENS_PER_REQUEST` | — | Approximate tokens one request may spend |
| `RAGU_API_MAX_CONCURRENT_GENERATIONS` | — | Generations that may run at once |
| `RAGU_API_API_KEYS` | — | Comma-separated API keys; empty leaves the service open |
| `RAGU_API_CORS_ORIGINS` | — | Comma-separated browser origins |
| `RAGU_API_MAX_BODY_BYTES` | `33554432` | Largest request body read |
| `RAGU_API_REQUEST_TIMEOUT` | `300` | Seconds one request may take |
| `RAGU_API_BACKEND` | `ragu` | `ragu` loads a real graph, `stub` serves canned answers |
| `RAGU_API_HOST` | `127.0.0.1` | Bind address; the container image passes `--host 0.0.0.0` itself |
| `RAGU_API_PORT` | `8020` | Bind port |
| `RAGU_API_STORAGE_FOLDER` | `ragu_working_dir` | Folder holding the built graph |
| `RAGU_API_LANGUAGE` | `russian` | Passed to `Settings.language` |
| `RAGU_API_SETTINGS_FILE` | — | `Settings` JSON saved at build time, loaded instead of the defaults |
| `RAGU_API_EMBEDDER_DIM` | — | Embedding dimension; auto-detected with a probe request when unset |
| `RAGU_API_RATE_MIN_DELAY` | — | Minimum delay between LLM calls, seconds |
| `RAGU_API_RATE_MAX_SIMULTANEOUS` | — | Maximum simultaneous LLM calls |
| `RAGU_API_LLM_CACHE` | — | Path to the LLM response cache; unset disables caching |
| `RAGU_API_RERANK_TIMEOUT` | `10` | Seconds to wait for the reranker before answering without it |
| `RAGU_API_ENGINE_CACHE_SIZE` | `32` | How many (mode, language) engines to keep built |
| `RAGU_API_MAX_BATCH_SIZE` | `50` | Maximum number of queries a /batch route accepts |
| `RAGU_API_MAX_TOP_K` | `100` | Ceiling applied to a client-supplied `top_k` / `rerank_top_k` |
| `RAGU_API_MIN_CLUSTER_SIZE_FLOOR` | `1` | Floor applied to global `min_cluster_size` |
| `RAGU_API_STUB_MISSING_CAPABILITIES` | — | Stub only: capabilities to report as missing. An unknown name fails startup rather than simulating nothing |

`RAGU_API_STORAGE_FOLDER` must point at one graph directory, not at a folder of
them, and the directory must already exist and be non-empty — the service
refuses to start otherwise rather than creating it.

LLM and embedder credentials come from the same `.env` the rest of RAGU uses
(`ragu.common.env.Env`): `LLM_MODEL_NAME`, `LLM_BASE_URL`, `LLM_API_KEY`, and
optionally `EMBEDDER_BASE_URL`, `EMBEDDER_API_KEY`, `EMBEDDER_MODEL_NAME`.

## The graph catalogue

One process serves several graphs. Each is named, and the name is a path
segment:

| Route | Returns |
|---|---|
| `GET /v1/graphs` | every graph with its sizes and available modes |
| `GET /v1/graphs/{id}` | one graph |
| `GET /v1/graphs/{id}/capabilities` | which modes it can serve, and why not the rest |

```json
{"default": "books",
 "graphs": [{"id": "books", "loaded": true, "language": "russian",
             "stats": {"entities": 128400, "relations": 170233,
                       "chunks": 9812, "community_summaries": 341},
             "modes": [{"mode": "local", "available": true,
                        "missing_capability": null, "reason": null},
                       {"mode": "global", "available": false,
                        "missing_capability": "community_summaries",
                        "reason": "This graph carries no community summaries..."}],
             "error": null}]}
```

`capabilities` exists so a client can grey out the modes a corpus cannot serve
instead of offering them and reading the 409.

Configure the catalogue with `RAGU_API_GRAPHS`, a JSON list:

```bash
RAGU_API_GRAPHS='[{"id":"books","storage_folder":"/data/books","language":"russian"},
                  {"id":"papers","storage_folder":"/data/papers","language":"english",
                   "embedder_dim":1024}]'
```

Each entry takes `id`, `storage_folder`, and optionally `language`,
`settings_file` and `embedder_dim`. With the variable unset, one graph named
`default` is built from the flat `RAGU_API_STORAGE_FOLDER` / `RAGU_API_LANGUAGE`
variables, so a single-graph deployment needs no catalogue.

A graph that fails to load is recorded and skipped: the others stay servable and
`/v1/graphs` reports why that one is missing.

### How several graphs fit in one process

`Settings` is a process-wide singleton, and `Index` reads
`Settings.storage_folder` in its constructor. It is read *only* there, and the
embedder and engines likewise read their token limits and tokenizer names at
construction — so the registry builds graphs one at a time, applies that graph's
settings around the construction, and rolls the singleton back afterwards.

Engines built later — one per language, on demand — would otherwise pick up
whatever the singleton holds by then, so the backend captures the token limit
and tokenizer at load time and passes them explicitly.

## The graph surface

Reads of the structure, for a client that draws the corpus or traces an answer
back to it.

| Route | |
|---|---|
| `GET /v1/graphs/{id}/stats` | sizes, `embedding_dim`, documents, mode availability |
| `GET /v1/graphs/{id}/entities` | `limit`, `offset`, `type`, `search` (name substring) |
| `GET /v1/graphs/{id}/relations` | `limit`, `offset`, `min_strength` |
| `GET /v1/graphs/{id}/entities/{eid}/neighbors` | `depth` (1–4), `limit` |
| `GET /v1/graphs/{id}/communities` | `limit`, `offset`, `level` |
| `GET /v1/graphs/{id}/communities/{cid}` | one community with its members |
| `GET /v1/graphs/{id}/chunks/{cid}` | one source chunk |
| `GET /v1/graphs/{id}/consistency` | the cross-storage audit |
| `POST /v1/graphs/{id}/reindex/{kind}` | `community`, `descriptions` or `graph`, as a job |
| `GET /v1/ontology` | the NEREL entity and relation types |

Neighbourhoods return no coordinates: a client laying the graph out knows its
own viewport and does the layout itself. A neighbourhood that would exceed
`limit` nodes comes back with `truncated: true` rather than growing without
bound.

`get_all_nodes` rebuilds an `Entity` per node on every call, so a client paging
through a large graph would pay O(n) per page. The entity and relation lists are
therefore materialized once and kept, and dropped whenever the graph is written
to — by ingestion or by a reindex.

Reindexing writes into the stores searches read, so it takes the same write lock
ingestion does: the graph answers `409 GRAPH_BUSY` while it runs.

## Ingestion

Building a graph takes minutes to hours, which no request can hold open, so
documents are submitted as a job and the job is polled.

| Route | |
|---|---|
| `POST /v1/graphs/{id}/documents` | `202` with a job; `Location` points at it |
| `GET /v1/jobs` | every job this process knows, newest first (`?graph_id=`) |
| `GET /v1/jobs/{job_id}` | one job |
| `DELETE /v1/jobs/{job_id}` | ask a running job to stop |

Send `Idempotency-Key` so a client retry does not build the same corpus twice.

**Searches on a graph answer `409 GRAPH_BUSY` while it is being built.**
`build_from_docs` writes into the same stores the search reads, and the
file-backed ones tolerate no concurrent access, so the write wins and the read
is told to come back.

Ingestion is off by default. A graph accepts documents only when its spec says
so, because serving a prebuilt graph and building one are different workloads
with different configuration:

```json
{"id": "corpus", "storage_folder": "/data/corpus",
 "build": {"enabled": true, "chunker": "simple", "chunk_size": 1200,
           "chunk_overlap": 100, "vector_only": false,
           "make_community_summary": true}}
```

`vector_only` builds chunk vectors and skips entity extraction — the only mode
that works without an extractor, and the one that leaves `local` and `global`
unavailable.

Jobs live in memory: one replica, and a restart loses what was running. The
store is an interface so a shared one can replace it; that is deliberately not
done yet.

## Search endpoints

Four modes — `global`, `local`, `naive`, `mix` — each in four shapes:

Every search route lives under a graph. `POST /v1/graphs/{id}/search/{mode}`
is canonical; the flat `POST /v1/search/{mode}` is kept for clients written
before the service served more than one graph and addresses the default graph.

| Route | Returns |
|---|---|
| `POST /v1/graphs/{id}/search/{mode}` | one answer with its sources |
| `POST /v1/graphs/{id}/search/{mode}/retrieve` | context only, nothing generated |
| `POST /v1/graphs/{id}/search/{mode}/batch` | one answer per query in `queries` |
| `POST /v1/graphs/{id}/search/{mode}/stream` | the answer as Server-Sent Events |

`mix` ensembles the local and naive engines. Its child parameters are named
separately — `local_params` and `naive_params` — because `MixSearchEngine` reads
them from its constructor: `batch_search` ignores its `params` argument
entirely, and `batch_query` reads only `ensemble_responses` from it.

| Mode | Body |
|---|---|
| `global` | `query`, `params: GlobalSearchParams` |
| `local` | `query`, `use_query_plan=true`, `params: LocalParams` |
| `naive` | `query`, `use_query_plan=true`, `params: NaiveSearchParams` |
| `mix` | `query`, `use_query_plan=true`, `params: MixQueryParams`, `local_params`, `naive_params` |

`params` is the engine's own parameter class, embedded in the request model
rather than restated field by field:

```json
{
  "query": "Who wrote the novel?",
  "use_query_plan": true,
  "params": {"top_k": 20, "rerank_top_k": null, "use_summary": true, "use_chunks": true}
}
```

Three consequences of that choice:

- **Defaults are the engine's.** Omitting `params` yields `LocalParams()` /
  `NaiveSearchParams()`, so `use_summary` is `false` and `top_k` is `20`.
  Clients that care should send the values.
- **Bounds are the service's.** The dataclasses carry none, so `top_k` and
  `rerank_top_k` are clamped to `RAGU_API_MAX_TOP_K` and `min_cluster_size` is
  raised to `RAGU_API_MIN_CLUSTER_SIZE_FLOOR`.
- **Unknown fields are rejected, inside `params` as well as outside it.** A
  top-level `top_k` (the shape before `params` existed) and a mistyped
  `params.topk` both answer `400` instead of quietly serving the engine default.

Global search takes no `use_query_plan`: its answer is synthesized from
summaries that already cover the whole corpus, so decomposing the question would
only re-rate the same summaries once per subquery. `used_query_plan` stays in
the response envelope and is always `false` for that mode.

Success response:

```json
{
  "query": "Who wrote the novel 'Quo Vadis'?",
  "mode": "local",
  "used_query_plan": true,
  "answer": "The novel was written by Henryk Sienkiewicz...",
  "sources": [{"id": "chunk-8f3...", "type": "chunk", "content": "...", "score": 0.87}],
  "subqueries": [{"query": "Who wrote the novel?", "answer": "Henryk Sienkiewicz"}]
}
```

`sources` is the retrieval flattened to `{id, type, content, score}`: chunks and
their scores from `NaiveSearchResult`, entities / relations / summaries / chunks
from `LocalSearchResult`, rated insights from `GlobalSearchResult`. A result
type the service does not model degrades to a single source rendered with
`to_text()`.

When a query plan ran, `sources` carries the evidence of **every** subquery,
deduplicated, not only that of the final one — the intermediate answers are
returned in `subqueries`, so their evidence is returned with them.


### Answer language

Every request that generates an answer takes an optional `language`; without it
the service default (`RAGU_API_LANGUAGE`) applies.

This is a per-request parameter and not a deployment setting because the engines
bake the language in at construction — each one reads it when it renders the
answer prompt — so a language fixed at startup means a Russian question against
an English corpus is answered in English. Engines are therefore cached per
`(mode, language)`; `RAGU_API_ENGINE_CACHE_SIZE` bounds that cache, since the
client chooses the language.

The value is interpolated into the prompt, so it is constrained to a plain
language name (`^[A-Za-z][A-Za-z \-]*$`, 2–32 characters) rather than accepted
as free text.

### Reranking

`rerank` (default `true`) asks for the deployment's reranker; `rerank_top_k`
inside `params` says how many results to keep after it.

The service never constructs a reranker — on a CPU-only deployment the model
runs in its own container — so one is passed to `create_app(reranker=...)`. With
none configured, `rerank` is a no-op.

A reranker that fails or exceeds `RAGU_API_RERANK_TIMEOUT` costs ranking
quality, not the answer: the retrieval order is kept and `engines.rerank_error`
says what happened, with `engines.reranked` false and `engines.degraded` true.
Without that a reranker outage would surface as a 500 for a request the engines
could still answer.

### Retrieval without generation

`POST /v1/search/{mode}/retrieve` returns the context and nothing else, for
clients that generate the answer themselves. It takes no `use_query_plan`:
`QueryPlanEngine.batch_search` delegates straight to the wrapped engine and does
no planning, so accepting the flag would promise a decomposition that never
happens. Sending it answers `400`.

### Batches

`POST /v1/search/{mode}/batch` takes `queries` and answers all of them in one
pass. This is where the engines earn their keep: retrieval is shared across the
list, and with a query plan the independent subqueries of *different* top-level
queries are answered in the same child batch.

A query that retrieved nothing carries `error` instead of an answer, so one
empty query does not fail the batch. `RAGU_API_MAX_BATCH_SIZE` bounds the list.

```json
{"mode": "naive", "used_query_plan": true, "engines": {"...": "..."},
 "results": [{"query": "...", "answer": "...", "sources": [], "subqueries": []}]}
```

### Streaming

`POST /v1/search/{mode}/stream` returns `text/event-stream`: one `meta` event
carrying the retrieval and the engine report, then `delta` events carrying the
text, then `done` with the final engine report. A failure after the headers are
sent arrives as an `error` event — the capability check runs *before* the
response starts, so an unservable mode still fails with a status code.

### What actually ran

Every response carries `engines`:

```json
{"requested": "mix", "used": "MixSearchEngine", "query_plan": true,
 "degraded": true,
 "children": [{"engine": "LocalSearchEngine", "mode": "local", "ok": false,
               "error": "RuntimeError: entity vector store unreachable"},
              {"engine": "NaiveSearchEngine", "mode": "naive", "ok": true,
               "error": null}]}
```

`MixSearchEngine` runs with `allow_partial_failures=True` and drops a child that
raises, so without this "graph and chunks" would be indistinguishable from
"chunks only". `degraded` is true whenever some child did not contribute.

## Operations

### Authentication

`RAGU_API_API_KEYS` is a comma-separated list. A request presents one as
`Authorization: Bearer <key>` or `X-API-Key: <key>`; comparison is constant-time.

With the variable empty the service is **open**, which suits a local stub and
nothing else — the log says so at startup. `/health*`, `/metrics` and the
OpenAPI documents stay open regardless: an orchestrator has no key, and a client
needs the schema to talk.

### Correlation

Every response carries `X-Request-ID`, echoing the client's if it sent one, and
the error envelope repeats it in `request_id`. A `500` says only "see the
service log"; without a shared id there is no way to find the line it refers to.

### Metrics

`GET /metrics` renders Prometheus text: `ragu_api_requests_total`,
`ragu_api_request_duration_seconds` (a histogram out to five minutes, because a
global search is N+1 LLM calls), `ragu_api_searches_total` by mode and by
whether the answer was degraded, and gauges for graphs and jobs.

Labels use the route template, so a graph id or a job id cannot open a time
series of its own.

### Cost and budgets

Every response carries `usage`: LLM calls and token counts, broken down by the
stage that spent them — the engines already label their calls (`QueryPlan
decompose`, `GlobalSearch batch meta-eval`, `NaiveSearch batch query`), and
those labels are the breakdown.

```json
{"estimated": true, "calls": 7, "prompt_tokens": 4210, "completion_tokens": 380,
 "total_tokens": 4590,
 "stages": {"QueryPlan decompose": {"calls": 1, "prompt_tokens": 210, "completion_tokens": 60},
            "NaiveSearch batch query": {"calls": 6, "prompt_tokens": 4000, "completion_tokens": 320}}}
```

`estimated` is always true and means it: the LLM clients return the parsed
answer, not the raw response, so provider `usage` never reaches this layer.
Counts are measured with the tokenizer — close enough to price a request and to
stop a runaway one, not close enough to bill from.

| Variable | Default | |
|---|---|---|
| `RAGU_API_MAX_LLM_CALLS_PER_REQUEST` | — | Over it, `429 BUDGET_EXCEEDED` |
| `RAGU_API_MAX_TOKENS_PER_REQUEST` | — | Over it, `429 BUDGET_EXCEEDED` |
| `RAGU_API_MAX_CONCURRENT_GENERATIONS` | — | Beyond it, `429 TOO_MANY_REQUESTS` |

Refusing at the door is deliberate. Without a ceiling every accepted request
fans straight out to the LLM and the provider's rate limit becomes the queue —
held open inside this process, one socket and one buffer per waiting request.

### Bounds

| Variable | Default | |
|---|---|---|
| `RAGU_API_MAX_BODY_BYTES` | 32 MiB | Larger bodies answer `413` |
| `RAGU_API_REQUEST_TIMEOUT` | `300` | Longer requests answer `504`; they hold an LLM budget open while they wait |
| `RAGU_API_CORS_ORIGINS` | — | Comma-separated; empty sends no CORS headers |

## Health and readiness

| Route | Status | Use |
|---|---|---|
| `GET /health` | always `200` | Human/debug view; body says whether it is ready |
| `GET /health/live` | always `200` | Liveness probe: the process is up |
| `GET /health/ready` | `200` / `503` + `Retry-After` | Readiness probe: searches can be served |

Loading a large graph takes minutes, so liveness and readiness are separate: an
orchestrator must not restart a service that is still starting up.

```json
{
  "status": "ok",
  "graph_loaded": true,
  "stats": {"entities": 128400, "relations": 170233, "chunks": 9812, "community_summaries": 341},
  "error": null
}
```

`stats` is measured once at startup and is what `graph_loaded` is derived from:
it means "the graph holds something a search mode can read", not "an object was
constructed". When startup failed, `error` carries the reason and the same
reason appears in the `503` body of every search — the diagnostics are not
buried in the log.

## Errors

All errors share one envelope:

```json
{"error": {"code": "CAPABILITY_UNAVAILABLE", "mode": "global",
           "missing_capability": "community_summaries", "message": "..."}}
```

| Status | Code | Situation |
|---|---|---|
| 400 | `INVALID_REQUEST` | empty `query`, bad field type, unknown field |
| 401 | `UNAUTHORIZED` | no accepted API key was presented |
| 429 | `BUDGET_EXCEEDED` | the request spent its allowance of calls or tokens |
| 429 | `TOO_MANY_REQUESTS` | the service is already at its generation ceiling |
| 413 | `PAYLOAD_TOO_LARGE` | the request body exceeds the limit |
| 504 | `REQUEST_TIMEOUT` | the request outlived RAGU_API_REQUEST_TIMEOUT |
| 404 | `NOT_FOUND` | no such entity, community or chunk in this graph |
| 404 | `GRAPH_NOT_FOUND` | the path names a graph that is not configured |
| 404 | `JOB_NOT_FOUND` | no job with that id in this process |
| 409 | `GRAPH_BUSY` | the graph is being built; carries `Retry-After` |
| 409 | `CAPABILITY_UNAVAILABLE` | the graph cannot serve this mode, or this query found nothing |
| 503 | `SERVICE_NOT_READY` | graph not loaded, or startup failed; carries `Retry-After` |
| 500 | `INTERNAL_ERROR` | LLM unavailable, embedder timeout, engine failure |

`409` exists so a client can react on its own: a graph built without community
summaries cannot answer global searches, and one built without entity extraction
cannot answer local ones. `missing_capability` tells the two `409` cases apart:

- **named** (`community_summaries`, `entity_graph`, `vector_index`) — the graph
  has nothing this mode reads. Detected from the startup measurement, so no
  generation call is spent on it.
- **`null`** — the graph does support the mode, but this particular query
  retrieved no evidence. The engines answer confidently from an empty context,
  so an answer built on nothing is reported rather than returned.

`500` responses never quote the underlying exception: engine and LLM-client
errors routinely include the endpoint URL and parts of the request. The detail
goes to the service log.

## Design decisions

**Why the request embeds the engine's parameter class.** `QueryPlanEngine`
forwards `params` untouched to the engine it wraps, so per-request options have
to travel as parameter objects rather than constructor arguments. Reusing those
same classes as the request schema means an engine parameter added upstream is
available over HTTP with no change here — and one renamed upstream changes the
wire contract, which is the price of not restating them.

**Why capabilities are measured, not inferred.** The engines answer from
whatever context they gathered, empty included. Counting the entity, chunk and
community-summary stores once at startup is what lets the service refuse an
unsupported mode before paying for a generation call, and what makes
`graph_loaded` truthful.

**Why conversion dispatches on result types.** The engines are part of this
package and versioned with it, so `ragu/api/mapping.py` dispatches on the
concrete `*SearchResult` classes instead of probing for attributes. A result
type that changes shape then fails loudly instead of producing an empty source
list that the service would report as a missing capability.

**Why the cost knobs are server-side.** One global search costs N+1 LLM calls
for N surviving communities. `RAGU_API_MIN_CLUSTER_SIZE_FLOOR` is the only cap
on that, just as `RAGU_API_MAX_TOP_K` caps retrieval width; both are applied in
the backend base class so every backend enforces them identically.

## Logging

`python -m ragu.api` calls `configure_logging` (`ragu/api/logging_setup.py`),
which installs an intercept handler on the stdlib root and re-emits every
record through loguru. Uvicorn, httpx and the service itself then share the one
sink and one format the engines already use, so a single request can be
followed end to end. `--log-level` sets that sink's level.

Importing `create_app` into another server (gunicorn, an existing uvicorn
setup) leaves that server's logging untouched — call `configure_logging`
yourself if you want the same behaviour.

## Current limits

Known gaps, listed so they are not mistaken for oversights:

- **The graph surface is read-only.** Entities and relations can be listed and
  walked, not created or edited over HTTP; ingestion and reindexing are the
  write paths.
