# arc-model-lab

Audience: backend engineers running or extending the service. Reading time: 5 minutes.

A minimal, production-shaped service that loads a HuggingFace model, exposes a
summarization endpoint, and records every inference in Postgres.

It is intentionally small: two domain entities, one endpoint, clean module
boundaries, and no speculative abstraction.

## Architecture

```
src/arc_model_lab/
├── api/          FastAPI routing layer (schemas, dependencies, routes)
├── domain/       Pure data models (Model, Inference) — no framework imports
├── services/     Business logic (model loading, inference workflow)
├── db/           SQLAlchemy ORM models + repositories
├── config.py     Environment-driven settings
└── main.py       Composition root (lifespan wiring + ASGI app)
```

Dependencies flow inward: `api → services → db → domain`. The domain layer
depends on nothing but the standard library and Pydantic.

### Request flow (`POST /summarize`)

1. Accept `input_text`.
2. Build chat messages (system + user) for the summarization task.
3. Render the chat template and generate via the (pre-loaded) `ModelService`.
4. Persist an `Inference` row (input, rendered prompt, output, tokens, latency).
5. Return the stored record.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Docker + Docker Compose

## Quickstart (local)

Postgres runs in Docker; the app runs on your host, so model weights cache
locally (clear them any time with `make model.clear-cache`).

```bash
# 1. Configuration
cp .env.example .env

# 2. Start Postgres
docker compose up -d postgres

# 3. Install dependencies (creates .venv)
uv sync

# 4. Apply database migrations
make migrate            # or: uv run alembic upgrade head

# 5. Seed the model catalog (required; without it /summarize returns 404)
make model.seed

# 6. Run the service (model weights download lazily on first request)
make run                # or: uv run arc-model-lab
```

Verify the service is up:

```bash
curl -s http://localhost:8000/health     # -> {"status":"ok"}
```

The API is available at `http://localhost:8000`; interactive docs at
[`/docs`](http://localhost:8000/docs).

## Run with Docker

One command builds the image, waits for Postgres, applies migrations, seeds the
catalog, and serves the API:

```bash
docker compose up --build    # requires a committed uv.lock
```

Model weights download once into the `hf_cache` volume, never into the image.

## Usage

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/summarize` | Summarize `input_text`; persists and returns one inference |
| `GET` | `/health` | Liveness probe |
| `GET` | `/docs` | Interactive OpenAPI docs |

Summarize a document:

```bash
curl -s http://localhost:8000/summarize \
  -H 'content-type: application/json' \
  -d '{"input_text": "Large language models can summarize long documents into concise overviews."}'
```

The response is the persisted inference row:

```json
{
  "id": "8f0c1e2a-...",
  "model_id": "3b1a9d4f-...",
  "input_text": "Large language models can ...",
  "prompt": "<rendered chat prompt>",
  "output_text": "A concise summary.",
  "latency_ms": 812,
  "prompt_tokens": 42,
  "completion_tokens": 18,
  "created_at": "2026-06-30T12:00:00Z"
}
```

To target a specific catalog model, pass its registered `name` (not the
HuggingFace id). Omit `model_name` to use the configured default.

```bash
curl -s http://localhost:8000/summarize \
  -H 'content-type: application/json' \
  -d '{"input_text": "...", "model_name": "qwen2.5-1.5b-instruct"}'
```

Inspect and manage the catalog from the CLI:

```bash
make model.list                                 # list registered models
make model.get NAME=qwen2.5-1.5b-instruct       # show one model
make model.smoke NAME=qwen2.5-1.5b-instruct     # load + run one summary
make model.activate NAME=gemma-3-1b-it          # allow /summarize to use it
make model.deactivate NAME=gemma-3-1b-it        # block it (returns 409)
```

## Evaluation

After an inference is stored, `arc-model-lab` can send it to the `arc-eval`
service for quality scoring. Each metric score (faithfulness, answer relevance,
and so on) is persisted in `evaluation_results`, linked to the inference id.
Evaluation is a separate service boundary, so inference storage never depends on
it.

Point the service at a running `arc-eval` and set a request timeout:

```bash
ARC_EVAL_SERVICE_URL=http://localhost:8001   # empty disables evaluation
ARC_EVAL_TIMEOUT_SECONDS=30
```

Request evaluation inline by adding `"evaluate": true`:

```bash
curl -s http://localhost:8000/summarize \
  -H 'content-type: application/json' \
  -d '{"input_text": "A long article.", "evaluate": true}'
```

The response carries the scores next to the summary:

```json
{
  "id": "8f0c1e2a-...",
  "output_text": "A concise summary.",
  "evaluation": {
    "status": "completed",
    "results": [
      { "metric_name": "faithfulness", "score": 0.91, "evaluator_name": "summary-faithfulness", "evaluator_version": "v1" }
    ]
  }
}
```

`status` is `completed` when `arc-eval` scored the summary, `failed` when it was
unreachable or returned an unusable response (online requests fail open: the
summary is still returned and stored), or `skipped` when no
`ARC_EVAL_SERVICE_URL` is configured.

Evaluate inferences that predate this feature, or whose evaluation failed, from
the CLI. These commands upsert on the metric key, so they are safe to re-run:

```bash
make eval.run ID=<inference-uuid>                            # evaluate one inference
make eval.replay LIMIT=100                                   # evaluate unevaluated rows
make eval.backfill SINCE=2026-01-01 UNTIL=2026-02-01 LIMIT=500
make eval.contract                                           # consumer contract tests (mocked)
make eval.smoke                                              # live end-to-end (needs ARC_EVAL_SERVICE_URL)
```

## Common tasks

Run `make help` for the full list. Most-used targets:

| Command | Purpose |
| --- | --- |
| `make run` | Run the API locally with auto-reload |
| `make migrate` | Apply migrations to head |
| `make model.seed` | Seed the catalog from `seeds/models.local.json` |
| `make model.list` | List registered models |
| `make eval.replay` | Evaluate unevaluated inferences via arc-eval |
| `make eval.backfill` | Evaluate unevaluated inferences in a time range |
| `make test` | Run tests with coverage |
| `make lint` | Ruff format check, Ruff lint, mypy |

## Configuration

All settings are environment variables with the `ARC_` prefix (see
[.env.example](.env.example)).

| Variable | Default | Purpose |
| --- | --- | --- |
| `ARC_DATABASE_URL` | `postgresql+psycopg://arc:arc@localhost:5432/arc_model_lab` | Postgres connection URL |
| `ARC_MODEL_ID` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace causal instruct model to load |
| `ARC_TOKENIZER_ID` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace tokenizer to load |
| `ARC_MODEL_NAME` | `qwen2.5-1.5b-instruct` | Registered name (unique key) |
| `ARC_MAX_INPUT_TOKENS` | `1024` | Input truncation length |
| `ARC_MAX_NEW_TOKENS` | `256` | Generated token budget |
| `ARC_NUM_BEAMS` | `1` | Decoding: 1 = greedy, >1 = beam search |
| `ARC_API_PORT` | `8000` | HTTP server port |
| `ARC_EVAL_SERVICE_URL` | (empty) | arc-eval base URL; empty disables evaluation |
| `ARC_EVAL_TIMEOUT_SECONDS` | `30` | arc-eval request timeout (seconds) |

## Development

Linting and type-checking tools live in the `lint` dependency group:

```bash
uv sync --group lint       # install ruff + mypy
uv run ruff check .
uv run ruff format
uv run mypy src
```

## Design notes

- **Causal instruct model.** The default is Qwen2.5-1.5B-Instruct, loaded via
  `AutoModelForCausalLM`. Requests are expressed as chat messages rendered
  through the tokenizer's chat template; `ModelService.generate()` is a
  task-agnostic primitive, so a new task only adds a new message builder (no
  model changes).
- **Sync end to end.** Model inference is CPU/GPU-bound and blocking; sync
  route handlers run in FastAPI's threadpool, so the event loop is never
  blocked. No async database or task queue is introduced.
- **Repositories return domain objects.** ORM types never escape `db/`, so the
  rest of the app is free of SQLAlchemy session/detach concerns.
- **Schema management.** The schema is owned by Alembic migrations in
  `migrations/`; apply them with `make migrate` before starting the app. A
  metadata naming convention keeps constraint names stable across autogenerate.
