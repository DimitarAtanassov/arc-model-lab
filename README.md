# arc-model-lab

Audience: backend engineers running or extending the service. Reading time: 5 minutes.

A small, production-shaped service that loads a HuggingFace model, runs inference
through `POST /inference`, and records every inference in Postgres. Scoring and
experimentation live in the separate arc-eval-service; the two services do not
call each other directly. arc-platform orchestrates them: it runs an inference
here, then hands that inference's input and output to arc-eval-service to score.

It stays intentionally small: a compact domain, a couple of inference endpoints,
clean module boundaries, and no speculative abstraction.

## Architecture

```
src/arc_model_lab/
├── api/          FastAPI routing layer (schemas, dependencies, routes)
├── domain/       Pure data models (Model, Inference), no framework imports
├── services/     Business logic (model loading, inference workflow)
├── db/           SQLAlchemy ORM models + repositories
├── config.py     Environment-driven settings
└── main.py       Composition root (lifespan wiring + ASGI app)
```

Dependencies flow inward: `api → services → db → domain`. The domain layer depends
on nothing but the standard library and Pydantic.

### Request flow (`POST /inference`)

1. Accept `model_name`, `input_text`, and an optional `temperature`.
2. Send `input_text` to the model as a single user turn.
3. Render the tokenizer chat template and generate via the (pre-loaded) `ModelService`.
4. Persist an `Inference` row (input, rendered prompt, output, tokens, latency).
5. Return the stored record. `/inference` never evaluates; scoring lives in the
   separate arc-eval-service.

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

# 5. Seed the model catalog (required; without it /inference returns 404)
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
| `POST` | `/inference` | Run the model named by `model_name` on `input_text`; persists and returns one inference |
| `GET` | `/inference` | List recent inferences, newest first |
| `GET` | `/inference/{id}` | Return one inference by id |
| `GET` | `/models` | List the catalog models |
| `GET` | `/models/{name}` | Return one catalog model by name |
| `GET` | `/health` | Liveness probe |
| `GET` | `/docs` | Interactive OpenAPI docs |

New here? See the interactive API docs at [`/docs`](http://localhost:8000/docs).

Summarize a document (the caller names the model and the decoding config):

```bash
curl -s http://localhost:8000/inference \
  -H 'content-type: application/json' \
  -d '{"model_name": "qwen2.5-1.5b-instruct", "input_text": "Large language models can summarize long documents into concise overviews.", "temperature": 0.0}'
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

`/inference` is pure inference: it never scores its output, so the response
carries no scores. The caller
names the model by `model_name`; an unknown name returns `404`. `temperature` is
optional: pass it (0 is greedy and deterministic, higher samples) to override, or
omit it to use the server default (`ARC_TEMPERATURE`). Output length is not a
caller knob; it always uses the server default (`ARC_MAX_OUTPUT_TOKENS`).

Models are registered by seeding, and each carries a `status` (`active`,
`inactive`, or `deprecated`); `/inference` serves only `active` models. To change a
model's status, edit its entry in `seeds/models.local.json` and re-run
`make model.seed` (an idempotent upsert keyed by name).

## Common tasks

Run `make help` for the full list. Most-used targets:

| Command | Purpose |
| --- | --- |
| `make run` | Run the API locally with auto-reload |
| `make migrate` | Apply migrations to head |
| `make model.seed` | Seed the catalog from `seeds/models.local.json` |
| `make test` | Run tests with coverage |
| `make lint` | Ruff format check, Ruff lint, mypy |

## Configuration

All settings are environment variables with the `ARC_` prefix (see
[.env.example](.env.example)).

| Variable | Default | Purpose |
| --- | --- | --- |
| `ARC_DATABASE_URL` | `postgresql+psycopg://arc:arc@localhost:5432/arc_model_lab` | Postgres connection URL |
| `ARC_DB_ECHO` | `false` | Echo SQL to logs (debugging only) |
| `ARC_MODEL_NAME` | `qwen2.5-1.5b-instruct` | Default catalog model resolved by name |
| `ARC_MODEL_CACHE_DIR` | (unset) | HuggingFace weight cache dir; unset uses the HF default |
| `ARC_DEVICE` | `cpu` | Compute device: `auto`, `cpu`, `mps`, `cuda` |
| `ARC_MAX_INPUT_TOKENS` | `1024` | Input truncation length |
| `ARC_MAX_OUTPUT_TOKENS` | `256` | Default generated token budget |
| `ARC_TEMPERATURE` | `0.0` | Default sampling temperature: 0 = greedy, up to 2.0 |
| `ARC_API_HOST` | `0.0.0.0` | HTTP bind address |
| `ARC_API_PORT` | `8000` | HTTP server port |

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
