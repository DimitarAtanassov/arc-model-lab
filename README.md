# arc-model-lab

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
- Docker + Docker Compose (for Postgres)

## Quickstart

```bash
# 1. Configuration
cp .env.example .env

# 2. Start Postgres
docker compose up -d

# 3. Install dependencies (creates .venv)
uv sync

# 4. Run the service (loads the model on startup, then serves)
uv run arc-model-lab
```

The API is then available at `http://localhost:8000` (interactive docs at
`/docs`).

### Example

```bash
curl -s http://localhost:8000/summarize \
  -H 'content-type: application/json' \
  -d '{"input_text": "Large language models can summarize long documents into concise overviews."}'
```

Each call returns the persisted inference (including `id`, token counts, and
`latency_ms`) and writes one row to the `inferences` table.

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
  task-agnostic primitive, so a new task only adds a new message builder — no
  model changes.
- **Sync end to end.** Model inference is CPU/GPU-bound and blocking; sync
  route handlers run in FastAPI's threadpool, so the event loop is never
  blocked. No async database or task queue is introduced.
- **Repositories return domain objects.** ORM types never escape `db/`, so the
  rest of the app is free of SQLAlchemy session/detach concerns.
- **Schema management.** Tables are created on startup via
  `Base.metadata.create_all`. Alembic can be layered in later without touching
  the domain or service layers.
