# arc-model-lab

Audience: backend engineers running or extending the service. Reading time: 5 minutes.

A small, production-shaped service that loads a HuggingFace model, runs inference
through a single endpoint (`POST /inference`), records every inference in
Postgres, groups reproducible runs as experiments, and can score each output
through the `arc-eval` service.

It stays intentionally small: a compact domain, an inference endpoint and a small
set of experiment endpoints, clean module boundaries, and no speculative
abstraction.

## Architecture

```
src/arc_model_lab/
├── api/          FastAPI routing layer (schemas, dependencies, routes)
├── domain/       Pure data models (Model, Inference, Experiment, EvaluationResult), no framework imports
├── services/     Business logic (model loading, inference workflow)
├── clients/      Outbound clients for external services (arc-eval)
├── db/           SQLAlchemy ORM models + repositories
├── config.py     Environment-driven settings
└── main.py       Composition root (lifespan wiring + ASGI app)
```

Dependencies flow inward: `api → services → db → domain`. Outbound integrations
live in `clients/`, called by `services/` and depending only on `domain`. The
domain layer depends on nothing but the standard library and Pydantic.

### Request flow (`POST /inference`)

1. Accept `model_name`, `input_text`, and an optional `temperature`.
2. Build chat messages (system + user) for the summarization task.
3. Render the chat template and generate via the (pre-loaded) `ModelService`.
4. Persist an `Inference` row (input, rendered prompt, output, tokens, latency).
5. Return the stored record. `/inference` never evaluates; scoring lives in
   experiments and the evaluation CLI.

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
| `POST` | `/inference` | Run the model named by `model_name` on `input_text`; persists and returns one inference (no evaluation) |
| `POST` | `/inference/{id}/evaluate` | Score an existing inference against named metrics (no experiment) |
| `POST` | `/experiments` | Create a named run configuration (model + decoding) |
| `POST` | `/experiments/{id}/run` | Run an experiment: infer, store, link, and (when `metrics` are named) evaluate |
| `GET` | `/experiments/{id}/results` | Aggregated scores for an experiment, per metric |
| `GET` | `/experiments/{id}/compare/{other_id}` | Compare two experiments' scores side by side |
| `GET` | `/health` | Liveness probe |
| `GET` | `/docs` | Interactive OpenAPI docs |

New here? The [inference, evaluation, and experiments guide](docs/usage.md) walks
through each with runnable, real-world examples. To validate the whole stack from
a clean checkout, follow the [end-to-end testing guide](docs/end-to-end-testing.md).

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

`/inference` is pure inference: it never evaluates and never runs under an
experiment, so the response carries no `experiment_id` and no scores. The caller
names the model by `model_name`; an unknown name returns `404`. `temperature` is
optional: pass it (0 is greedy and deterministic, higher samples) to override, or
omit it to use the server default (`ARC_TEMPERATURE`). Output length is not a
caller knob; it always uses the server default (`ARC_MAX_OUTPUT_TOKENS`).

Inspect and manage the catalog from the CLI:

```bash
make model.list                                 # list registered models
make model.get NAME=qwen2.5-1.5b-instruct       # show one model
make model.smoke NAME=qwen2.5-1.5b-instruct     # load + run one summary
make model.activate NAME=gemma-3-1b-it          # allow /inference to serve it
make model.deactivate NAME=gemma-3-1b-it        # take it out of /inference (409)
```

## Evaluation

Evaluation runs inside an experiment, not `/inference`. An experiment pins a
model and decoding config; running it infers, stores the inference, and (when the
run names `metrics`) sends the interaction to `arc-eval` for scoring. Each metric
score (faithfulness, answer relevance, and so on) is persisted in
`evaluation_results`, linked to the inference id. Evaluation is a separate service
boundary, so inference storage never depends on it.

Point the service at a running `arc-eval` and set a request timeout:

```bash
ARC_EVAL_SERVICE_URL=http://localhost:8001   # empty disables evaluation
ARC_EVAL_TIMEOUT_SECONDS=30
```

Create an experiment, then run it with the metrics to score. An unknown metric
name returns `404`:

```bash
curl -s http://localhost:8000/experiments \
  -H 'content-type: application/json' \
  -d '{"name": "greedy-baseline", "model_name": "qwen2.5-1.5b-instruct"}'

curl -s http://localhost:8000/experiments/<experiment-id>/run \
  -H 'content-type: application/json' \
  -d '{"input_text": "A long article.", "metrics": ["faithfulness", "answer_relevance"]}'
```

The run response carries the scores and the experiment id next to the summary:

```json
{
  "id": "8f0c1e2a-...",
  "experiment_id": "1c2d3e4f-...",
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
unreachable or returned an unusable response (runs fail open: the summary is
still returned and stored), or `skipped` when no `ARC_EVAL_SERVICE_URL` is
configured.

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
| `make exp.smoke` | Create, run, and compare one scored experiment (needs arc-eval) |
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
| `ARC_MAX_OUTPUT_TOKENS` | `256` | Default generated token budget |
| `ARC_TEMPERATURE` | `0.0` | Default sampling temperature: 0 = greedy, up to 2.0 |
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
