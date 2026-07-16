# Running inference

Audience: engineers new to arc-model-lab. Reading time: 4 minutes.

The lab does one thing: run a model on some text and store the result. It never
scores its output. Scoring and experiments live in the separate arc-eval-service.
The two services do not call each other; arc-platform orchestrates them.

## Endpoints at a glance

| Method and path | What it is for |
| --- | --- |
| `POST /inference` | Run an active model once and store the output (online serving). |
| `GET /inference` | List recent inferences, newest first. |
| `GET /inference/{id}` | Fetch one inference by id. |
| `GET /models` | List the catalog models. |
| `GET /models/{name}` | Fetch one catalog model by name. |
| `GET /health` | Liveness check. |

## Before you start

The service is running and the catalog is seeded (see the
[README quickstart](../README.md#quickstart-local)):

```bash
curl -s http://localhost:8000/health        # -> {"status":"ok"}
```

The model you name on `/inference` must be **active**. An unknown name returns
`404`; an inactive one returns `409`. The seed file (`seeds/models.local.json`) sets
each model's status; change it there and re-run `make model.seed`.

Examples below use `qwen2.5-1.5b-instruct` (from `seeds/models.local.json`).

## Run an inference

`temperature` is optional: `0` is greedy and deterministic, higher (up to `2.0`)
samples more freely. Omit it to use the server default (`ARC_TEMPERATURE`). Output
length is set server-side (`ARC_MAX_OUTPUT_TOKENS`), not by the caller.

```bash
curl -s http://localhost:8000/inference \
  -H 'content-type: application/json' \
  -d '{
    "model_name": "qwen2.5-1.5b-instruct",
    "input_text": "Customer: order #4021 arrived cracked and support has not replied in three days. I want a replacement or a refund."
  }'
```

The response is the stored row:

```json
{
  "id": "8f0c1e2a-...",
  "model_id": "3b1a9d4f-...",
  "input_text": "Customer: order #4021 ...",
  "prompt": "<rendered chat prompt>",
  "output_text": "A concise summary.",
  "latency_ms": 812,
  "prompt_tokens": 42,
  "completion_tokens": 18,
  "created_at": "2026-07-05T12:00:00Z"
}
```

Read it back or list recent rows:

```bash
curl -s http://localhost:8000/inference/8f0c1e2a-...     # one inference by id
curl -s "http://localhost:8000/inference?limit=20"       # recent, newest first
```

## Manage the catalog

The catalog is defined in `seeds/models.local.json` and applied with
`make model.seed` (an idempotent upsert keyed by name). Each entry sets the model's
`status`; `/inference` serves only `active` models. To add, retire, or re-activate a
model, edit that file and re-run `make model.seed`.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `404` model not found | Name not in the catalog | `make model.seed` (registers the seed models) |
| `409` model is not active | Model status is not `active` | Set `status` to `active` in `seeds/models.local.json` and re-run `make model.seed` |
| `413` too large | Input over 50,000 characters | Shorten the input |
| `422` invalid body | Missing field, or a stale `metrics`/`max_output_tokens` on `/inference` | `/inference` takes only `model_name`, `input_text`, and optional `temperature` |

## See also

- [architecture.md](architecture.md): components and request flow.
- [dataflow.md](dataflow.md): how a request becomes a stored inference.
- Scoring and experiments now live in the arc-eval-service repository.
