# Running inference

Audience: engineers new to arc-model-lab. Reading time: 4 minutes.

The lab does one thing: run a model on some text and store the result. It never
scores its output. Scoring and experiments live in the separate arc-eval-service,
which calls back into the lab (`POST /v1/inference:run`) to run a candidate model.

## Endpoints at a glance

| Method and path | What it is for |
| --- | --- |
| `POST /inference` | Run an active model once and store the output (online serving). |
| `POST /v1/inference:run` | Service-to-service: run a named model with an explicit generation config, optionally allowing an inactive model. |
| `GET /inference` | List recent inferences, newest first. |
| `GET /inference/{id}` | Fetch one inference by id. |
| `GET /health` | Liveness check. |

## Before you start

The service is running and the catalog is seeded (see the
[README quickstart](../README.md#quickstart-local)):

```bash
curl -s http://localhost:8000/health        # -> {"status":"ok"}
make model.list                             # names you can pass as model_name
```

The model you name on `/inference` must be **active**. An unknown name returns
`404`; an inactive one returns `409`. Activate one with `make model.activate NAME=...`.

Examples below use `qwen2.5-1.5b-instruct`; swap in a name from `make model.list`.

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

## Service-to-service: `/v1/inference:run`

arc-eval-service calls this to run a candidate model for an experiment. It takes an
explicit `generation_config` and an `allow_inactive` flag, so it can run a model
that is not yet active. The response is the same inference row shape.

```bash
curl -s http://localhost:8000/v1/inference:run \
  -H 'content-type: application/json' \
  -d '{
    "model_name": "qwen2.5-1.5b-instruct",
    "input_text": "A long article about battery recycling.",
    "generation_config": {"temperature": 0.0, "max_output_tokens": 256},
    "allow_inactive": true
  }'
```

`allow_inactive` defaults to `false` (the endpoint fails closed, like `/inference`);
arc-eval-service opts in by sending `allow_inactive: true`. `generation_config`
defaults to the server decoding config when omitted.

## Manage the catalog

```bash
make model.list                                 # list registered models
make model.get NAME=qwen2.5-1.5b-instruct       # show one model
make model.smoke NAME=qwen2.5-1.5b-instruct     # load + run one summary
make model.activate NAME=gemma-3-1b-it          # allow /inference to serve it
make model.deactivate NAME=gemma-3-1b-it        # take it out of /inference (409)
```

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `404` model not found | Name not in the catalog | `make model.list`, then `make model.seed` |
| `409` model is not active | Model deactivated | `make model.activate NAME=...`, or call `/v1/inference:run` with `allow_inactive` |
| `413` too large | Input over 50,000 characters | Shorten the input |
| `422` invalid body | Missing field, or a stale `metrics`/`max_output_tokens` on `/inference` | `/inference` takes only `model_name`, `input_text`, and optional `temperature` |

## See also

- [architecture.md](architecture.md): components and request flow.
- [dataflow.md](dataflow.md): how a request becomes a stored inference.
- Scoring and experiments now live in the arc-eval-service repository.
