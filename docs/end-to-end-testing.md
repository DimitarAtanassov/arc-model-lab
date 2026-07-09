# End-to-end testing: inference

Audience: engineers validating the lab locally. Reading time: 6 minutes.

This is a copy-paste walkthrough. You will start `arc-model-lab`, run inference
over both endpoints, read the rows back, and confirm they persisted. The lab is
pure model serving: it stores inferences and never scores them. Scoring and
experiments live in arc-eval-service, which has its own end-to-end guide.

## What you will run

One service and its database:

```mermaid
flowchart LR
    You[You: curl + CLI] -->|":8000"| ML["arc-model-lab (make run, :8000)"]
    ML --> PG[("Postgres :5432")]
```

## Prerequisites

- macOS or Linux, with [`uv`](https://docs.astral.sh/uv/), Docker + Docker Compose,
  `curl`, and [`jq`](https://jqlang.github.io/jq/).
- This repository checked out.

Set the repo path once (adjust to where you cloned it):

```bash
export ML=~/playground/arc/arc-model-lab
```

Use two terminals: **Terminal A** runs the server and stays open; **Terminal B**
runs the test commands.

## Setup: start arc-model-lab

In **Terminal B**, prepare the database and catalog:

```bash
cd "$ML"
cp .env.example .env                          # first time only
docker compose up -d postgres                 # Postgres on :5432
uv sync                                        # create the venv
make migrate                                   # apply schema (loads .env)
make model.seed                                # seed the model catalog
make model.list                                # confirm: qwen2.5-1.5b-instruct, gemma-3-1b-it
```

In **Terminal A**, start the server and leave it running:

```bash
cd "$ML"
make run                                        # serves on http://localhost:8000
```

Back in **Terminal B**, confirm it is up:

```bash
curl -s localhost:8000/health | jq            # -> {"status":"ok"}
```

## Walkthrough

Run everything below in **Terminal B**. A sample article is reused; `jq` builds
each body so the text quotes safely.

```bash
export ARTICLE="The city council approved a plan to add 20 miles of protected bike lanes over three years, funded by a state grant. Supporters say the lanes cut commute times; critics worry about reduced parking."
```

### 1. Health check

```bash
curl -s localhost:8000/health | jq            # -> {"status":"ok"}
```

### 2. Run a plain inference

`POST /inference` runs an active model and stores one row. The first call downloads
the model weights and may take a minute; later calls are fast.

```bash
RESP=$(curl -s localhost:8000/inference \
  -H 'content-type: application/json' \
  -d "$(jq -n --arg m qwen2.5-1.5b-instruct --arg t "$ARTICLE" '{model_name:$m, input_text:$t}')")

echo "$RESP" | jq                             # the stored inference row
INFERENCE_ID=$(echo "$RESP" | jq -r '.id')
echo "inference id: $INFERENCE_ID"
```

The response is the stored inference. It carries no scores: `/inference` is pure
inference.

### 3. Read it back and list recent rows

```bash
curl -s localhost:8000/inference/$INFERENCE_ID | jq       # one inference by id
curl -s "localhost:8000/inference?limit=5" | jq           # recent, newest first
```

### 4. Run the service-to-service endpoint

`POST /v1/inference:run` is what arc-eval-service calls. It takes an explicit
generation config and can run an inactive candidate model.

```bash
curl -s localhost:8000/v1/inference:run \
  -H 'content-type: application/json' \
  -d "$(jq -n --arg m qwen2.5-1.5b-instruct --arg t "$ARTICLE" \
    '{model_name:$m, input_text:$t, generation_config:{temperature:0.0, max_output_tokens:256}, allow_inactive:true}')" | jq
```

### 5. Confirm the active-model gate

`/inference` serves only active models. Deactivate one and watch it return `409`,
then confirm `/v1/inference:run` still runs it with `allow_inactive`.

```bash
make model.deactivate NAME=gemma-3-1b-it

curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/inference \
  -H 'content-type: application/json' \
  -d '{"model_name":"gemma-3-1b-it","input_text":"hi"}'                          # -> 409

curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/v1/inference:run \
  -H 'content-type: application/json' \
  -d '{"model_name":"gemma-3-1b-it","input_text":"hi","allow_inactive":true}'   # -> 201

make model.activate NAME=gemma-3-1b-it
```

### 6. Verify the rows in the database

```bash
cd "$ML"
docker compose exec postgres psql -U arc -d arc_model_lab -c \
  "SELECT id, left(output_text, 50) AS output FROM inference ORDER BY created_at DESC LIMIT 5;"
```

You should see the rows from steps 2 and 4. There are no score or experiment tables
in the lab: those live in arc-eval-service.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `404` on `/inference` | Model name not in the catalog | `make model.list`, then `make model.seed` |
| `409` on `/inference` | Model is not active | `make model.activate NAME=...` |
| First inference is slow | Weights downloading on first use | Wait; later calls are fast |
| `413` too large | Input over 50,000 characters | Shorten the input |

## See also

- [usage.md](usage.md): the endpoints in detail.
- Scoring and experiments end to end: the arc-eval-service repository.
