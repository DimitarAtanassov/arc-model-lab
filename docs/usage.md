# Running inference, evaluation, and experiments

Audience: engineers new to arc-model-lab. Reading time: 8 minutes.

This guide shows how to do the three things you will do most:

- **Inference**: run a model on some text and store the result.
- **Evaluation**: score stored outputs with the `arc-eval` service.
- **Experiments**: pin a model and decoding config, run it, and score it, so runs are reproducible and comparable.

Rule of thumb: `/inference` is the fast path and never scores. Scoring always
happens separately, either inside an experiment run or through the evaluation
CLI on rows that already exist.

## The three concepts

Three ideas power this service. They build on each other, so it helps to keep
them straight.

| Concept | What it is | Where the work happens |
| --- | --- | --- |
| **Inference** | One run of a model on some text, saved as a row. | Local, in `arc-model-lab`. |
| **Evaluation** | A quality score for one inference (for example `faithfulness`). | Remote, in the `arc-eval` service. |
| **Experiment** | A named, fixed configuration (a model plus decoding settings) you run and re-run to compare. | `arc-model-lab` orchestrates; scoring is delegated to `arc-eval`. |

### Inference vs evaluation

They answer different questions:

- **Inference** answers *"what did the model produce?"* It generates text and
  stores it. It never scores.
- **Evaluation** answers *"how good was that output?"* It takes an inference that
  already exists and asks `arc-eval` to score it against named metrics.

You can always infer without evaluating. You can only evaluate something that was
already inferred.

### How an experiment differs from an evaluation

An **evaluation** is a single scoring action on one inference. An **experiment**
is the harness that makes those scores *comparable*:

- An experiment **pins the configuration** (which model, what temperature, how
  many output tokens) under a name, so every run is reproducible.
- Running an experiment does the whole loop in one call: infer, store, score
  (when you ask for it), and record the run under the experiment.
- Because every run is recorded under its experiment, you can **aggregate and
  compare** scores across configurations with a single request.

Put simply: an evaluation is one score; an experiment is a labeled, repeatable
harness that collects those scores so you can compare configuration A against B.

## Endpoints at a glance

The service exposes one inference endpoint and a small set of experiment
endpoints. Evaluation has no endpoint of its own in `arc-model-lab`: it runs
inside an experiment `run`, or from the evaluation CLI on rows that already
exist.

| Method and path | What it is for |
| --- | --- |
| `POST /inference` | Run a model once and store the output. No scoring, no experiment link. |
| `POST /experiments` | Define an experiment: a name, a model, and a decoding config. |
| `GET /experiments/{id}` | Fetch one experiment's configuration. |
| `POST /experiments/{id}/run` | Run the experiment once: infer, store, link, and (with `metrics`) score. |
| `GET /experiments/{id}/results` | Aggregated scores for the experiment, per metric. |
| `GET /experiments/{id}/compare/{other_id}` | Results for two experiments, side by side. |
| `GET /health` | Liveness check. |

The experiment lifecycle is three steps, in order: **define** it
(`POST /experiments`), **run** it one or more times (`POST /experiments/{id}/run`),
then **read** the aggregated scores (`GET /experiments/{id}/results`, or
`.../compare/...` for two side by side).

## Before you start

1. The service is running and the catalog is seeded. See the
   [README quickstart](../README.md#quickstart-local). Check it is up:

   ```bash
   curl -s http://localhost:8000/health        # -> {"status":"ok"}
   make model.list                             # names you can pass as model_name
   ```

2. The model you name must be **active**. An unknown name returns `404`; an
   inactive one returns `409`. Activate one with `make model.activate NAME=...`.

3. For evaluation only: point the service at a running `arc-eval` instance.
   An empty value disables scoring (outcomes come back `skipped`, not an error).

   ```bash
   export ARC_EVAL_SERVICE_URL=http://localhost:8001
   ```

Examples below use the model name `qwen2.5-1.5b-instruct`. Swap in a name from
`make model.list`.

---

## 1. Inference

Run a model on `input_text` and store one inference row. `temperature` is
optional: `0` is greedy and deterministic, higher (up to `2.0`) samples more
freely. Omit it to use the server default (`ARC_TEMPERATURE`). Output length is
set server-side, not by the caller.

### From the API

Summarize a support ticket (greedy, the reproducible default):

```bash
curl -s http://localhost:8000/inference \
  -H 'content-type: application/json' \
  -d '{
    "model_name": "qwen2.5-1.5b-instruct",
    "input_text": "Customer: order #4021 arrived with a cracked screen and support has not replied in three days. I want a replacement or a refund."
  }'
```

Summarize a product description with more varied phrasing (sampling):

```bash
curl -s http://localhost:8000/inference \
  -H 'content-type: application/json' \
  -d '{
    "model_name": "qwen2.5-1.5b-instruct",
    "input_text": "The TrailLite 2 is a two-person tent weighing 1.9kg, with a waterproof rating of 3000mm and a 10-minute pitch.",
    "temperature": 0.7
  }'
```

The response is the stored row. It carries no `experiment_id` and no scores,
because `/inference` never evaluates:

```json
{
  "id": "8f0c1e2a-...",
  "model_id": "3b1a9d4f-...",
  "input_text": "Customer: order #4021 ...",
  "output_text": "A concise summary.",
  "latency_ms": 812,
  "prompt_tokens": 42,
  "completion_tokens": 18,
  "created_at": "2026-07-05T12:00:00Z"
}
```

Keep the `id`; you need it to score the output later.

### From the command line

Load a model and print one summary (useful for a quick check):

```bash
make model.smoke NAME=qwen2.5-1.5b-instruct
```

---

## 2. Evaluation

Evaluation scores a **stored** inference against one or more metrics (for
summarization: `faithfulness` and `answer_relevance`) and saves each score next
to the inference. Use this to score history or fill gaps. To score brand-new
output instead, use an experiment (next section).

Set `ARC_EVAL_SERVICE_URL` first (see [Before you start](#before-you-start)).

Score a single inference by id (the `id` from an inference or experiment run):

```bash
make eval.run ID=8f0c1e2a-...
```

Score every inference that has no scores yet:

```bash
make eval.replay                 # add LIMIT=500 to raise the batch cap
```

Backfill a time range (for example, yesterday's traffic):

```bash
make eval.backfill SINCE=2026-07-01 UNTIL=2026-07-02
```

Choose specific metrics with the raw CLI (the `make` targets use the default
set). Re-running is safe: scores upsert on the metric, they do not duplicate.

```bash
uv run python -m arc_model_lab.cli.evaluations run \
  --inference-id 8f0c1e2a-... \
  --metrics faithfulness answer_relevance
```

An outcome is `completed` when arc-eval scored it, `skipped` when no
`ARC_EVAL_SERVICE_URL` is set, and `failed` when arc-eval was unreachable (the
inference is untouched). An unknown metric name returns `404`.

---

## 3. Experiments

An experiment pins a model and a decoding config (temperature, output length)
under a name. Running it infers, stores the inference, links that inference to
the experiment, and, when you name metrics, scores it. The inference row stays
clean (it holds no experiment id); the link lives in a separate `experiment_runs`
record. Because every run is linked, you can compare configurations with plain
aggregates.

Use experiments in three steps:

1. **Define** the experiment (`POST /experiments`): a name, a model, a config.
2. **Run** it (`POST /experiments/{id}/run`): once per input; name `metrics` to score.
3. **Read** the scores (`GET /experiments/{id}/results`, or `.../compare/...`).

A run response is the inference, its experiment link, and (when scored) the
evaluation block:

```json
{
  "id": "1c7f2b8e-...",
  "model_id": "3b1a9d4f-...",
  "input_text": "A long article about battery recycling.",
  "output_text": "A concise summary.",
  "latency_ms": 734,
  "prompt_tokens": 55,
  "completion_tokens": 20,
  "experiment_id": "9a2d5c11-...",
  "created_at": "2026-07-05T12:05:00Z",
  "evaluation": {
    "status": "completed",
    "results": [
      {
        "metric_name": "faithfulness",
        "score": 0.91,
        "evaluator_name": "summary-faithfulness",
        "evaluator_version": "v1"
      }
    ]
  }
}
```

Here `id` is the inference id and `experiment_id` names the experiment this run
belongs to. The `evaluation` block is present only when you named `metrics`.
Compare this with the `/inference` response above: the inference fields are
identical, but a run adds `experiment_id` and `evaluation`; `/inference` has
neither.

### Example A: greedy vs creative decoding

Create two experiments on the same model with different temperatures, run both
on the same input, then compare their average scores.

```bash
# Greedy baseline (make sets no temperature, so it uses the default 0.0)
make exp.create NAME=greedy MODEL=qwen2.5-1.5b-instruct
# -> <exp-A>   greedy     qwen2.5-1.5b-instruct

# Creative variant (raw CLI, so we can set --temperature)
uv run python -m arc_model_lab.cli.experiments create \
  --name creative --model-name qwen2.5-1.5b-instruct --temperature 0.9
# -> <exp-B>   creative   qwen2.5-1.5b-instruct

# Run each with scoring
uv run python -m arc_model_lab.cli.experiments run \
  --experiment-id <exp-A> \
  --input-text "A long article about battery recycling." \
  --metrics faithfulness answer_relevance

uv run python -m arc_model_lab.cli.experiments run \
  --experiment-id <exp-B> \
  --input-text "A long article about battery recycling." \
  --metrics faithfulness answer_relevance

# Compare average scores side by side
make exp.compare ID=<exp-A> OTHER=<exp-B>
```

`run` prints the inference id, the summary, and the scores. `compare` prints one
line per metric for each experiment (`(no scores)` if nothing was evaluated).

### Example B: A/B two models on the same input (via the API)

```bash
# Create one experiment per model
curl -s http://localhost:8000/experiments \
  -H 'content-type: application/json' \
  -d '{"name": "model-a", "model_name": "qwen2.5-1.5b-instruct"}'

curl -s http://localhost:8000/experiments \
  -H 'content-type: application/json' \
  -d '{"name": "model-b", "model_name": "gemma-3-1b-it"}'

# Run each with the same input and metrics (use the ids from the responses)
curl -s http://localhost:8000/experiments/<exp-a-id>/run \
  -H 'content-type: application/json' \
  -d '{"input_text": "A long article.", "metrics": ["faithfulness", "answer_relevance"]}'

curl -s http://localhost:8000/experiments/<exp-b-id>/run \
  -H 'content-type: application/json' \
  -d '{"input_text": "A long article.", "metrics": ["faithfulness", "answer_relevance"]}'

# Aggregated scores for one, or both side by side
curl -s http://localhost:8000/experiments/<exp-a-id>/results
curl -s http://localhost:8000/experiments/<exp-a-id>/compare/<exp-b-id>
```

Omit `metrics` on `run` to generate output without scoring it. The run response
carries the `experiment_id` and, when scored, the `evaluation` block.

### One-command smoke check

Create, run, and score one experiment end to end (needs `ARC_EVAL_SERVICE_URL`):

```bash
make exp.smoke MODEL=qwen2.5-1.5b-instruct
```

---

## How they fit together

| You want to... | Use | Scores? | Reproducible config? |
| --- | --- | --- | --- |
| Get a summary fast | `/inference` | No | No (per-request) |
| Score output you just generated | Experiment `run` with `metrics` | Yes | Yes |
| Score outputs already in the database | `make eval.run` / `eval.replay` / `eval.backfill` | Yes | n/a |
| Compare two configs or models | Two experiments, then `compare` | Yes | Yes |

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `404` model not found | Name not in the catalog | `make model.list`, then `make model.seed` |
| `409` model is not active | Model deactivated | `make model.activate NAME=...` |
| `evaluation.status` is `skipped` | No arc-eval configured | Set `ARC_EVAL_SERVICE_URL` |
| `404` on run/eval with a metric | Metric not defined in arc-eval | Use a valid metric name (e.g. `faithfulness`) |
| `413` too large | Input over 50,000 characters | Shorten the input |

## See also

- [end-to-end-testing.md](end-to-end-testing.md): a copy-paste walkthrough of the full stack, from inference to scored experiments.
- [architecture.md](architecture.md): components and request flow.
- [dataflow.md](dataflow.md): how a run moves through inference and evaluation.
