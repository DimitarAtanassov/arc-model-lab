# arc-model-lab Database Schema

Audience: backend engineers working on persistence or migrations. Reading time: 6 minutes.

The service persists three tables in PostgreSQL 16: `models` (the inference model
catalog), `inference` (one row per executed inference), and `evaluation_results`
(one metric score per row, written when a request asks for scoring). The schema is
owned by Alembic migrations in `migrations/` and mirrored by the SQLAlchemy ORM in
`src/arc_model_lab/db/models.py`.

## Entity relationship diagram

```mermaid
erDiagram
    models ||--o{ inference : produces
    inference ||--o{ evaluation_results : "scored by"

    models {
        uuid id PK
        varchar name UK
        varchar provider
        varchar model_id
        varchar tokenizer_id
        varchar revision "nullable"
        varchar adapter_path "nullable"
        varchar status "default active"
        timestamptz created_at
        timestamptz updated_at
    }

    inference {
        uuid id PK
        uuid model_id FK
        text input_text
        text prompt
        text output_text
        integer latency_ms
        integer prompt_tokens "nullable"
        integer completion_tokens "nullable"
        timestamptz created_at
    }

    evaluation_results {
        uuid id PK
        uuid inference_id FK
        text metric_name
        double score
        text reasoning "nullable"
        text evaluator_name
        text evaluator_version "nullable"
        timestamptz created_at
    }
```

## Table: models

The catalog of loadable inference models. `name` is the stable handle used by API
requests and the CLI; the HuggingFace coordinates (`model_id`, `tokenizer_id`,
`revision`) are loading details.

| Column | Type | Nullable | Default | Notes |
| --- | --- | --- | --- | --- |
| `id` | uuid | no | application | Primary key, generated in the app with `uuid4` |
| `name` | varchar(255) | no | | Unique catalog handle |
| `provider` | varchar(255) | no | | Runtime family; currently only `huggingface` |
| `model_id` | varchar(255) | no | | HuggingFace model identifier |
| `tokenizer_id` | varchar(255) | no | | HuggingFace tokenizer identifier |
| `revision` | varchar(255) | yes | null | Pinned model revision; null loads the default |
| `adapter_path` | varchar(1024) | yes | null | Optional LoRA adapter path |
| `status` | varchar(32) | no | `active` | One of `active`, `inactive`, `deprecated` |
| `created_at` | timestamptz | no | `now()` | Row creation time (DB default) |
| `updated_at` | timestamptz | no | `now()` | Refreshed on update via ORM `onupdate` |

Constraints:

- `pk_models` primary key on (`id`).
- `uq_models_name` unique on (`name`).
- `ck_models_valid_status` check: `status IN ('active', 'inactive', 'deprecated')`.

`updated_at` is maintained by SQLAlchemy (`onupdate=func.now()`), not a database
trigger. A write that bypasses the ORM will not refresh it.

## Table: inference

One row per executed summarization. This is the durable record that later
capabilities (evaluation, dataset extraction, regression analysis) read from, so
rows are append-only and are not deleted in normal operation.

| Column | Type | Nullable | Default | Notes |
| --- | --- | --- | --- | --- |
| `id` | uuid | no | application | Primary key, generated in the app with `uuid4` |
| `model_id` | uuid | no | | Foreign key to `models.id` |
| `input_text` | text | no | | Original user payload |
| `prompt` | text | no | | Rendered chat prompt sent to the model |
| `output_text` | text | no | | Generated model output |
| `latency_ms` | integer | no | | Generation wall-clock time in milliseconds |
| `prompt_tokens` | integer | yes | null | Prompt token count when available |
| `completion_tokens` | integer | yes | null | Completion token count when available |
| `created_at` | timestamptz | no | `now()` | Row creation time (DB default) |

Constraints:

- `pk_inference` primary key on (`id`).
- `fk_inference_model_id_models` foreign key (`model_id`) references `models(id)`
  `ON DELETE RESTRICT`.

## Table: evaluation_results

One metric score for one inference, written by the scoring step when a request
asks for it. One metric per row (not a JSON blob) keeps scores easy to query,
index, and aggregate. This is a lean local copy of the score; `arc-eval` keeps its
own richer record (judge, prompt, latency) in its own database.

| Column | Type | Nullable | Default | Notes |
| --- | --- | --- | --- | --- |
| `id` | uuid | no | application | Primary key, generated in the app with `uuid4` |
| `inference_id` | uuid | no | | Foreign key to `inference.id`; the scored inference |
| `metric_name` | text | no | | The metric that produced the score, for example `faithfulness` |
| `score` | double precision | no | | The metric score |
| `reasoning` | text | yes | null | Short rationale from the evaluator, when provided |
| `evaluator_name` | text | no | | The evaluator that scored the metric |
| `evaluator_version` | text | yes | null | Evaluator version, when provided |
| `created_at` | timestamptz | no | `now()` | Row creation time (DB default) |

Constraints:

- `pk_evaluation_results` primary key on (`id`).
- `fk_evaluation_results_inference_id_inference` foreign key (`inference_id`)
  references `inference(id)` `ON DELETE CASCADE`.
- `uq_evaluation_results_inference_metric_evaluator` unique on (`inference_id`,
  `metric_name`, `evaluator_name`).

The unique key is what makes replay and backfill idempotent: re-scoring an
inference updates the existing row for that metric and evaluator instead of adding
a duplicate. `ON DELETE CASCADE` means deleting an inference also removes its
scores, since a score has no meaning without the inference it describes.

## Relationships

`models` to `inference` is one to many. Every `inference` row references exactly
one model (`model_id` is not null); a model may have zero or more inferences.

`ON DELETE RESTRICT` blocks deletion of a model while any inference references it.
This preserves the provenance of stored inferences: a model row cannot disappear
out from under the records it produced. To retire a model, set `status` to
`inactive` or `deprecated` rather than deleting the row.

`inference` to `evaluation_results` is also one to many. Every score references
exactly one inference; an inference has zero or more scores, one per metric.
Unlike the model relationship, this one cascades: deleting an inference removes
its scores too.

## Indexes

`models` and `inference` carry only the indexes their constraints create:

- `pk_models`, `pk_inference` (primary keys).
- `uq_models_name` (unique on `models.name`).

There is no standalone index on `inference.model_id` or `inference.created_at`.
Filtering or joining large inference volumes by model or time will plan
sequential scans. Covering indexes are deferred until query patterns and data
volume justify the write cost.

`evaluation_results` is indexed for its read paths: `inference_id`, `metric_name`,
and `created_at` each have a b-tree index, alongside the unique key on
(`inference_id`, `metric_name`, `evaluator_name`). These keep score-by-inference,
score-by-metric, and time-window queries off sequential scans.

## Migration lineage

| Revision | File | Change |
| --- | --- | --- |
| `0001_initial` | `migrations/versions/0001_initial.py` | Creates `models` and `inference` with primary keys and the `inference -> models` foreign key |
| `0002_model_catalog_fields` | `migrations/versions/0002_model_catalog_fields.py` | Adds `models.revision`, `models.status`, `models.updated_at`, and the `ck_models_valid_status` check |
| `0003_evaluation_results` | `migrations/versions/0003_evaluation_results.py` | Creates `evaluation_results` with its foreign key to `inference`, the unique key, and the read indexes |

Constraint names are deterministic because `Base.metadata` sets a naming
convention in `src/arc_model_lab/db/base.py`. Keep it in place so autogenerated
migrations stay stable.

## ORM and domain mapping

Each table maps to one ORM record and one frozen domain dataclass. Repositories
in `src/arc_model_lab/db/repositories.py` translate between them; ORM types never
cross that boundary.

| Table | ORM record | Domain entity |
| --- | --- | --- |
| `models` | `ModelRecord` | `Model` |
| `inference` | `InferenceRecord` | `Inference` |
| `evaluation_results` | `EvaluationResultRecord` | `EvaluationResult` |

The domain entities and the service flow that writes these rows are described in
[architecture.md](architecture.md).
