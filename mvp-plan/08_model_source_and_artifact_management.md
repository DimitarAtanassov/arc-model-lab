# 08 - Model Source and Artifact Management

> Current state: implemented. The `models` catalog, `seeds/models.local.json`,
> the `seed_models` loader, the `model.*` CLI, and the `hf_cache` volume all
> exist and match this document. What remains is the next step in the Seed File
> Strategy section: make the catalog declarative (reconcile, not upsert) and
> environment-aware, or move it to a config repo.

## Purpose

This document defines how `arc-model-lab` manages model configuration, model source resolution, model file caching, and future model artifact storage.

The service must not store model weights, tokenizer files, or large artifacts in the codebase or Docker image.

Instead, the service should follow a clean enterprise pattern:

```text
Codebase stores service logic.
Database stores model metadata.
External model source stores model files.
Runtime cache stores downloaded files.
Inference records store which model was used.
```

For the Initial Slice, the recommended model source is Hugging Face Hub. The internal `models` table acts as the service-owned catalog. The runtime downloads model and tokenizer files into a local cache directory outside the repository.

The system starts with:

```text
Postgres model catalog -> Hugging Face Hub -> local cache -> runtime model
```

Later, after training and registry capabilities exist, the platform can evolve toward:

```text
Postgres model registry -> object storage / OCI artifacts -> validated runtime cache
```

This document is intentionally scoped to the Initial Slice and a future upgrade path. It should not introduce MLflow, DVC, MinIO, OCI artifacts, or a custom model registry before they are needed.

## Why This Matters

Model files are large, mutable, and operationally sensitive.

If model files are stored in the codebase:

- repository size grows quickly
- pull requests become slow
- code and model artifact lifecycles become coupled
- CI becomes expensive and unreliable
- security scanning becomes noisy
- rollback semantics become unclear

If model files are stored inside Docker images too early:

- image sizes become very large
- image builds become slow
- every model change requires an image rebuild
- promotion of code and promotion of model artifacts become coupled
- multiple models require multiple images or bloated images

The better pattern is to separate concerns:

| Concern | Owner |
|---|---|
| Service logic | Git repository |
| Runtime configuration | Environment variables |
| Model metadata | Postgres `models` table |
| Base model files | Hugging Face Hub in Initial Slice |
| Runtime file cache | Local disk or Docker volume |
| Future trained artifacts | Object storage or OCI registry |
| Future lifecycle management | Model registry service/table |

This keeps the service maintainable, extendable, operable, evolvable, and scalable.

## Initial Slice Architecture

The Initial Slice uses:

- `models` table as the internal model catalog
- Hugging Face Hub as the external source of base model and tokenizer files
- local Hugging Face cache as the runtime cache
- `ModelService` as the resolver/loader
- `InferenceService` as the orchestration layer

```mermaid
flowchart LR
    API[FastAPI API] --> InferenceService[InferenceService]
    InferenceService --> ModelRepo[ModelRepository]
    ModelRepo --> DB[(Postgres models table)]

    InferenceService --> ModelService[ModelService]
    ModelService --> HFHub[Hugging Face Hub]
    HFHub --> Cache[Local HF Cache]
    Cache --> Runtime[Loaded Runtime Model]

    InferenceService --> InferenceRepo[InferenceRepository]
    InferenceRepo --> DB
```

The important separation is:

```text
ModelRepository:
  knows which model should be used

ModelService:
  knows how to load and run that model

Hugging Face Hub:
  stores base model and tokenizer files

Local cache:
  stores downloaded files for reuse

InferenceRepository:
  records which model produced each output
```

## Design Principles

### Do not store model files in Git

The repository should not contain:

- model weights
- tokenizer files
- LoRA adapters
- quantized files
- downloaded Hugging Face snapshots
- local model cache directories

Add ignore rules early:

```text
artifacts/
models/
.cache/
.huggingface/
*.safetensors
*.bin
*.gguf
```

### Do not store model files in Postgres

Postgres should store:

- model metadata
- model source references
- revision identifiers
- status
- lineage
- artifact URIs later

Postgres should not store:

- model weights
- tokenizer files
- adapter binary files
- quantized model files

The `models` table is a catalog, not a file store.

### Do not bake model files into the Docker image for Initial Slice

The Docker image should contain:

- application code
- Python dependencies
- startup command

The Docker image should not contain:

- base model weights
- tokenizer snapshots
- local Hugging Face cache

For local development, use a Docker volume for the cache. In production, use either a persistent volume, pre-warmed cache, or artifact download step depending on the deployment environment.

### Pin model revisions for reproducibility

Model identifiers such as `main` are convenient, but they are mutable.

For local development, `revision = "main"` is acceptable.

For controlled environments, pin the exact commit hash or tag.

The service should treat this tuple as the source identity:

```text
provider + model_id + tokenizer_id + revision + adapter_path
```

### Keep model resolution data-driven

Service code should not hardcode model identifiers.

Avoid:

```python
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
```

Prefer:

```python
model = model_repository.get_by_name("qwen-1.5b-instruct")
result = model_service.generate(model=model, prompt=prompt)
```

This allows model changes through catalog updates instead of code changes.

## Recommended Initial Schema

The Initial Slice schema should add a few fields beyond the smallest possible version so the service remains reproducible and operable without becoming over-engineered.

```sql
CREATE TABLE models (
    id UUID PRIMARY KEY,

    name TEXT NOT NULL UNIQUE,
    provider TEXT NOT NULL,

    model_id TEXT NOT NULL,
    tokenizer_id TEXT NOT NULL,
    revision TEXT,

    adapter_path TEXT,

    status TEXT NOT NULL DEFAULT 'active',

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Recommended indexes:

```sql
CREATE UNIQUE INDEX uq_models_name ON models(name);
CREATE INDEX ix_models_provider ON models(provider);
CREATE INDEX ix_models_status ON models(status);
```

Optional status constraint:

```sql
ALTER TABLE models
ADD CONSTRAINT ck_models_status
CHECK (status IN ('active', 'inactive', 'deprecated'));
```

## Field Rationale

| Field | Purpose |
|---|---|
| `id` | Stable internal identifier |
| `name` | Human-friendly lookup key |
| `provider` | Initially `huggingface`; future-proof without provider abstraction |
| `model_id` | External model source identifier |
| `tokenizer_id` | External tokenizer source identifier |
| `revision` | Branch, tag, or commit hash for reproducibility |
| `adapter_path` | Future LoRA adapter reference |
| `status` | Enables activation/deactivation without deleting rows |
| `created_at` | Auditability |
| `updated_at` | Operational debugging and metadata tracking |

## Domain Entity

```python
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Model:
    id: UUID
    name: str
    provider: str

    model_id: str
    tokenizer_id: str
    revision: str | None

    adapter_path: str | None
    status: str

    created_at: datetime
    updated_at: datetime
```

The domain entity should not contain Hugging Face runtime objects.

Do not add:

```python
tokenizer: AutoTokenizer
model: AutoModelForCausalLM
```

Runtime objects belong inside `ModelService`.

## Model Loading Flow

```mermaid
sequenceDiagram
    participant API
    participant InferenceService
    participant ModelRepository
    participant DB as Postgres
    participant ModelService
    participant HFHub as Hugging Face Hub
    participant Cache as Local Cache
    participant Runtime as Runtime Model

    API->>InferenceService: summarize(input_text)
    InferenceService->>ModelRepository: get_by_name(deployed_model_name)
    ModelRepository->>DB: SELECT model metadata
    DB-->>ModelRepository: model row
    ModelRepository-->>InferenceService: Model domain entity

    InferenceService->>ModelService: generate(model, prompt)
    ModelService->>Cache: check cached files
    alt cache miss
        ModelService->>HFHub: download model/tokenizer by revision
        HFHub-->>Cache: store snapshot
    end
    ModelService->>Runtime: load tokenizer + model
    Runtime-->>ModelService: generated output
    ModelService-->>InferenceService: generation result
```

## ModelService Responsibilities

`ModelService` owns runtime model resolution and generation.

Responsibilities:

- resolve model metadata into a loadable runtime
- load tokenizer
- load model
- use configured cache directory
- cache runtime objects in process
- generate text
- return output and token counts when available
- expose a local smoke test path

Non-responsibilities:

- choosing which model to use
- storing inference records
- storing model metadata
- evaluating output
- training
- model promotion

Recommended internal cache key:

```text
model.name + model.revision + model.adapter_path
```

If `revision` is null, use a stable placeholder such as `default`.

## Example ModelService Loading Logic

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._runtime_cache: dict[str, RuntimeModel] = {}

    def _cache_key(self, model: Model) -> str:
        revision = model.revision or "default"
        adapter = model.adapter_path or "no-adapter"
        return f"{model.name}:{revision}:{adapter}"

    def load(self, model: Model) -> RuntimeModel:
        key = self._cache_key(model)

        if key in self._runtime_cache:
            return self._runtime_cache[key]

        tokenizer = AutoTokenizer.from_pretrained(
            model.tokenizer_id,
            revision=model.revision,
            cache_dir=self._settings.model_cache_dir,
        )

        runtime_model = AutoModelForCausalLM.from_pretrained(
            model.model_id,
            revision=model.revision,
            cache_dir=self._settings.model_cache_dir,
        )

        # The implemented version resolves cuda -> mps -> cpu and errors on an
        # explicitly requested accelerator that is unavailable.
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        runtime_model.to(device)
        runtime_model.eval()

        runtime = RuntimeModel(
            tokenizer=tokenizer,
            model=runtime_model,
            device=device,
        )

        self._runtime_cache[key] = runtime
        return runtime
```

This is intentionally not a provider abstraction. It is a direct Hugging Face implementation because the Initial Slice only supports Hugging Face.

## Runtime Cache Strategy

Use a cache path outside the repository. The implemented paths are:

```text
container path : /var/cache/huggingface   (compose volume hf_cache)
local dev cache : .cache/huggingface       (make model.clear-cache removes it)
```

Environment variable set in compose:

```bash
HF_HOME=/var/cache/huggingface
```

`Settings` exposes the cache directory as `model_cache_dir` (env `ARC_MODEL_CACHE_DIR`), passed to `ModelService` as `cache_dir`.

## Docker Compose Cache Volume

This mirrors the committed `compose.yaml` (sync `psycopg`, `ARC_DATABASE_URL`, and the app runs migrate + seed before serving):

```yaml
services:
  api:
    build: .
    command: ["sh", "-c", "alembic upgrade head && python -m arc_model_lab.db.seed_models seeds/models.local.json && uvicorn arc_model_lab.main:app --host 0.0.0.0 --port 8000"]
    environment:
      ARC_DATABASE_URL: postgresql+psycopg://arc:arc@postgres:5432/arc_model_lab
      HF_HOME: /var/cache/huggingface
    ports:
      - "8000:8000"
    volumes:
      - hf_cache:/var/cache/huggingface
    depends_on:
      postgres:
        condition: service_healthy

  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: arc
      POSTGRES_PASSWORD: arc
      POSTGRES_DB: arc_model_lab
    ports:
      - "5432:5432"

volumes:
  arc_pgdata:
  hf_cache:
```

This prevents repeated model downloads across container restarts.

## Seed File Strategy

Implemented: `seeds/models.local.json` is loaded by `seed_models.py`, which upserts on `name` (idempotent). The current file:

```json
[
  {
    "name": "qwen2.5-1.5b-instruct",
    "provider": "huggingface",
    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "tokenizer_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "revision": "main",
    "adapter_path": null,
    "status": "active"
  },
  {
    "name": "gemma-3-1b-it",
    "provider": "huggingface",
    "model_id": "google/gemma-3-1b-it",
    "tokenizer_id": "google/gemma-3-1b-it",
    "revision": "main",
    "adapter_path": null,
    "status": "active"
  }
]
```

For stronger reproducibility, replace `main` with pinned commit hashes before shared benchmarks or production-like runs.

### Next step: make the catalog declarative

The loader upserts but never removes, and the file is single-environment, so the DB and the file drift. The next step is to reconcile (add/update present entries, deactivate absent ones behind an opt-in flag) and resolve the file per environment off the existing `environment` setting, keeping git as the source of truth. If multiple services need the catalog, move it to a config repo applied at deploy. This stays declarative desired-state config; it does not add a datastore.

## Model Metadata Lifecycle in Initial Slice

```mermaid
stateDiagram-v2
    [*] --> active
    active --> inactive
    inactive --> active
    active --> deprecated
    inactive --> deprecated
    deprecated --> [*]
```

Initial meaning:

| Status | Meaning |
|---|---|
| `active` | Can be used for inference |
| `inactive` | Cannot be used for new inference |
| `deprecated` | Kept for historical reproducibility but not recommended |

Do not delete model rows once inference records reference them. Phase 06 widens these states (draft/validated/active/deprecated/archived).

## Inference Linkage

The `inference` table should keep the foreign key to `models.id`.

```sql
CREATE TABLE inference (
    id UUID PRIMARY KEY,
    model_id UUID NOT NULL REFERENCES models(id),

    input_text TEXT NOT NULL,
    prompt TEXT NOT NULL,
    output_text TEXT NOT NULL,

    latency_ms INTEGER NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

This guarantees every output can answer:

- Which internal model record produced it?
- Which external model source did that record reference?
- Which revision was intended?
- Was an adapter used?

## Make Targets for Initial Slice

These targets are implemented and support model metadata and local cache workflows:

```make
model.seed:
	uv run python -m arc_model_lab.db.seed_models seeds/models.local.json

model.list:
	uv run python -m arc_model_lab.cli.models list

model.get:
	uv run python -m arc_model_lab.cli.models get --name $(name)

model.activate:
	uv run python -m arc_model_lab.cli.models activate --name $(name)

model.deactivate:
	uv run python -m arc_model_lab.cli.models deactivate --name $(name)

model.clear-cache:
	rm -rf .cache/huggingface

model.smoke:
	uv run python -m arc_model_lab.cli.models smoke --name $(name)
```

Minimum required targets:

```make
make model.seed
make model.list
make model.smoke
```

## CI/CD for Initial Slice

CI should not download large model files by default.

Use a fake or tiny model runtime for normal PR checks.

### Pull Request Pipeline

```mermaid
flowchart TD
    A[Pull Request] --> B[Install Dependencies]
    B --> C[Format Check]
    C --> D[Lint]
    D --> E[Type Check]
    E --> F[Unit Tests with Fake Runtime]
    F --> G[Repository Tests]
    G --> H[Migration Validation]
    H --> I[Seed File Validation]
    I --> J[Docker Build]
```

### Seed File Validation

CI should validate:

- JSON parses
- required fields exist
- model names are unique
- status is valid
- provider is supported
- no local filesystem-only paths are accidentally committed as default model source

### Optional Scheduled Smoke Test

A nightly or manual job can download and load the real model:

```text
manual workflow
  -> start postgres
  -> seed models
  -> download selected model
  -> run model.smoke
  -> run summarize smoke test
```

This avoids slowing every PR while still catching upstream dependency issues.

## Local Development Flow

First-time setup (matches the real Makefile targets):

```bash
uv sync
docker compose up -d postgres
make migrate
make model.seed
make model.list
make model.smoke NAME=qwen2.5-1.5b-instruct
make run
```

Inference request (deployed model, no `model_name`):

```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"input_text": "Long article text here"}'
```

## Failure Modes

| Failure | Recommended behavior |
|---|---|
| Model row missing | Return 404 or 400 |
| Model inactive | Return 409 or 400 |
| Hugging Face download failure | Return 503 |
| Cache directory not writable | Fail startup or fail model load |
| Tokenizer load failure | Return 503 |
| Model load failure | Return 503 |
| Generation failure | Return 500 |
| DB persistence failure | Do not return successful inference |
| Revision not found | Return 503 with safe error |

Do not expose internal stack traces or Hugging Face tokens to API callers.

Note: with the deployed-model design, a missing or inactive model is a server-side misconfiguration rather than caller input. The current handlers still map these to 404/409; remapping them to 503 is an open follow-up.

## Security Considerations

- Do not commit Hugging Face tokens.
- Use environment variables or a secret manager for private model access.
- Do not log full prompts, inputs, or outputs at application log level.
- Do not store secrets in `models` table.
- Validate model source metadata before seeding.
- Restrict who can activate or deactivate models in shared environments.
- Treat model source changes as operationally significant.

## Future Upgrade Path

The Initial Slice should start with:

```text
Postgres catalog + Hugging Face Hub + local cache
```

After the full platform is rolled out, the model source and artifact strategy can evolve through the following stages.

## Stage 1: Initial Slice

```mermaid
flowchart LR
    DB[(models table)] --> HF[Hugging Face Hub]
    HF --> Cache[Local HF Cache]
    Cache --> Runtime[Runtime Model]
```

Use for:

- base models
- local inference
- early evaluation
- early experiments

Do not add object storage yet.

## Stage 2: After Training Exists

Once training produces LoRA adapters, store artifacts outside the repo.

Recommended options:

- local artifact directory for MVP
- MinIO for local enterprise-style S3-compatible storage
- S3/GCS/Azure Blob for cloud environments

```mermaid
flowchart LR
    TrainingRun[TrainingRun] --> ArtifactStore[(Object Storage)]
    ArtifactStore --> Models[(models table artifact_uri / adapter_path)]
    Models --> ModelService[ModelService]
    ModelService --> Cache[Runtime Cache]
```

Extend schema:

```sql
ALTER TABLE models
ADD COLUMN artifact_uri TEXT,
ADD COLUMN artifact_checksum TEXT,
ADD COLUMN base_model_id UUID REFERENCES models(id);
```

Use object storage for:

- LoRA adapters
- training configs
- training metrics
- tokenizer overrides if needed
- quantized variants

## Stage 3: After Model Registry Exists

Once there are multiple trained models and promotion decisions, formalize lifecycle management.

```mermaid
flowchart LR
    Training[Training] --> Draft[Draft Model]
    Draft --> Validation[Validation]
    Validation --> Active[Active Model]
    Active --> Deprecated[Deprecated]
    Deprecated --> Archived[Archived]
```

Lifecycle states:

```text
draft
validated
active
deprecated
archived
```

At this stage, the `models` table becomes a lightweight model registry.

Model promotion should be explicit and separate from code deployment.

## Stage 4: Optional MLflow Registry

Use MLflow only if the internal lightweight registry becomes insufficient.

Consider MLflow when you need:

- registry UI
- model stage transitions
- experiment tracking integration
- artifact lineage UI
- cross-team model discovery

Do not introduce MLflow just for the Initial Slice.

## Stage 5: Optional DVC for Dataset and Artifact Versioning

Use DVC when dataset reproducibility becomes painful.

Good fit:

- versioning datasets
- linking Git commits to dataset snapshots
- managing large benchmark files
- reproducible training pipelines

Poor fit:

- online runtime model lookup
- request-time inference loading

DVC should complement dataset/training workflows, not replace the runtime model catalog.

## Stage 6: Optional OCI Artifacts / ORAS

Use OCI artifacts when the organization wants model artifacts promoted like container images.

```mermaid
flowchart LR
    Build[Training Build] --> OCI[OCI Registry]
    OCI --> Dev[Dev]
    Dev --> Stage[Stage]
    Stage --> Prod[Prod]
```

Good fit:

- immutable artifact promotion
- enterprise registry controls
- artifact signing
- environment promotion workflows

This is a later enterprise hardening path, not an Initial Slice requirement.

## Recommended Long-term Target

The long-term model artifact architecture should look like:

```mermaid
flowchart LR
    Registry[Model Registry Service] --> DB[(Model Metadata)]
    Registry --> Artifacts[(Object Storage or OCI Registry)]
    Registry --> Eval[(Evaluation Results)]
    Registry --> Training[(Training Runs)]

    ModelService[Model Service] --> Registry
    ModelService --> Cache[Runtime Cache]
    Cache --> Runtime[Loaded Model]
```

Responsibilities:

| Component | Responsibility |
|---|---|
| Model Registry Service | lifecycle, status, metadata, promotion |
| Object Storage / OCI | binary artifacts |
| Postgres | metadata and lineage |
| ModelService | runtime loading and caching |
| Evaluation | readiness signals |
| Training | artifact production |
| CI/CD | validation and controlled promotion |

## Anti-patterns to Avoid

### Anti-pattern: model files committed to Git

Why it is bad:

- bloats repository
- slows developer workflow
- creates unclear rollback semantics

### Anti-pattern: model files baked into application image too early

Why it is bad:

- couples code deployment to model changes
- creates massive images
- makes multi-model support awkward

### Anti-pattern: Postgres as artifact store

Why it is bad:

- inefficient storage
- difficult backups
- poor streaming behavior
- wrong operational model

### Anti-pattern: adding MLflow before training exists

Why it is bad:

- extra service without clear need
- duplicates the initial `models` table
- increases onboarding complexity

### Anti-pattern: provider abstraction before second provider

Why it is bad:

- increases code complexity
- hides simple Hugging Face behavior
- creates abstractions without proven duplication

## Definition of Done for Initial Slice

Model source management is complete for the Initial Slice when:

- `models` table stores model metadata and revision.
- No model files are stored in Git.
- No model files are baked into the Docker image.
- Hugging Face cache is outside the repository.
- Docker Compose uses a named cache volume.
- Model seed file exists.
- `make model.seed` seeds local model metadata.
- `make model.list` shows registered models.
- `make model.smoke` can load and run a selected model.
- CI validates seed files without downloading large models.
- Inference records reference `models.id`.
- Service code does not hardcode specific model IDs.

## Future Decision Points

Revisit the model source strategy when one of these becomes true:

- training produces adapter artifacts
- model artifacts need promotion across environments
- multiple engineers are producing models
- benchmark results need to gate activation
- local cache misses are operationally expensive
- private model access becomes required
- model rollback becomes a recurring need
- model artifact lineage becomes hard to audit

Until then, keep the system simple:

```text
Postgres catalog
+ Hugging Face Hub
+ local runtime cache
```
