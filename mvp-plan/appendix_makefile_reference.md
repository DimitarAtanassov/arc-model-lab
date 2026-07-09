# Appendix - Makefile Reference

## Purpose

The Makefile is the canonical developer interface for `arc-model-lab`.

Engineers should not need to remember raw commands for `uv`, `ruff`, `pytest`, `alembic`, `docker`, dataset export, training, or observability.

## Base Targets (implemented)

These match the committed Makefile. Postgres runs via `docker compose`, not a Make target.

```make
prepare:      # uv sync --all-groups
lintable:     # ruff format + ruff check --fix
lint:         # uv lock --check + ruff format --check + ruff check + mypy src
test:         # coverage run -m pytest + coverage report
migrate:      # alembic upgrade head
migration:    # alembic revision --autogenerate -m "$(NAME)"
downgrade:    # alembic downgrade -1
run:          # uvicorn arc_model_lab.main:app --reload (port 8000)
```

Database lifecycle uses compose directly:

```bash
docker compose up -d postgres     # start Postgres
docker compose up --build         # full stack (migrate + seed + serve)
```

## Phase Targets

Implemented today.

Evaluation (phase 01):

```make
eval.run
eval.replay
eval.backfill
eval.contract
eval.smoke
```

Model catalog (phase 08):

```make
model.seed
model.validate
model.list
model.get
model.activate
model.deactivate
model.smoke
model.clear-cache
```

Planned by phase. Add these as each phase lands, following the `<area>.<verb>` convention.

```text
Experiments (02)    : exp.create, exp.run, exp.compare, exp.smoke
Prompts (03)        : prompt.create, prompt.version, prompt.activate, prompt.rollback, prompt.render
Datasets (04)       : data.from-inference, data.from-exp, data.validate, data.export
Training (05)       : train.run, train.smoke, train.evaluate, train.validate
Model Registry (06) : model.register, model.validate, model.deprecate   (extends model.*)
OpenTelemetry (07)  : otel.up, otel.down, otel.smoke, otel.disabled-test
```

## Principle

If a task is repeated more than twice, it should become a Make target.
