# arc-model-lab — developer task runner.
# Recipe lines MUST be TAB-indented (GNU Make requirement).

APP := arc_model_lab
sources := src
lint_paths := src tests

.DEFAULT_GOAL := help

.PHONY: help ## Show this help
help:
	@grep -E '^\.PHONY: [a-zA-Z0-9._-]+ ##' $(MAKEFILE_LIST) | sed -E 's/^\.PHONY: ([a-zA-Z0-9._-]+) ## (.*)/\1|\2/' | awk -F'|' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: prepare ## Sync the virtualenv with all dependency groups
prepare:
	uv sync --all-groups

.PHONY: lintable ## Apply auto-formatting and auto-linting
lintable: prepare
	uv run ruff format $(lint_paths)
	uv run ruff check --fix $(lint_paths)

.PHONY: lint ## Run linting checks
lint: prepare
	uv lock --check
	uv run ruff format --check $(lint_paths)
	uv run ruff check $(lint_paths)
	uv run mypy $(sources)

.PHONY: test ## Run tests and coverage reports
test: prepare
	uv run coverage run -m pytest
	uv run coverage report

.PHONY: migrate ## Apply database migrations to head (needs ARC_DATABASE_URL)
migrate: prepare
	uv run alembic upgrade head

.PHONY: migration ## Autogenerate a migration (NAME=description, needs ARC_DATABASE_URL)
migration: prepare
	uv run alembic revision --autogenerate -m "$(or $(NAME),change)"

.PHONY: downgrade ## Roll back the last migration (needs ARC_DATABASE_URL)
downgrade: prepare
	uv run alembic downgrade -1

.PHONY: run ## Run the app locally with auto-reload
run: prepare
	uv run uvicorn $(APP).main:app --reload --reload-dir src --host 0.0.0.0 --port 8000

.PHONY: model.seed ## Seed the model catalog from seeds/models.local.json
model.seed: prepare
	uv run python -m arc_model_lab.db.seed_models seeds/models.local.json

.PHONY: model.validate ## Validate the seed file without touching the database
model.validate: prepare
	uv run python -m arc_model_lab.db.seed_models --check seeds/models.local.json

.PHONY: model.list ## List catalog models
model.list: prepare
	uv run python -m arc_model_lab.cli.models list

.PHONY: model.get ## Show one model (NAME=...)
model.get: prepare
	uv run python -m arc_model_lab.cli.models get --name $(NAME)

.PHONY: model.activate ## Activate a model (NAME=...)
model.activate: prepare
	uv run python -m arc_model_lab.cli.models activate --name $(NAME)

.PHONY: model.deactivate ## Deactivate a model (NAME=...)
model.deactivate: prepare
	uv run python -m arc_model_lab.cli.models deactivate --name $(NAME)

.PHONY: model.smoke ## Load a model and run a summary (NAME=...)
model.smoke: prepare
	uv run python -m arc_model_lab.cli.models smoke --name $(NAME)

.PHONY: model.clear-cache ## Remove the local HuggingFace cache
model.clear-cache:
	rm -rf .cache/huggingface

.PHONY: eval.run ## Evaluate a single inference by id (ID=...)
eval.run: prepare
	uv run python -m arc_model_lab.cli.evaluations run --inference-id $(ID)

.PHONY: eval.replay ## Evaluate unevaluated inference rows (LIMIT=100)
eval.replay: prepare
	uv run python -m arc_model_lab.cli.evaluations replay --limit $(or $(LIMIT),100)

.PHONY: eval.backfill ## Backfill evaluations over a time range (SINCE=, UNTIL=, LIMIT=)
eval.backfill: prepare
	uv run python -m arc_model_lab.cli.evaluations backfill $(if $(SINCE),--since $(SINCE)) $(if $(UNTIL),--until $(UNTIL)) --limit $(or $(LIMIT),100)

.PHONY: eval.contract ## Run arc-eval contract tests (mocked, no live service)
eval.contract: prepare
	uv run pytest -m contract

.PHONY: eval.smoke ## Run the end-to-end eval smoke test (needs ARC_EVAL_SERVICE_URL + arc-eval)
eval.smoke: prepare
	uv run pytest -m eval_smoke

.PHONY: exp.create ## Create an experiment (NAME=, MODEL=)
exp.create: prepare
	uv run python -m arc_model_lab.cli.experiments create --name $(NAME) --model-name $(MODEL)

.PHONY: exp.run ## Run an experiment against sample input (ID=, TEXT=)
exp.run: prepare
	uv run python -m arc_model_lab.cli.experiments run --experiment-id $(ID) --input-text "$(TEXT)"

.PHONY: exp.compare ## Compare two experiments by id (ID=, OTHER=)
exp.compare: prepare
	uv run python -m arc_model_lab.cli.experiments compare --experiment-id $(ID) --other-id $(OTHER)
