# arc-model-lab — developer task runner.
# Recipe lines MUST be TAB-indented (GNU Make requirement).

APP := arc_model_lab
sources := src

.DEFAULT_GOAL := help

.PHONY: help ## Show this help
help:
	@grep -E '^\.PHONY: [a-zA-Z0-9_-]+ ##' $(MAKEFILE_LIST) | sed -E 's/^\.PHONY: ([a-zA-Z0-9_-]+) ## (.*)/\1|\2/' | awk -F'|' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

.PHONY: prepare ## Sync the virtualenv with all dependency groups
prepare:
	uv sync --all-groups

.PHONY: lintable ## Apply auto-formatting and auto-linting
lintable: prepare
	uv run ruff format $(sources)
	uv run ruff check --fix $(sources)

.PHONY: lint ## Run linting checks
lint: prepare
	uv lock --check
	uv run ruff format --check $(sources)
	uv run ruff check $(sources)
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
