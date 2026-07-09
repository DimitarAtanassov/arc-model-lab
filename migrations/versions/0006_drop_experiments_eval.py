"""drop experiments, experiment_runs, evaluation_results

Revision ID: 0006_drop_experiments_eval
Revises: 0005_experiment_runs
Create Date: 2026-07-08 00:00:00

Experimentation and evaluation move to arc-eval-service; arc-model-lab is pure
model serving. This drops the three tables it no longer owns: experiment_runs,
experiments, and evaluation_results.

Rollout ordering (a contract step, run it deliberately):
- Deploy the code that stops reading these tables BEFORE running this migration.
  During a rolling deploy an old instance still serving the removed /experiments
  and /inference/{id}/evaluate routes would hit dropped tables and error, so run
  this as a discrete post-deploy step, never on app boot.
- This is destructive and not data-reversible. downgrade() recreates the tables
  empty for schema reversibility only; the rows are NOT restored. Export anything
  worth keeping first (arc-eval-service owns the authoritative copy).

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0006_drop_experiments_eval"
down_revision: str | None = "0005_experiment_runs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Fail fast rather than queue behind a long-running query and block every
    # request that arrives after us; the runner retries on lock_timeout.
    op.execute("SET lock_timeout = '4s'")
    op.execute("SET statement_timeout = '1min'")
    # DROP TABLE removes each table's own indexes, so they are not dropped
    # explicitly. Drop children before parents to respect the foreign keys.
    op.drop_table("experiment_runs")
    op.drop_table("experiments")
    op.drop_table("evaluation_results")


def downgrade() -> None:
    op.execute("SET lock_timeout = '4s'")
    op.execute("SET statement_timeout = '1min'")
    op.create_table(
        "evaluation_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("inference_id", sa.Uuid(), nullable=False),
        sa.Column("metric_name", sa.Text(), nullable=False),
        sa.Column("score", sa.Double(), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("evaluator_name", sa.Text(), nullable=False),
        sa.Column("evaluator_version", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_evaluation_results"),
        sa.ForeignKeyConstraint(
            ["inference_id"],
            ["inference.id"],
            name="fk_evaluation_results_inference_id_inference",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "inference_id",
            "metric_name",
            "evaluator_name",
            name="uq_evaluation_results_inference_metric_evaluator",
        ),
    )
    op.create_index("ix_evaluation_results_inference_id", "evaluation_results", ["inference_id"])
    op.create_index("ix_evaluation_results_metric_name", "evaluation_results", ["metric_name"])
    op.create_index("ix_evaluation_results_created_at", "evaluation_results", ["created_at"])

    op.create_table(
        "experiments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("model_id", sa.Uuid(), nullable=False),
        sa.Column("generation_config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_experiments"),
        sa.ForeignKeyConstraint(
            ["model_id"], ["models.id"], name="fk_experiments_model_id_models", ondelete="RESTRICT"
        ),
        sa.UniqueConstraint("name", name="uq_experiments_name"),
    )

    op.create_table(
        "experiment_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("experiment_id", sa.Uuid(), nullable=False),
        sa.Column("inference_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_experiment_runs"),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.id"],
            name="fk_experiment_runs_experiment_id_experiments",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["inference_id"],
            ["inference.id"],
            name="fk_experiment_runs_inference_id_inference",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("inference_id", name="uq_experiment_runs_inference_id"),
    )
    op.create_index("ix_experiment_runs_experiment_id", "experiment_runs", ["experiment_id"])
