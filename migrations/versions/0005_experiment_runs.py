"""experiment_runs: decouple inference from experiments via an association table

Revision ID: 0005_experiment_runs
Revises: 0004_experiments
Create Date: 2026-07-05 00:00:00

The experiment->inference link moves off ``inference.experiment_id`` and into a
dedicated ``experiment_runs`` table, so an inference row never references an
experiment. Existing links are preserved before the column is dropped, and the
downgrade restores them, so the change is reversible without data loss.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005_experiment_runs"
down_revision: str | None = "0004_experiments"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Fail fast rather than queue behind a long-running query on `inference`.
    op.execute("SET lock_timeout = '3s'")

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

    # Preserve existing links: every inference already tagged with an experiment
    # becomes an experiment_runs row before the column is dropped. gen_random_uuid
    # is a Postgres 13+ core function (the deployment targets Postgres 16).
    op.execute(
        """
        INSERT INTO experiment_runs (id, experiment_id, inference_id, created_at)
        SELECT gen_random_uuid(), experiment_id, id, created_at
        FROM inference
        WHERE experiment_id IS NOT NULL
        """
    )

    op.drop_constraint("fk_inference_experiment_id_experiments", "inference", type_="foreignkey")
    op.drop_column("inference", "experiment_id")


def downgrade() -> None:
    op.execute("SET lock_timeout = '3s'")

    op.add_column("inference", sa.Column("experiment_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_inference_experiment_id_experiments",
        "inference",
        "experiments",
        ["experiment_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Restore the denormalized tag from the association before dropping it. The
    # unique inference_id in experiment_runs makes this mapping deterministic.
    op.execute(
        """
        UPDATE inference
        SET experiment_id = er.experiment_id
        FROM experiment_runs er
        WHERE er.inference_id = inference.id
        """
    )

    op.drop_index("ix_experiment_runs_experiment_id", table_name="experiment_runs")
    op.drop_table("experiment_runs")
