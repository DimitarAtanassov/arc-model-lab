"""evaluation results: one metric score per inference

Revision ID: 0003_evaluation_results
Revises: 0002_model_catalog_fields
Create Date: 2026-07-01 00:00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_evaluation_results"
down_revision: str | None = "0002_model_catalog_fields"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
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
    # Non-concurrent index creation is safe here: the table is created empty in
    # this same transaction and is not yet serving traffic, so there is no lock
    # contention to avoid. CONCURRENTLY is reserved for indexing existing,
    # populated production tables.
    op.create_index("ix_evaluation_results_inference_id", "evaluation_results", ["inference_id"])
    op.create_index("ix_evaluation_results_metric_name", "evaluation_results", ["metric_name"])
    op.create_index("ix_evaluation_results_created_at", "evaluation_results", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_evaluation_results_created_at", table_name="evaluation_results")
    op.drop_index("ix_evaluation_results_metric_name", table_name="evaluation_results")
    op.drop_index("ix_evaluation_results_inference_id", table_name="evaluation_results")
    op.drop_table("evaluation_results")
