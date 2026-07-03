"""experiments: named run configurations, and inference.experiment_id

Revision ID: 0004_experiments
Revises: 0003_evaluation_results
Create Date: 2026-07-02 00:00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0004_experiments"
down_revision: str | None = "0003_evaluation_results"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Fail fast rather than queue behind a long-running query on `inference`.
    op.execute("SET lock_timeout = '3s'")

    op.create_table(
        "experiments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("model_id", sa.Uuid(), nullable=False),
        sa.Column("prompt_version_id", sa.Uuid(), nullable=True),
        sa.Column(
            "generation_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_experiments"),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["models.id"],
            name="fk_experiments_model_id_models",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("name", name="uq_experiments_name"),
    )
    # New empty table: a plain index build is safe (no traffic, no lock contention).
    op.create_index("ix_experiments_model_id", "experiments", ["model_id"])

    # Adding a NULLable column is instant in PG 11+ (no table rewrite). The FK
    # validates instantly because every existing row is NULL.
    op.add_column("inference", sa.Column("experiment_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_inference_experiment_id_experiments",
        "inference",
        "experiments",
        ["experiment_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # `inference` is a live table, so index it CONCURRENTLY (outside the migration
    # transaction) to avoid blocking writes.
    with op.get_context().autocommit_block():
        op.create_index(
            "ix_inference_experiment_id",
            "inference",
            ["experiment_id"],
            postgresql_concurrently=True,
            if_not_exists=True,
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.drop_index(
            "ix_inference_experiment_id",
            table_name="inference",
            postgresql_concurrently=True,
            if_exists=True,
        )
    op.drop_constraint("fk_inference_experiment_id_experiments", "inference", type_="foreignkey")
    op.drop_column("inference", "experiment_id")

    op.drop_index("ix_experiments_model_id", table_name="experiments")
    op.drop_table("experiments")
