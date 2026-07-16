"""generation_preset: named, reusable, model-agnostic decoding presets

Revision ID: 0008_generation_preset
Revises: 0007_inference_generation_config
Create Date: 2026-07-15 00:00:00

A preset is a named bundle of decoding parameters (a GenerationConfig payload),
reusable across models. CREATE TABLE on a brand-new table takes no lock on
existing traffic, so the table and its partial unique index are created together
(the index is safe to build inline on an empty table).

The name is unique only among active presets: an archived preset keeps its row
and name for lineage, and the name becomes reusable for a new active preset. That
rule is the partial unique index uq_generation_preset_active_name.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0008_generation_preset"
down_revision: str | None = "0007_inference_generation_config"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Fail fast rather than queue behind a long-running query; the runner retries
    # on lock_timeout. CREATE TABLE on a new relation takes no lock on live traffic.
    op.execute("SET lock_timeout = '4s'")
    op.execute("SET statement_timeout = '1min'")
    op.create_table(
        "generation_preset",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=32), server_default="active", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("status IN ('active', 'archived')", name="valid_status"),
        sa.PrimaryKeyConstraint("id", name="pk_generation_preset"),
    )
    # Partial unique index: name unique among active presets only, reusable after
    # archive. Safe to build inline here because the table is empty at creation.
    op.create_index(
        "uq_generation_preset_active_name",
        "generation_preset",
        ["name"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '4s'")
    op.execute("SET statement_timeout = '1min'")
    op.drop_index("uq_generation_preset_active_name", table_name="generation_preset")
    op.drop_table("generation_preset")
