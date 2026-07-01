"""model catalog fields: revision, status, updated_at

Revision ID: 0002_model_catalog_fields
Revises: 0001_initial
Create Date: 2026-06-30 00:00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_model_catalog_fields"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("models", sa.Column("revision", sa.String(length=255), nullable=True))
    op.add_column(
        "models",
        sa.Column("status", sa.String(length=32), server_default="active", nullable=False),
    )
    op.add_column(
        "models",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_check_constraint(
        "ck_models_valid_status",
        "models",
        "status IN ('active', 'inactive', 'deprecated')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_models_valid_status", "models", type_="check")
    op.drop_column("models", "updated_at")
    op.drop_column("models", "status")
    op.drop_column("models", "revision")
