"""drop experiments.created_by

The experiment API no longer accepts a caller-supplied ``created_by`` label (it
was never authenticated). Drop the now-unused column.

Revision ID: 0005_drop_experiment_created_by
Revises: 0004_experiments
Create Date: 2026-07-05 00:00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005_drop_experiment_created_by"
down_revision: str | None = "0004_experiments"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Fail fast rather than queue behind a long-running query on `experiments`.
    op.execute("SET lock_timeout = '3s'")
    op.drop_column("experiments", "created_by")


def downgrade() -> None:
    op.execute("SET lock_timeout = '3s'")
    # Re-add as a nullable column: the label was always optional.
    op.add_column("experiments", sa.Column("created_by", sa.String(length=255), nullable=True))
