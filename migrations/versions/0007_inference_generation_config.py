"""inference generation_config: capture the decoding config per row

Revision ID: 0007_inference_generation_config
Revises: 0006_drop_experiments_eval
Create Date: 2026-07-14 00:00:00

Records the resolved decoding config (temperature, max_output_tokens, and any
future knob) each inference row was produced with, so a row reproduces the call
that made it. Stored as JSONB, not columns, because the knob set is open and
evolving.

Adding a column with a constant default is metadata-only in PostgreSQL 11+ (it
writes no existing rows), so this takes a brief lock and no table rewrite.
Existing rows carry '{}' meaning "pre-capture"; every new row writes the config.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0007_inference_generation_config"
down_revision: str | None = "0006_drop_experiments_eval"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Fail fast rather than queue behind a long-running query; the runner retries
    # on lock_timeout. The add itself is metadata-only, so the lock is brief.
    op.execute("SET lock_timeout = '4s'")
    op.execute("SET statement_timeout = '1min'")
    op.add_column(
        "inference",
        sa.Column(
            "generation_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '4s'")
    op.execute("SET statement_timeout = '1min'")
    op.drop_column("inference", "generation_config")
