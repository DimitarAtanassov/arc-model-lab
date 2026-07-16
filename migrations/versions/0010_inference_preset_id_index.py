"""inference.preset_id index: build the FK index without blocking writes

Revision ID: 0010_inference_preset_id_index
Revises: 0009_inference_preset_id
Create Date: 2026-07-15 00:00:02

The index on `inference.preset_id` serves parent-delete performance (the FK's
ON DELETE RESTRICT check) and "inferences by preset" reads. It is a standalone
migration, separate from the column add (0009), because `inference` is trafficked:
a plain CREATE INDEX takes a SHARE lock that blocks all writes for the whole build,
so the index is built with CREATE INDEX CONCURRENTLY instead, which does not block
concurrent writes.

CREATE INDEX CONCURRENTLY cannot run inside a transaction, and Alembic wraps every
migration in one, so the build runs inside an autocommit block. Because the build
is outside a transaction a failure is not rolled back and can leave an INVALID
index; if_not_exists / if_exists plus the concurrent drop make the migration safe
to re-run after a partial failure.

lock_timeout is bounded so the brief metadata locks the build still needs fail fast
rather than queueing. statement_timeout is intentionally left unbounded for the
build itself: a concurrent index build on a large table can legitimately run long,
and capping it would abort a healthy build partway.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0010_inference_preset_id_index"
down_revision: str | None = "0009_inference_preset_id"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX_NAME = "ix_inference_preset_id"


def upgrade() -> None:
    # Bounded lock_timeout for the brief metadata locks the concurrent build takes;
    # the build proper runs outside a transaction so it cannot hold a long lock.
    op.execute("SET lock_timeout = '4s'")
    with op.get_context().autocommit_block():
        op.create_index(
            _INDEX_NAME,
            "inference",
            ["preset_id"],
            postgresql_concurrently=True,
            if_not_exists=True,
        )


def downgrade() -> None:
    op.execute("SET lock_timeout = '4s'")
    with op.get_context().autocommit_block():
        op.drop_index(
            _INDEX_NAME,
            table_name="inference",
            postgresql_concurrently=True,
            if_exists=True,
        )
