"""inference.preset_id: nullable lineage link to the preset that informed a row

Revision ID: 0009_inference_preset_id
Revises: 0008_generation_preset
Create Date: 2026-07-15 00:00:01

`inference` is the trafficked table, so this expand-only change is split into
non-blocking steps per the db-migrations discipline (spec 0001 §2.4, migration B):

1. ADD COLUMN preset_id uuid, nullable, no default. Adding a nullable column with
   no default is metadata-only in PostgreSQL 11+ (it writes no existing rows), so it
   takes only a brief lock and no table rewrite. Existing rows carry NULL, meaning
   "no preset informed this row"; a row run from ad-hoc params or server defaults
   also stays NULL.
2. ADD the foreign key as NOT VALID first, so the initial add does not scan the whole
   table under an ACCESS EXCLUSIVE lock, then VALIDATE CONSTRAINT in a separate
   statement, which takes only a SHARE UPDATE EXCLUSIVE lock and does not block reads
   or writes. ON DELETE RESTRICT keeps the lineage link valid; because presets are
   soft-deleted (archived, never physically removed once referenced), RESTRICT never
   blocks in practice.

The index on preset_id is deferred to a standalone migration (0010) built with
CREATE INDEX CONCURRENTLY, because no read needs it yet and building it here would
hold a lock for the whole build.

lock_timeout and statement_timeout are bounded so the migration fails fast and
retries rather than queueing behind a long transaction.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0009_inference_preset_id"
down_revision: str | None = "0008_generation_preset"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FK_NAME = "fk_inference_preset_id_generation_preset"


def upgrade() -> None:
    # Fail fast rather than queue behind a long-running query; the runner retries
    # on lock_timeout. Each statement below is metadata-only or a non-blocking scan.
    op.execute("SET lock_timeout = '4s'")
    op.execute("SET statement_timeout = '1min'")

    # 1. Metadata-only: nullable column, no default, so no existing row is rewritten.
    op.add_column("inference", sa.Column("preset_id", sa.Uuid(), nullable=True))

    # 2. Add the FK unvalidated so the ADD does not scan the table under an
    #    ACCESS EXCLUSIVE lock, then validate it separately under a lighter lock.
    op.execute(
        f"ALTER TABLE inference "
        f"ADD CONSTRAINT {_FK_NAME} "
        f"FOREIGN KEY (preset_id) REFERENCES generation_preset (id) "
        f"ON DELETE RESTRICT NOT VALID"
    )
    op.execute(f"ALTER TABLE inference VALIDATE CONSTRAINT {_FK_NAME}")


def downgrade() -> None:
    op.execute("SET lock_timeout = '4s'")
    op.execute("SET statement_timeout = '1min'")
    op.drop_constraint(_FK_NAME, "inference", type_="foreignkey")
    op.drop_column("inference", "preset_id")
