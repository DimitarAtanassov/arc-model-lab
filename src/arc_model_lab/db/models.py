from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from arc_model_lab.db.base import Base

_VALID_STATUSES = "status IN ('active', 'inactive', 'deprecated')"
_VALID_PRESET_STATUSES = "status IN ('active', 'archived')"


class ModelRecord(Base):
    __tablename__ = "models"
    __table_args__ = (
        UniqueConstraint("name"),
        CheckConstraint(_VALID_STATUSES, name="valid_status"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)  # noqa: A003 - primary key
    name: Mapped[str] = mapped_column(String(255))
    provider: Mapped[str] = mapped_column(String(255))
    model_id: Mapped[str] = mapped_column(String(255))
    tokenizer_id: Mapped[str] = mapped_column(String(255))
    revision: Mapped[str | None] = mapped_column(String(255), nullable=True)
    adapter_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    status: Mapped[str] = mapped_column(String(32), server_default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class InferenceRecord(Base):
    __tablename__ = "inference"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)  # noqa: A003 - primary key
    model_id: Mapped[UUID] = mapped_column(ForeignKey("models.id", ondelete="RESTRICT"))
    input_text: Mapped[str] = mapped_column(Text)
    prompt: Mapped[str] = mapped_column(Text)
    output_text: Mapped[str] = mapped_column(Text)
    latency_ms: Mapped[int] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # The resolved decoding config used for this inference. JSONB, not columns,
    # because the knob set is open and evolving; empty for pre-capture rows.
    generation_config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    # Lineage link to the preset that informed this row, if any. Nullable because a
    # row run from ad-hoc params or server defaults references no preset. ON DELETE
    # RESTRICT plus the archive soft-delete keeps the link valid without ever blocking
    # in practice. generation_config above, not this, is the reproducibility source.
    preset_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("generation_preset.id", ondelete="RESTRICT"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class GenerationPresetRecord(Base):
    __tablename__ = "generation_preset"
    __table_args__ = (
        CheckConstraint(_VALID_PRESET_STATUSES, name="valid_status"),
        # Name is unique only among active presets: an archived preset keeps its row
        # and name for lineage, and the name becomes reusable for a new active preset.
        Index(
            "uq_generation_preset_active_name",
            "name",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)  # noqa: A003 - primary key
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # A GenerationConfig.to_dict() payload; JSONB for the same reason the inference
    # row is: the knob set is open and evolving, so a new knob is a registry entry,
    # not a migration. Every read rehydrates through GenerationConfig.from_dict.
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    status: Mapped[str] = mapped_column(String(32), server_default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
