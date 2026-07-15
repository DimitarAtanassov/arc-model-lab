from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
