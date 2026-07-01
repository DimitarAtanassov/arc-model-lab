"""SQLAlchemy ORM models. These mirror the domain entities for persistence."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from arc_model_lab.db.base import Base


class ModelRecord(Base):
    __tablename__ = "models"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)  # noqa: A003 - primary key
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    provider: Mapped[str] = mapped_column(String(255))
    model_id: Mapped[str] = mapped_column(String(255))
    tokenizer_id: Mapped[str] = mapped_column(String(255))
    adapter_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class InferenceRecord(Base):
    __tablename__ = "inferences"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)  # noqa: A003 - primary key
    model_id: Mapped[UUID] = mapped_column(
        ForeignKey("models.id", ondelete="RESTRICT"), index=True
    )
    input_text: Mapped[str] = mapped_column(Text)
    prompt: Mapped[str] = mapped_column(Text)
    output_text: Mapped[str] = mapped_column(Text)
    latency_ms: Mapped[int] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
