"""SQLAlchemy ORM models. These mirror the domain entities for persistence."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvaluationResultRecord(Base):
    """One metric score for one inference, produced by the arc-eval service.

    One metric per row (not a JSON blob) so scores stay queryable and indexable.
    The unique key ``(inference_id, metric_name, evaluator_name)`` makes replay
    and backfill idempotent: re-evaluating an inference upserts rather than
    duplicating.
    """

    __tablename__ = "evaluation_results"
    __table_args__ = (
        UniqueConstraint(
            "inference_id",
            "metric_name",
            "evaluator_name",
            name="uq_evaluation_results_inference_metric_evaluator",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)  # noqa: A003 - primary key
    inference_id: Mapped[UUID] = mapped_column(ForeignKey("inference.id", ondelete="CASCADE"), index=True)
    metric_name: Mapped[str] = mapped_column(Text, index=True)
    score: Mapped[float] = mapped_column(Double)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    evaluator_name: Mapped[str] = mapped_column(Text)
    evaluator_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
