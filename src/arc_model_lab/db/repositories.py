from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.db.models import InferenceRecord, ModelRecord
from arc_model_lab.domain import (
    Inference,
    Model,
    ModelNotFoundError,
    ModelStatus,
    Provider,
)


class ModelRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_name(self, name: str) -> Model | None:
        record = await self._session.scalar(select(ModelRecord).where(ModelRecord.name == name))
        return _to_model(record) if record is not None else None

    async def get_by_id(self, model_id: UUID) -> Model | None:
        record = await self._session.get(ModelRecord, model_id)
        return _to_model(record) if record is not None else None

    async def require_by_name(self, name: str) -> Model:
        """Return the model with this name, or raise ModelNotFoundError (404)."""
        model = await self.get_by_name(name)
        if model is None:
            raise ModelNotFoundError(f"Model not found: {name}")
        return model

    async def require_by_id(self, model_id: UUID) -> Model:
        """Return the model with this id, or raise ModelNotFoundError (404)."""
        model = await self.get_by_id(model_id)
        if model is None:
            raise ModelNotFoundError(f"Model not found: {model_id}")
        return model

    async def add(self, model: Model) -> Model:
        self._session.add(_to_model_record(model))
        await self._session.flush()
        return model

    async def list_all(self) -> list[Model]:
        records = (await self._session.scalars(select(ModelRecord).order_by(ModelRecord.name))).all()
        return [_to_model(record) for record in records]

    async def upsert(self, model: Model) -> Model:
        record = await self._session.scalar(select(ModelRecord).where(ModelRecord.name == model.name))
        if record is None:
            return await self.add(model)
        record.provider = model.provider
        record.model_id = model.model_id
        record.tokenizer_id = model.tokenizer_id
        record.revision = model.revision
        record.adapter_path = model.adapter_path
        record.status = model.status
        await self._session.flush()
        await self._session.refresh(record)
        return _to_model(record)

    async def set_status(self, name: str, status: ModelStatus) -> Model | None:
        record = await self._session.scalar(select(ModelRecord).where(ModelRecord.name == name))
        if record is None:
            return None
        record.status = status
        await self._session.flush()
        await self._session.refresh(record)
        return _to_model(record)


class InferenceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, inference: Inference) -> Inference:
        self._session.add(_to_inference_record(inference))
        await self._session.flush()
        return inference

    async def get(self, inference_id: UUID) -> Inference | None:
        record = await self._session.get(InferenceRecord, inference_id)
        return _to_inference(record) if record is not None else None

    async def list_recent(self, limit: int) -> list[Inference]:
        """Return the most recent inferences, newest first (bounded page size)."""
        records = (
            await self._session.scalars(
                select(InferenceRecord).order_by(InferenceRecord.created_at.desc()).limit(limit)
            )
        ).all()
        return [_to_inference(record) for record in records]


def _to_model(record: ModelRecord) -> Model:
    return Model(
        id=record.id,
        name=record.name,
        provider=Provider(record.provider),
        model_id=record.model_id,
        tokenizer_id=record.tokenizer_id,
        revision=record.revision,
        adapter_path=record.adapter_path,
        status=ModelStatus(record.status),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _to_model_record(model: Model) -> ModelRecord:
    return ModelRecord(
        id=model.id,
        name=model.name,
        provider=model.provider,
        model_id=model.model_id,
        tokenizer_id=model.tokenizer_id,
        revision=model.revision,
        adapter_path=model.adapter_path,
        status=model.status,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _to_inference(record: InferenceRecord) -> Inference:
    return Inference(
        id=record.id,
        model_id=record.model_id,
        input_text=record.input_text,
        prompt=record.prompt,
        output_text=record.output_text,
        latency_ms=record.latency_ms,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        created_at=record.created_at,
    )


def _to_inference_record(inference: Inference) -> InferenceRecord:
    return InferenceRecord(
        id=inference.id,
        model_id=inference.model_id,
        input_text=inference.input_text,
        prompt=inference.prompt,
        output_text=inference.output_text,
        latency_ms=inference.latency_ms,
        prompt_tokens=inference.prompt_tokens,
        completion_tokens=inference.completion_tokens,
        created_at=inference.created_at,
    )
