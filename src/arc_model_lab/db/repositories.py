from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.db.models import GenerationPresetRecord, InferenceRecord, ModelRecord
from arc_model_lab.domain import (
    GenerationConfig,
    GenerationPreset,
    Inference,
    Model,
    ModelNotFoundError,
    ModelStatus,
    PresetNameConflictError,
    PresetStatus,
    Provider,
)


class ModelRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_name(self, name: str) -> Model | None:
        record = await self._session.scalar(select(ModelRecord).where(ModelRecord.name == name))
        return _to_model(record) if record is not None else None

    async def require_by_name(self, name: str) -> Model:
        """Return the model with this name, or raise ModelNotFoundError (404)."""
        model = await self.get_by_name(name)
        if model is None:
            raise ModelNotFoundError(f"Model not found: {name}")
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


# Postgres unique_violation; the authority behind the duplicate-active-name 409.
_UNIQUE_VIOLATION = "23505"


class PresetRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, preset: GenerationPreset) -> GenerationPreset:
        """Persist a new preset, translating a duplicate active name to a domain 409.

        The partial unique index is the authority: catching its violation here means
        two concurrent creates of the same name both resolve to PresetNameConflictError
        rather than one racing to a 500 (any service-level pre-check is a fast path only).
        """
        self._session.add(_to_preset_record(preset))
        try:
            await self._session.flush()
        except IntegrityError as exc:
            if getattr(exc.orig, "sqlstate", None) == _UNIQUE_VIOLATION:
                raise PresetNameConflictError(f"An active preset already uses the name: {preset.name}") from exc
            raise
        return preset

    async def get(self, preset_id: UUID) -> GenerationPreset | None:
        record = await self._session.get(GenerationPresetRecord, preset_id)
        return _to_preset(record) if record is not None else None

    async def get_active_by_name(self, name: str) -> GenerationPreset | None:
        record = await self._session.scalar(
            select(GenerationPresetRecord).where(
                GenerationPresetRecord.name == name,
                GenerationPresetRecord.status == PresetStatus.ACTIVE.value,
            )
        )
        return _to_preset(record) if record is not None else None

    async def list_active(self) -> list[GenerationPreset]:
        """Return active presets, newest first (archived presets stay hidden)."""
        records = (
            await self._session.scalars(
                select(GenerationPresetRecord)
                .where(GenerationPresetRecord.status == PresetStatus.ACTIVE.value)
                .order_by(GenerationPresetRecord.created_at.desc())
            )
        ).all()
        return [_to_preset(record) for record in records]

    async def update(self, preset: GenerationPreset) -> GenerationPreset:
        """Persist edits to description, config, and/or status on an existing preset."""
        record = await self._session.get(GenerationPresetRecord, preset.id)
        if record is None:
            return preset
        record.description = preset.description
        record.config = preset.config.to_dict()
        record.status = preset.status.value
        await self._session.flush()
        await self._session.refresh(record)
        return _to_preset(record)


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
        generation_config=GenerationConfig.from_dict(record.generation_config),
        preset_id=record.preset_id,
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
        generation_config=inference.generation_config.to_dict(),
        preset_id=inference.preset_id,
        created_at=inference.created_at,
    )


def _to_preset(record: GenerationPresetRecord) -> GenerationPreset:
    return GenerationPreset(
        id=record.id,
        name=record.name,
        description=record.description,
        config=GenerationConfig.from_dict(record.config),
        status=PresetStatus(record.status),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _to_preset_record(preset: GenerationPreset) -> GenerationPresetRecord:
    return GenerationPresetRecord(
        id=preset.id,
        name=preset.name,
        description=preset.description,
        config=preset.config.to_dict(),
        status=preset.status.value,
        created_at=preset.created_at,
        updated_at=preset.updated_at,
    )
