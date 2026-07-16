from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.db.repositories import PresetRepository
from arc_model_lab.domain import (
    GenerationConfig,
    GenerationPreset,
    PresetNameConflictError,
    PresetStatus,
)

pytestmark = pytest.mark.integration


def _preset(name: str, *, temperature: float = 0.7) -> GenerationPreset:
    return GenerationPreset(
        name=name,
        description=f"{name} preset",
        config=GenerationConfig(do_sample=True, temperature=temperature, max_output_tokens=256),
    )


async def test_add_and_get_round_trip(db_session: AsyncSession) -> None:
    repo = PresetRepository(db_session)
    preset = _preset("balanced")
    await repo.add(preset)

    fetched = await repo.get(preset.id)

    assert fetched is not None
    assert fetched.name == "balanced"
    assert fetched.config.temperature == 0.7
    assert fetched.status is PresetStatus.ACTIVE


async def test_get_returns_none_when_absent(db_session: AsyncSession) -> None:
    assert await PresetRepository(db_session).get(uuid4()) is None


async def test_list_active_orders_newest_first_and_hides_archived(db_session: AsyncSession) -> None:
    repo = PresetRepository(db_session)
    first = await repo.add(_preset("first"))
    second = await repo.add(_preset("second"))
    await db_session.commit()

    # Archive the first; it must drop out of the active listing.
    await repo.update(replace(first, status=PresetStatus.ARCHIVED))
    await db_session.commit()

    active = await repo.list_active()

    assert [preset.id for preset in active] == [second.id]
    assert first.id not in {preset.id for preset in active}


async def test_duplicate_active_name_raises_conflict(db_session: AsyncSession) -> None:
    # The partial unique index is the authority: a second active preset with the
    # same name is a PresetNameConflictError (mapped to 409), not a 500.
    repo = PresetRepository(db_session)
    await repo.add(_preset("dup"))

    with pytest.raises(PresetNameConflictError):
        await repo.add(_preset("dup"))


async def test_name_is_reusable_after_archive(db_session: AsyncSession) -> None:
    # An archived preset keeps its row and name for lineage, and the name becomes
    # reusable for a new active preset (partial-unique-among-active behavior).
    repo = PresetRepository(db_session)
    original = await repo.add(_preset("reusable", temperature=0.5))
    await repo.update(replace(original, status=PresetStatus.ARCHIVED))
    await db_session.commit()

    revived = await repo.add(_preset("reusable", temperature=1.2))
    await db_session.commit()

    assert revived.id != original.id
    active = await repo.list_active()
    assert [preset.id for preset in active] == [revived.id]


async def test_update_persists_config_and_description(db_session: AsyncSession) -> None:
    repo = PresetRepository(db_session)
    preset = await repo.add(_preset("editable"))
    await db_session.commit()

    edited = replace(
        preset,
        description="edited note",
        config=GenerationConfig(do_sample=True, temperature=1.5, max_output_tokens=512),
    )
    updated = await repo.update(edited)

    assert updated.description == "edited note"
    assert updated.config.temperature == 1.5
    assert updated.config.max_output_tokens == 512
