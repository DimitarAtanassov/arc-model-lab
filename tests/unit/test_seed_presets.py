from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from arc_model_lab.config import Settings
from arc_model_lab.db import seed_presets
from arc_model_lab.db.models import GenerationPresetRecord
from arc_model_lab.domain import InvalidGenerationConfigError, PresetStatus

_CAP = 2048


def _entry(name: str, **overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "name": name,
        "description": f"{name} preset",
        "config": {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "max_output_tokens": 256},
    }
    entry.update(overrides)
    return entry


def test_load_presets_parses_valid_entries(tmp_path: Path) -> None:
    path = tmp_path / "seed.json"
    path.write_text(
        json.dumps(
            [
                _entry("balanced"),
                {"name": "greedy", "config": {"do_sample": False, "max_output_tokens": 128}},
            ]
        )
    )

    presets = seed_presets.load_presets(path, max_output_tokens_cap=_CAP)

    assert [p.name for p in presets] == ["balanced", "greedy"]
    assert presets[0].description == "balanced preset"
    assert presets[0].config.temperature == pytest.approx(0.7)
    assert presets[0].config.max_output_tokens == 256
    assert presets[1].description is None
    assert presets[1].config.do_sample is False


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"not": "an array"}, "must be a JSON array"),
        (["not-an-object"], "must be a JSON object"),
        ([{"config": {"max_output_tokens": 64}}], "missing required field 'name'"),
        ([{"name": "m"}], "missing required field 'config'"),
        ([{"name": "m", "config": "not-an-object"}], "must be a JSON object"),
        (
            [_entry("dup"), _entry("dup", config={"do_sample": False, "max_output_tokens": 64})],
            "Duplicate preset names",
        ),
    ],
)
def test_load_presets_rejects_invalid_structure(tmp_path: Path, raw: object, match: str) -> None:
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match=match):
        seed_presets.load_presets(path, max_output_tokens_cap=_CAP)


@pytest.mark.parametrize(
    "config",
    [
        {"do_sample": True, "temperature": 99.0},  # out of range
        {"num_beams": 4, "do_sample": True},  # illegal beam+sampling combination
        {"max_output_tokens": _CAP + 1},  # over the server cap
    ],
)
def test_load_presets_rejects_invalid_config(tmp_path: Path, config: dict[str, object]) -> None:
    path = tmp_path / "seed.json"
    path.write_text(json.dumps([{"name": "bad", "config": config}]))

    with pytest.raises(InvalidGenerationConfigError):
        seed_presets.load_presets(path, max_output_tokens_cap=_CAP)


def test_main_check_validates_without_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(seed_presets, "get_settings", Settings)
    path = tmp_path / "seed.json"
    path.write_text(json.dumps([_entry("p1"), _entry("p2")]))

    seed_presets.main(["--check", str(path)])

    assert "Seed file OK: 2 preset(s)" in capsys.readouterr().out


def test_main_seeds_and_reports_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(seed_presets, "seed", AsyncMock(return_value=4))

    seed_presets.main([str(tmp_path / "seed.json")])

    assert "Seeded 4 preset(s)" in capsys.readouterr().out


@pytest.mark.integration
async def test_seed_writes_rows_to_database(
    engine: AsyncEngine, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    url = engine.url.render_as_string(hide_password=False)
    monkeypatch.setattr(seed_presets, "get_settings", lambda: Settings(database_url=url))
    path = tmp_path / "seed.json"
    path.write_text(json.dumps([_entry("p1"), _entry("p2")]))

    count = await seed_presets.seed(path)

    assert count == 2
    rows = (await db_session.execute(select(GenerationPresetRecord))).scalars().all()
    assert {row.name for row in rows} == {"p1", "p2"}
    assert all(row.status == PresetStatus.ACTIVE.value for row in rows)


@pytest.mark.integration
async def test_seed_is_idempotent_and_updates_existing(
    engine: AsyncEngine, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    url = engine.url.render_as_string(hide_password=False)
    monkeypatch.setattr(seed_presets, "get_settings", lambda: Settings(database_url=url))
    path = tmp_path / "seed.json"
    path.write_text(json.dumps([_entry("p1", description="first")]))
    await seed_presets.seed(path)

    path.write_text(json.dumps([_entry("p1", description="second")]))
    count = await seed_presets.seed(path)

    assert count == 1
    rows = (await db_session.execute(select(GenerationPresetRecord))).scalars().all()
    assert len(rows) == 1
    assert rows[0].description == "second"
