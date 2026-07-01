"""Seed loader: validation rules, the ``--check`` path, and a real DB write.

The loader is pure and stdlib-only, so validation is unit-tested with temp JSON
files. ``seed`` itself opens its own engine, so it is exercised once against a
real Postgres to prove rows are committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from arc_model_lab.config import Settings
from arc_model_lab.db import seed_models
from arc_model_lab.db.models import ModelRecord
from arc_model_lab.domain import ModelStatus, Provider


def _entry(name: str, **overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "name": name,
        "provider": "huggingface",
        "model_id": f"org/{name}",
        "tokenizer_id": f"org/{name}",
    }
    entry.update(overrides)
    return entry


def test_load_models_parses_valid_entries(tmp_path: Path) -> None:
    path = tmp_path / "seed.json"
    path.write_text(
        json.dumps(
            [
                _entry("full", revision="v1", adapter_path="/a", status="inactive"),
                _entry("minimal"),
            ]
        )
    )

    models = seed_models.load_models(path)

    assert [m.name for m in models] == ["full", "minimal"]
    assert models[0].provider is Provider.HUGGINGFACE
    assert models[0].revision == "v1"
    assert models[0].adapter_path == "/a"
    assert models[0].status is ModelStatus.INACTIVE
    assert models[1].revision is None
    assert models[1].adapter_path is None
    assert models[1].status is ModelStatus.ACTIVE


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"not": "an array"}, "must be a JSON array"),
        (["not-an-object"], "must be a JSON object"),
        ([{"name": "m", "provider": "huggingface", "model_id": "x"}], "missing required field"),
        ([{"name": "m", "provider": "nope", "model_id": "x", "tokenizer_id": "x"}], "not a valid Provider"),
        (
            [{"name": "m", "provider": "huggingface", "model_id": "x", "tokenizer_id": "x", "status": "nope"}],
            "not a valid ModelStatus",
        ),
        (
            [
                {"name": "dup", "provider": "huggingface", "model_id": "x", "tokenizer_id": "x"},
                {"name": "dup", "provider": "huggingface", "model_id": "y", "tokenizer_id": "y"},
            ],
            "Duplicate model names",
        ),
    ],
)
def test_load_models_rejects_invalid_seed(tmp_path: Path, raw: object, match: str) -> None:
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match=match):
        seed_models.load_models(path)


def test_main_check_validates_without_writing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "seed.json"
    path.write_text(json.dumps([_entry("m1"), _entry("m2")]))

    seed_models.main(["--check", str(path)])

    assert "Seed file OK: 2 model(s)" in capsys.readouterr().out


def test_main_seeds_and_reports_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(seed_models, "seed", lambda _path: 3)

    seed_models.main([str(tmp_path / "seed.json")])

    assert "Seeded 3 model(s)" in capsys.readouterr().out


@pytest.mark.integration
def test_seed_writes_rows_to_database(
    engine: Engine, db_session: Session, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    url = engine.url.render_as_string(hide_password=False)
    monkeypatch.setattr(seed_models, "get_settings", lambda: Settings(database_url=url))
    path = tmp_path / "seed.json"
    path.write_text(json.dumps([_entry("m1"), _entry("m2")]))

    count = seed_models.seed(path)

    assert count == 2
    rows = db_session.execute(select(ModelRecord)).scalars().all()
    assert {row.name for row in rows} == {"m1", "m2"}
