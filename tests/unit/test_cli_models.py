"""Model catalog CLI: argparse dispatch (unit) and handler behavior (DB-backed).

Dispatch tests stub the handlers to prove wiring. Handler tests run against a
real Postgres; ``_smoke`` fakes ``ModelService`` so no weights are loaded.
"""

from __future__ import annotations

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from arc_model_lab.cli import models as cli
from arc_model_lab.config import Settings
from arc_model_lab.db.repositories import ModelRepository
from arc_model_lab.domain import Model, ModelStatus, Provider
from arc_model_lab.services.model_service import GenerationResult


class _FakeService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def generate(self, model: Model, messages: object) -> GenerationResult:
        return GenerationResult(prompt="p", output_text="SMOKE OUT", prompt_tokens=1, completion_tokens=1, latency_ms=1)


def _seed(
    session: Session, *, name: str, model_id: str = "org/model", status: ModelStatus = ModelStatus.ACTIVE
) -> None:
    ModelRepository(session).upsert(
        Model(name=name, provider=Provider.HUGGINGFACE, model_id=model_id, tokenizer_id=model_id, status=status)
    )
    session.commit()


@pytest.fixture
def _cli_db(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the CLI's own session factory at the test container."""
    url = engine.url.render_as_string(hide_password=False)
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(database_url=url))


def test_main_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


@pytest.mark.parametrize(
    ("argv", "handler", "expected_args"),
    [
        (["list"], "_list", ()),
        (["get", "--name", "m"], "_get", ("m",)),
        (["smoke", "--name", "m"], "_smoke", ("m",)),
    ],
)
def test_main_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], handler: str, expected_args: tuple[object, ...]
) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(cli, handler, lambda *args: calls.append(args))

    cli.main(argv)

    assert calls == [expected_args]


@pytest.mark.parametrize(
    ("command", "expected_status"),
    [("activate", ModelStatus.ACTIVE), ("deactivate", ModelStatus.INACTIVE)],
)
def test_main_activate_deactivate_sets_status(
    monkeypatch: pytest.MonkeyPatch, command: str, expected_status: ModelStatus
) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(cli, "_set_status", lambda *args: calls.append(args))

    cli.main([command, "--name", "m"])

    assert calls == [("m", expected_status)]


@pytest.mark.integration
@pytest.mark.usefixtures("_cli_db")
def test_list_prints_every_model(db_session: Session, capsys: pytest.CaptureFixture[str]) -> None:
    _seed(db_session, name="alpha")
    _seed(db_session, name="beta")

    cli._list()

    out = capsys.readouterr().out
    assert "alpha" in out
    assert "beta" in out


@pytest.mark.integration
@pytest.mark.usefixtures("_cli_db")
def test_get_prints_the_requested_model(db_session: Session, capsys: pytest.CaptureFixture[str]) -> None:
    _seed(db_session, name="alpha", model_id="a/b")

    cli._get("alpha")

    out = capsys.readouterr().out
    assert "alpha" in out
    assert "a/b" in out


@pytest.mark.integration
@pytest.mark.usefixtures("_cli_db", "db_session")
def test_get_exits_when_model_missing() -> None:
    with pytest.raises(SystemExit, match="Model not found"):
        cli._get("ghost")


@pytest.mark.integration
@pytest.mark.usefixtures("_cli_db")
def test_set_status_deactivates_and_prints(db_session: Session, capsys: pytest.CaptureFixture[str]) -> None:
    _seed(db_session, name="alpha", status=ModelStatus.ACTIVE)

    cli._set_status("alpha", ModelStatus.INACTIVE)

    out = capsys.readouterr().out
    assert "alpha" in out
    assert "inactive" in out


@pytest.mark.integration
@pytest.mark.usefixtures("_cli_db", "db_session")
def test_set_status_exits_when_model_missing() -> None:
    with pytest.raises(SystemExit, match="Model not found"):
        cli._set_status("ghost", ModelStatus.INACTIVE)


@pytest.mark.integration
@pytest.mark.usefixtures("_cli_db")
def test_smoke_prints_generated_summary(
    db_session: Session, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed(db_session, name="alpha")
    monkeypatch.setattr(cli, "ModelService", _FakeService)

    cli._smoke("alpha")

    assert "SMOKE OUT" in capsys.readouterr().out


@pytest.mark.integration
@pytest.mark.usefixtures("_cli_db", "db_session")
def test_smoke_exits_when_model_missing() -> None:
    with pytest.raises(SystemExit, match="Model not found"):
        cli._smoke("ghost")
