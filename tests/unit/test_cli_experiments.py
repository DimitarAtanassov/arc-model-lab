"""Experiment CLI: argparse dispatch (unit) and handler behavior (DB-backed).

Dispatch tests stub the handlers to prove wiring and config assembly. Handler
tests run against a real Postgres; ``create``/``compare`` never load a model.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from arc_model_lab.cli import experiments as cli
from arc_model_lab.config import Settings
from arc_model_lab.db.repositories import ExperimentRepository, ModelRepository
from arc_model_lab.domain import (
    EvaluationOutcome,
    EvaluationResult,
    EvaluationStatus,
    Experiment,
    ExperimentMetricAggregate,
    GenerationConfig,
    Inference,
    Model,
    Provider,
)
from arc_model_lab.services.experiment_service import ExperimentRunResult

_ID = UUID("11111111-1111-1111-1111-111111111111")
_OTHER = UUID("22222222-2222-2222-2222-222222222222")
_CONFIG = GenerationConfig(temperature=0.0, max_output_tokens=256)


async def _seed_model(session: AsyncSession, name: str = "base") -> Model:
    model = await ModelRepository(session).upsert(
        Model(name=name, provider=Provider.HUGGINGFACE, model_id="org/model", tokenizer_id="org/model")
    )
    await session.commit()
    return model


@pytest.fixture
def _cli_db(engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the CLI's own session factory at the test container."""
    url = engine.url.render_as_string(hide_password=False)
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(database_url=url))


def test_main_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_main_create_assembles_config_from_args(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    async def _capture(*args: object) -> None:
        calls.append(args)

    monkeypatch.setattr(cli, "_create", _capture)

    cli.main(["create", "--name", "e", "--model-name", "m", "--temperature", "0.7"])

    assert calls == [("e", "m", GenerationConfig(temperature=0.7, max_output_tokens=256))]


@pytest.mark.parametrize(
    ("argv", "handler", "expected_args"),
    [
        (["run", "--experiment-id", str(_ID), "--input-text", "hi"], "_run", (_ID, "hi", None)),
        (["compare", "--experiment-id", str(_ID), "--other-id", str(_OTHER)], "_compare", (_ID, _OTHER)),
    ],
)
def test_main_dispatches_to_handler(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], handler: str, expected_args: tuple[object, ...]
) -> None:
    calls: list[tuple[object, ...]] = []

    async def _capture(*args: object) -> None:
        calls.append(args)

    monkeypatch.setattr(cli, handler, _capture)

    cli.main(argv)

    assert calls == [expected_args]


@pytest.mark.integration
@pytest.mark.usefixtures("_cli_db")
async def test_create_persists_and_prints(db_session: AsyncSession, capsys: pytest.CaptureFixture[str]) -> None:
    await _seed_model(db_session)

    await cli._create("exp-a", "base", _CONFIG)

    assert "exp-a" in capsys.readouterr().out


@pytest.mark.integration
@pytest.mark.usefixtures("_cli_db")
async def test_create_rejects_duplicate_name(db_session: AsyncSession) -> None:
    await _seed_model(db_session)
    await cli._create("dup", "base", _CONFIG)

    with pytest.raises(SystemExit, match="already exists"):
        await cli._create("dup", "base", _CONFIG)


@pytest.mark.integration
@pytest.mark.usefixtures("_cli_db", "db_session")
async def test_create_exits_when_model_missing() -> None:
    with pytest.raises(SystemExit, match="Model not found"):
        await cli._create("x", "ghost", _CONFIG)


@pytest.mark.integration
@pytest.mark.usefixtures("_cli_db")
async def test_compare_prints_a_line_per_experiment(
    db_session: AsyncSession, capsys: pytest.CaptureFixture[str]
) -> None:
    model = await _seed_model(db_session)
    repo = ExperimentRepository(db_session)
    first = await repo.add(Experiment(name="a", model_id=model.id, generation_config=_CONFIG))
    second = await repo.add(Experiment(name="b", model_id=model.id, generation_config=_CONFIG))
    await db_session.commit()

    await cli._compare(first.id, second.id)

    out = capsys.readouterr().out
    assert str(first.id) in out
    assert str(second.id) in out


def test_print_aggregates_prints_metric_rows(capsys: pytest.CaptureFixture[str]) -> None:
    aggregates = [ExperimentMetricAggregate(metric_name="faithfulness", average_score=0.9, evaluated_count=3)]

    cli._print_aggregates(_ID, aggregates)

    assert capsys.readouterr().out == f"{_ID}\tfaithfulness\t0.900\t3\n"


async def test_run_prints_scores_when_evaluation_present(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inference = Inference(
        model_id=_ID,
        input_text="input",
        prompt="prompt",
        output_text="summary",
        latency_ms=5,
    )
    scored = ExperimentRunResult(
        inference=inference,
        evaluation=EvaluationOutcome(
            status=EvaluationStatus.COMPLETED,
            results=(
                EvaluationResult(
                    inference_id=inference.id,
                    metric_name="faithfulness",
                    score=0.91,
                    evaluator_name="eval",
                ),
                EvaluationResult(
                    inference_id=inference.id,
                    metric_name="relevance",
                    score=0.75,
                    evaluator_name="eval",
                ),
            ),
        ),
    )

    class _Service:
        async def run(self, *args: object, **kwargs: object) -> ExperimentRunResult:
            return scored

    monkeypatch.setattr(cli, "_experiment_service", _Service)
    monkeypatch.setattr(cli, "_in_session", lambda op: op(None))

    await cli._run(_ID, "hello", ["faithfulness", "relevance"])

    assert capsys.readouterr().out == f"{inference.id}\tsummary\tfaithfulness=0.910, relevance=0.750\n"


async def test_run_prints_dash_when_evaluation_empty(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inference = Inference(
        model_id=_ID,
        input_text="input",
        prompt="prompt",
        output_text="summary",
        latency_ms=5,
    )
    no_scores = ExperimentRunResult(
        inference=inference,
        evaluation=EvaluationOutcome(status=EvaluationStatus.COMPLETED, results=()),
    )

    class _Service:
        async def run(self, *args: object, **kwargs: object) -> ExperimentRunResult:
            return no_scores

    monkeypatch.setattr(cli, "_experiment_service", _Service)
    monkeypatch.setattr(cli, "_in_session", lambda op: op(None))

    await cli._run(_ID, "hello", ["faithfulness"])

    assert capsys.readouterr().out == f"{inference.id}\tsummary\t-\n"


async def test_run_prints_dash_when_evaluation_absent(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inference = Inference(
        model_id=_ID,
        input_text="input",
        prompt="prompt",
        output_text="summary",
        latency_ms=5,
    )
    result_without_evaluation = ExperimentRunResult(inference=inference, evaluation=None)

    class _Service:
        async def run(self, *args: object, **kwargs: object) -> ExperimentRunResult:
            return result_without_evaluation

    monkeypatch.setattr(cli, "_experiment_service", _Service)
    monkeypatch.setattr(cli, "_in_session", lambda op: op(None))

    await cli._run(_ID, "hello", ["faithfulness"])

    assert capsys.readouterr().out == f"{inference.id}\tsummary\t-\n"
