from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import pytest

from arc_model_lab.cli import evaluations as cli
from arc_model_lab.domain import EvaluationStatus

_ID_A = UUID("123e4567-e89b-12d3-a456-426614174000")
_ID_B = UUID("123e4567-e89b-12d3-a456-426614174001")


class _AsyncSessionContext:
    """A minimal async context manager standing in for a session scope."""

    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *args: object) -> bool:
        return False


class _FakeSessionFactory:
    def __call__(self) -> _AsyncSessionContext:
        return _AsyncSessionContext()


def test_parse_timestamp_defaults_to_utc() -> None:
    parsed = cli._parse_timestamp("2026-01-02T03:04:05")

    assert parsed == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def test_parse_timestamp_preserves_timezone() -> None:
    parsed = cli._parse_timestamp("2026-01-02T03:04:05+01:00")

    assert parsed == datetime.fromisoformat("2026-01-02T03:04:05+01:00")


def test_session_factory_builds_session_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = object()
    factory = object()

    def _settings() -> object:
        return type("Settings", (), {"database_url": "postgresql://example/test"})()

    monkeypatch.setattr(cli, "get_settings", _settings)
    monkeypatch.setattr(cli, "create_async_engine_from_url", lambda url: engine)
    monkeypatch.setattr(cli, "create_async_session_factory", lambda created_engine: factory)

    assert cli._session_factory() is factory


def test_evaluation_service_exits_when_client_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "build_arc_eval_client", lambda settings: None)

    with pytest.raises(SystemExit, match="ARC_EVAL_SERVICE_URL"):
        cli._evaluation_service()


def test_evaluation_service_builds_service_when_client_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    client = object()
    monkeypatch.setattr(cli, "build_arc_eval_client", lambda settings: client)

    service = cli._evaluation_service()

    assert service._client is client


def test_main_dispatches_run_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[UUID, list[str]]] = []

    async def _capture_run(inference_id: UUID, metrics: list[str]) -> None:
        calls.append((inference_id, metrics))

    monkeypatch.setattr(cli, "_run", _capture_run)

    cli.main(["run", "--inference-id", "123e4567-e89b-12d3-a456-426614174000"])

    assert calls == [(_ID_A, ["faithfulness", "answer_relevance"])]


def test_main_dispatches_replay_and_backfill(monkeypatch: pytest.MonkeyPatch) -> None:
    replay_calls: list[tuple[object, object, int]] = []
    backfill_calls: list[tuple[object, object, int]] = []

    async def _capture_replay(*, created_after: object, created_before: object, limit: int, metrics: list[str]) -> None:
        replay_calls.append((created_after, created_before, limit))

    monkeypatch.setattr(cli, "_process_batch", _capture_replay)

    cli.main(["replay", "--limit", "7"])
    assert replay_calls == [(None, None, 7)]

    async def _capture_backfill(
        *, created_after: object, created_before: object, limit: int, metrics: list[str]
    ) -> None:
        backfill_calls.append((created_after, created_before, limit))

    monkeypatch.setattr(cli, "_process_batch", _capture_backfill)

    cli.main(["backfill", "--since", "2026-01-02T03:04:05", "--until", "2026-01-03T03:04:05", "--limit", "9"])

    assert backfill_calls[0][0] == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert backfill_calls[0][1] == datetime(2026, 1, 3, 3, 4, 5, tzinfo=UTC)
    assert backfill_calls[0][2] == 9


async def test_run_exits_when_inference_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeInferenceRepository:
        def __init__(self, session: object) -> None:
            self.session = session

        async def get(self, inference_id: UUID) -> object | None:
            return None

    class _FakeService:
        async def evaluate_inference(self, session: object, inference: object, metrics: object) -> object:
            raise AssertionError("should not be called")

    monkeypatch.setattr(cli, "_evaluation_service", _FakeService)
    monkeypatch.setattr(cli, "_session_factory", _FakeSessionFactory)
    monkeypatch.setattr(cli, "InferenceRepository", _FakeInferenceRepository)

    with pytest.raises(SystemExit, match="Inference not found"):
        await cli._run(_ID_A, ["faithfulness"])


async def test_run_prints_outcome_and_exits_on_failed_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _FakeInferenceRepository:
        def __init__(self, session: object) -> None:
            self.session = session

        async def get(self, inference_id: UUID) -> object:
            return SimpleNamespace(id=inference_id)

    class _FakeService:
        async def evaluate_inference(self, session: object, inference: object, metrics: object) -> object:
            return SimpleNamespace(status=EvaluationStatus.FAILED, results=())

    monkeypatch.setattr(cli, "_evaluation_service", _FakeService)
    monkeypatch.setattr(cli, "_session_factory", _FakeSessionFactory)
    monkeypatch.setattr(cli, "InferenceRepository", _FakeInferenceRepository)

    with pytest.raises(SystemExit, match="1"):
        await cli._run(_ID_A, ["faithfulness"])

    out = capsys.readouterr().out
    assert str(_ID_A) in out
    assert "failed" in out


async def test_run_prints_outcome_without_exiting_for_completed_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _FakeInferenceRepository:
        def __init__(self, session: object) -> None:
            self.session = session

        async def get(self, inference_id: UUID) -> object:
            return SimpleNamespace(id=inference_id)

    class _FakeService:
        async def evaluate_inference(self, session: object, inference: object, metrics: object) -> object:
            return SimpleNamespace(status=EvaluationStatus.COMPLETED, results=())

    monkeypatch.setattr(cli, "_evaluation_service", _FakeService)
    monkeypatch.setattr(cli, "_session_factory", _FakeSessionFactory)
    monkeypatch.setattr(cli, "InferenceRepository", _FakeInferenceRepository)

    await cli._run(_ID_A, ["faithfulness"])

    out = capsys.readouterr().out
    assert str(_ID_A) in out
    assert "completed" in out


class _BatchService:
    """Fake evaluation service exposing the score/persist seam the batch uses.

    Scoring status is COMPLETED except for _ID_B, which fails; persistence
    passes the scored status straight through.
    """

    async def score(self, inference: object, metrics: object) -> object:
        status = EvaluationStatus.FAILED if inference.id == _ID_B else EvaluationStatus.COMPLETED
        return SimpleNamespace(inference=inference, status=status)

    async def persist(self, session: object, scored: object) -> object:
        return SimpleNamespace(status=scored.status, results=())


async def test_process_batch_completes_without_exiting_when_all_succeed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _FakeInferenceRepository:
        def __init__(self, session: object) -> None:
            self.session = session

        async def list_unevaluated(self, *, limit: int, created_after: object, created_before: object) -> list[object]:
            return [SimpleNamespace(id=_ID_A)]

    monkeypatch.setattr(cli, "_evaluation_service", _BatchService)
    monkeypatch.setattr(cli, "_session_factory", _FakeSessionFactory)
    monkeypatch.setattr(cli, "InferenceRepository", _FakeInferenceRepository)

    await cli._process_batch(created_after=None, created_before=None, limit=5, metrics=["faithfulness"])

    out = capsys.readouterr().out
    assert "processed=1" in out
    assert "completed=1" in out
    assert "failed=0" in out


async def test_process_batch_reports_completed_and_failed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _FakeInferenceRepository:
        def __init__(self, session: object) -> None:
            self.session = session

        async def list_unevaluated(self, *, limit: int, created_after: object, created_before: object) -> list[object]:
            return [SimpleNamespace(id=_ID_A), SimpleNamespace(id=_ID_B)]

    monkeypatch.setattr(cli, "_evaluation_service", _BatchService)
    monkeypatch.setattr(cli, "_session_factory", _FakeSessionFactory)
    monkeypatch.setattr(cli, "InferenceRepository", _FakeInferenceRepository)

    with pytest.raises(SystemExit, match="1"):
        await cli._process_batch(created_after=None, created_before=None, limit=5, metrics=["faithfulness"])

    out = capsys.readouterr().out
    assert "processed=2" in out
    assert "completed=1" in out
    assert "failed=1" in out
