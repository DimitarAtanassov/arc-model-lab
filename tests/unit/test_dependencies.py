"""FastAPI dependency accessors read shared singletons from application state."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from arc_model_lab.api.dependencies import get_evaluation_service, get_inference_service, get_session


def test_get_inference_service_reads_from_app_state() -> None:
    sentinel = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(inference_service=sentinel)))

    assert get_inference_service(request) is sentinel


def test_get_evaluation_service_reads_from_app_state() -> None:
    sentinel = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(evaluation_service=sentinel)))

    assert get_evaluation_service(request) is sentinel


def test_get_session_yields_request_scoped_session() -> None:
    seen: list[object] = []

    class _SessionFactory:
        def __call__(self) -> object:
            @contextmanager
            def _manager() -> object:
                session = object()
                seen.append(session)
                yield session

            return _manager()

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(session_factory=_SessionFactory())))

    sessions = list(get_session(request))

    assert len(sessions) == 1
    assert sessions[0] is seen[0]
