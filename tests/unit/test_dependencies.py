from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace

from arc_model_lab.api.dependencies import (
    get_evaluation_service,
    get_inference_service,
    get_model_catalog_service,
    get_session,
)


def test_get_inference_service_reads_from_app_state() -> None:
    sentinel = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(inference_service=sentinel)))

    assert get_inference_service(request) is sentinel


def test_get_evaluation_service_reads_from_app_state() -> None:
    sentinel = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(evaluation_service=sentinel)))

    assert get_evaluation_service(request) is sentinel


def test_get_model_catalog_service_reads_from_app_state() -> None:
    sentinel = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(model_catalog_service=sentinel)))

    assert get_model_catalog_service(request) is sentinel


async def test_get_session_yields_request_scoped_session() -> None:
    seen: list[object] = []

    class _SessionFactory:
        def __call__(self) -> AsyncIterator[object]:
            @asynccontextmanager
            async def _manager() -> AsyncIterator[object]:
                session = object()
                seen.append(session)
                yield session

            return _manager()

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(session_factory=_SessionFactory())))

    sessions = [session async for session in get_session(request)]

    assert len(sessions) == 1
    assert sessions[0] is seen[0]
