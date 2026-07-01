"""FastAPI dependency accessors read shared singletons from application state."""

from __future__ import annotations

from types import SimpleNamespace

from arc_model_lab.api.dependencies import get_inference_service


def test_get_inference_service_reads_from_app_state() -> None:
    sentinel = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(inference_service=sentinel)))

    assert get_inference_service(request) is sentinel
