"""Unit tests for InferenceService input guards."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from arc_model_lab.domain import InputTooLargeError
from arc_model_lab.services.inference_service import InferenceService
from arc_model_lab.services.model_service import ModelService


def test_summarize_rejects_oversized_input(fake_model_service: ModelService) -> None:
    service = InferenceService(fake_model_service, "test-model")

    with pytest.raises(InputTooLargeError):
        service.summarize(MagicMock(spec=Session), "x" * 60_000)
