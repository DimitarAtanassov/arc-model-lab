"""Unit tests for summarization prompt construction."""

from __future__ import annotations

from arc_model_lab.services.inference_service import build_summary_messages


def test_build_summary_messages_has_system_then_user_turn() -> None:
    messages = build_summary_messages("the article body")

    assert [m["role"] for m in messages] == ["system", "user"]
    assert "the article body" in messages[1]["content"]
