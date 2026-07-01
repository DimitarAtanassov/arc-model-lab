"""Full-stack smoke test: API -> service -> repository -> Postgres."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from arc_model_lab.db.models import InferenceRecord

pytestmark = pytest.mark.integration


def test_summarize_persists_one_inference_row(client: TestClient, db_session: Session) -> None:
    response = client.post("/summarize", json={"input_text": "A long article to summarize."})

    assert response.status_code == 201
    assert response.json()["output_text"] == "fake summary"

    rows = db_session.execute(select(InferenceRecord)).scalars().all()
    assert len(rows) == 1
    assert rows[0].output_text == "fake summary"
