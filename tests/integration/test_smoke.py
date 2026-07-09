"""Full-stack smoke test: API -> service -> repository -> Postgres."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.db.models import InferenceRecord

pytestmark = pytest.mark.integration


async def test_inference_persists_one_inference_row(client: AsyncClient, db_session: AsyncSession) -> None:
    response = await client.post(
        "/inference", json={"model_name": "test-model", "input_text": "A long article to summarize."}
    )

    assert response.status_code == 201
    assert response.json()["output_text"] == "fake summary"

    rows = (await db_session.execute(select(InferenceRecord))).scalars().all()
    assert len(rows) == 1
    assert rows[0].output_text == "fake summary"
