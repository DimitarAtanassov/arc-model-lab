from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from arc_model_lab.domain import Model


class ModelResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: UUID  # noqa: A003 - mirrors the domain primary key
    name: str
    provider: str
    # The provider-native identifier (for HuggingFace, the repo id). Distinct from
    # name, which is the unique catalog handle callers pass as model_name.
    model_id: str
    tokenizer_id: str
    revision: str | None
    adapter_path: str | None
    status: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, model: Model) -> ModelResponse:
        return cls(
            id=model.id,
            name=model.name,
            provider=model.provider.value,
            model_id=model.model_id,
            tokenizer_id=model.tokenizer_id,
            revision=model.revision,
            adapter_path=model.adapter_path,
            status=model.status.value,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
