from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from arc_model_lab.config import get_settings
from arc_model_lab.db.base import create_async_engine_from_url, create_async_session_factory
from arc_model_lab.db.repositories import ModelRepository
from arc_model_lab.domain import Model, ModelStatus, Provider

_REQUIRED_FIELDS = ("name", "provider", "model_id", "tokenizer_id")


def load_models(path: Path) -> list[Model]:
    """Parse and validate a seed file. Raises ValueError on any problem."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("Seed file must be a JSON array of model objects")

    models = [_parse_entry(index, entry) for index, entry in enumerate(raw)]

    names = [model.name for model in models]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate model names in seed file: {duplicates}")

    return models


def _parse_entry(index: int, entry: object) -> Model:
    if not isinstance(entry, dict):
        raise ValueError(f"Entry {index} must be a JSON object")
    for key in _REQUIRED_FIELDS:
        if not entry.get(key):
            raise ValueError(f"Entry {index} is missing required field '{key}'")
    return Model(
        name=str(entry["name"]),
        provider=Provider(str(entry["provider"])),
        model_id=str(entry["model_id"]),
        tokenizer_id=str(entry["tokenizer_id"]),
        revision=_optional_str(entry.get("revision")),
        adapter_path=_optional_str(entry.get("adapter_path")),
        status=ModelStatus(str(entry.get("status", ModelStatus.ACTIVE))),
    )


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


async def seed(path: Path) -> int:
    models = load_models(path)
    engine = create_async_engine_from_url(get_settings().database_url)
    try:
        session_factory = create_async_session_factory(engine)
        async with session_factory() as session:
            repository = ModelRepository(session)
            for model in models:
                await repository.upsert(model)
            await session.commit()
    finally:
        await engine.dispose()
    return len(models)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Seed the model catalog from a JSON file.")
    parser.add_argument("path", type=Path, help="Path to the seed JSON file")
    parser.add_argument("--check", action="store_true", help="Validate only; do not write")
    args = parser.parse_args(argv)

    if args.check:
        count = len(load_models(args.path))
        print(f"Seed file OK: {count} model(s)")
        return

    count = asyncio.run(seed(args.path))
    print(f"Seeded {count} model(s)")


if __name__ == "__main__":
    main()
