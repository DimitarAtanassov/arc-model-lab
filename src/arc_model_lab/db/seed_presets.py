from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from arc_model_lab.config import get_settings
from arc_model_lab.db.base import create_async_engine_from_url, create_async_session_factory
from arc_model_lab.db.repositories import PresetRepository
from arc_model_lab.domain import GenerationConfig, GenerationPreset
from arc_model_lab.domain.generation import enforce_output_cap

_REQUIRED_FIELDS = ("name", "config")


def load_presets(path: Path, *, max_output_tokens_cap: int) -> list[GenerationPreset]:
    """Parse and validate a preset seed file. Raises ValueError on any problem.

    Each config is validated through GenerationConfig (registry bounds and mode
    checks) plus the server output cap at seed time, so an invalid starter fails
    the seed rather than reaching a caller.
    """
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("Seed file must be a JSON array of preset objects")

    presets = [
        _parse_entry(index, entry, max_output_tokens_cap=max_output_tokens_cap) for index, entry in enumerate(raw)
    ]

    names = [preset.name for preset in presets]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate preset names in seed file: {duplicates}")

    return presets


def _parse_entry(index: int, entry: object, *, max_output_tokens_cap: int) -> GenerationPreset:
    if not isinstance(entry, dict):
        raise ValueError(f"Entry {index} must be a JSON object")
    for key in _REQUIRED_FIELDS:
        if entry.get(key) is None:
            raise ValueError(f"Entry {index} is missing required field '{key}'")
    config_data = entry["config"]
    if not isinstance(config_data, dict):
        raise ValueError(f"Entry {index} field 'config' must be a JSON object")
    config = _build_config(config_data, max_output_tokens_cap=max_output_tokens_cap)
    return GenerationPreset(
        name=str(entry["name"]),
        description=_optional_str(entry.get("description")),
        config=config,
    )


def _build_config(data: dict[str, Any], *, max_output_tokens_cap: int) -> GenerationConfig:
    config = GenerationConfig.from_dict(data)
    enforce_output_cap(config, max_output_tokens_cap)
    return config


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


async def seed(path: Path) -> int:
    settings = get_settings()
    presets = load_presets(path, max_output_tokens_cap=settings.max_output_tokens_cap)
    engine = create_async_engine_from_url(settings.database_url)
    try:
        session_factory = create_async_session_factory(engine)
        async with session_factory() as session:
            repository = PresetRepository(session)
            for preset in presets:
                existing = await repository.get_active_by_name(preset.name)
                if existing is None:
                    await repository.add(preset)
                else:
                    await repository.update(replace(existing, description=preset.description, config=preset.config))
            await session.commit()
    finally:
        await engine.dispose()
    return len(presets)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Seed generation presets from a JSON file.")
    parser.add_argument("path", type=Path, help="Path to the seed JSON file")
    parser.add_argument("--check", action="store_true", help="Validate only; do not write")
    args = parser.parse_args(argv)

    if args.check:
        count = len(load_presets(args.path, max_output_tokens_cap=get_settings().max_output_tokens_cap))
        print(f"Seed file OK: {count} preset(s)")
        return

    count = asyncio.run(seed(args.path))
    print(f"Seeded {count} preset(s)")


if __name__ == "__main__":
    main()
