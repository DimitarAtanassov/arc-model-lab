from __future__ import annotations

import argparse
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from arc_model_lab.config import get_settings
from arc_model_lab.db.base import create_async_engine_from_url, create_async_session_factory
from arc_model_lab.db.repositories import ModelRepository
from arc_model_lab.domain import Model, ModelStatus
from arc_model_lab.services.inference_service import build_summary_messages
from arc_model_lab.services.model_service import ModelService

_SMOKE_INPUT = "The quick brown fox jumps over the lazy dog, repeatedly and with enthusiasm."


def _session_factory() -> async_sessionmaker[AsyncSession]:
    return create_async_session_factory(create_async_engine_from_url(get_settings().database_url))


def _print_model(model: Model) -> None:
    print(f"{model.name}\t{model.model_id}\t{model.revision or '-'}\t{model.status}")


async def _list() -> None:
    async with _session_factory()() as session:
        for model in await ModelRepository(session).list_all():
            _print_model(model)


async def _get(name: str) -> None:
    async with _session_factory()() as session:
        model = await ModelRepository(session).get_by_name(name)
    if model is None:
        raise SystemExit(f"Model not found: {name}")
    _print_model(model)


async def _set_status(name: str, status: ModelStatus) -> None:
    async with _session_factory()() as session:
        model = await ModelRepository(session).set_status(name, status)
        await session.commit()
    if model is None:
        raise SystemExit(f"Model not found: {name}")
    _print_model(model)


async def _smoke(name: str) -> None:
    async with _session_factory()() as session:
        model = await ModelRepository(session).get_by_name(name)
    if model is None:
        raise SystemExit(f"Model not found: {name}")
    result = ModelService(get_settings()).generate(model, build_summary_messages(_SMOKE_INPUT))
    print(result.output_text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Model catalog operations.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List catalog models")
    for command in ("get", "activate", "deactivate", "smoke"):
        sub.add_parser(command).add_argument("--name", required=True)

    args = parser.parse_args(argv)

    if args.command == "list":
        asyncio.run(_list())
    elif args.command == "get":
        asyncio.run(_get(args.name))
    elif args.command == "activate":
        asyncio.run(_set_status(args.name, ModelStatus.ACTIVE))
    elif args.command == "deactivate":
        asyncio.run(_set_status(args.name, ModelStatus.INACTIVE))
    elif args.command == "smoke":  # pragma: no branch - argparse restricts command to this exhaustive set
        asyncio.run(_smoke(args.name))


if __name__ == "__main__":
    main()
