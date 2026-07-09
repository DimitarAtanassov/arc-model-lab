"""Database engine, session factory, and declarative base."""

from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    # Deterministic constraint names keep Alembic migrations stable.
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


def create_async_engine_from_url(url: str, *, echo: bool = False) -> AsyncEngine:
    """Create the async engine used by the app's request path.

    The DSN is the same ``postgresql+psycopg://`` URL as the sync engine: psycopg3
    drives both, so async adoption needs no second driver and Alembic keeps running
    migrations synchronously against the same database.
    """
    return create_async_engine(url, echo=echo, pool_pre_ping=True, future=True)


def create_async_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
