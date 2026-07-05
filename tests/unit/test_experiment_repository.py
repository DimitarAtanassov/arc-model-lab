"""Unit tests for ExperimentRepository error translation (no database).

The session is faked so these run without Postgres. They prove the repository
seam maps a duplicate-name violation to ExperimentNameConflictError, lets any
other IntegrityError (e.g. a foreign key) propagate, and turns corrupt stored
config into CorruptStoredDataError rather than a client-facing validation error.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from arc_model_lab.db.models import ExperimentRecord
from arc_model_lab.db.repositories import ExperimentRepository
from arc_model_lab.domain import (
    CorruptStoredDataError,
    Experiment,
    ExperimentNameConflictError,
    GenerationConfig,
)


class _FakeDBAPIError(Exception):
    """Stand-in for the psycopg error carried on ``IntegrityError.orig``."""

    def __init__(self, constraint_name: str) -> None:
        super().__init__(constraint_name)
        self.diag = SimpleNamespace(constraint_name=constraint_name)


def _integrity_error(constraint_name: str) -> IntegrityError:
    return IntegrityError("INSERT", {}, _FakeDBAPIError(constraint_name))


def _experiment() -> Experiment:
    return Experiment(
        name="exp",
        model_id=uuid4(),
        generation_config=GenerationConfig(temperature=0.0, max_output_tokens=256),
    )


def test_add_translates_duplicate_name_to_conflict() -> None:
    session = MagicMock(spec=Session)
    session.flush.side_effect = _integrity_error("uq_experiments_name")

    with pytest.raises(ExperimentNameConflictError):
        ExperimentRepository(session).add(_experiment())


def test_add_reraises_non_name_integrity_error() -> None:
    # A foreign-key violation (e.g. a concurrently deleted model) must not be
    # mislabeled as a name conflict.
    session = MagicMock(spec=Session)
    session.flush.side_effect = _integrity_error("fk_experiments_model_id_models")

    with pytest.raises(IntegrityError):
        ExperimentRepository(session).add(_experiment())


def test_add_reraises_when_constraint_name_is_absent() -> None:
    # If the driver exposes no constraint name, the error is unclassifiable and
    # must propagate rather than be assumed to be a name conflict.
    session = MagicMock(spec=Session)
    session.flush.side_effect = IntegrityError("INSERT", {}, Exception("boom"))

    with pytest.raises(IntegrityError):
        ExperimentRepository(session).add(_experiment())


def test_get_raises_on_corrupt_stored_config() -> None:
    record = ExperimentRecord(
        id=uuid4(),
        name="corrupt",
        model_id=uuid4(),
        generation_config={"temperature": 1},
    )
    session = MagicMock(spec=Session)
    session.get.return_value = record

    with pytest.raises(CorruptStoredDataError):
        ExperimentRepository(session).get(uuid4())
