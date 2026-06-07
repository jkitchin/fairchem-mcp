"""Shared test fixtures and helpers."""

from __future__ import annotations

import time

import pytest

from fairchem_mcp import domain
from fairchem_mcp.session import Session


@pytest.fixture
def session() -> Session:
    return Session()


@pytest.fixture
def emt_setup(session: Session):
    """A rattled Cu bulk with an EMT calculator attached. Returns (session, sid, cid)."""
    cid = domain.attach_emt(session)["calculator_id"]
    sid = domain.build_structure(
        session,
        {
            "kind": "bulk",
            "name": "Cu",
            "crystalstructure": "fcc",
            "a": 3.6,
            "repeat": [3, 3, 3],
            "rattle": 0.2,
        },
    )["structure_id"]
    return session, sid, cid


def wait_for(predicate, timeout: float = 20.0, interval: float = 0.02):
    """Poll predicate() until truthy or timeout. Returns the last value."""
    deadline = time.time() + timeout
    value = predicate()
    while not value and time.time() < deadline:
        time.sleep(interval)
        value = predicate()
    return value
