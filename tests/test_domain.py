"""Tests for domain helpers (structure building, registries, model listing)."""

from __future__ import annotations

from fairchem_mcp import domain
from fairchem_mcp.session import Session


def test_build_bulk_and_register(session: Session):
    info = domain.build_structure(
        session,
        {"kind": "bulk", "name": "Cu", "crystalstructure": "fcc", "a": 3.6, "repeat": [2, 2, 2]},
    )
    assert info["natoms"] == 8
    assert info["formula"] == "Cu8"
    # Registered and aliased in the namespace.
    assert info["structure_id"] in session.structures
    assert session.namespace["atoms"] is session.structures[info["structure_id"]]


def test_build_molecule(session: Session):
    info = domain.build_structure(session, {"kind": "molecule", "name": "H2O"})
    assert info["natoms"] == 3
    assert info["formula"] == "H2O"


def test_shared_namespace_between_tools_and_repl(session: Session):
    cid = domain.attach_emt(session)["calculator_id"]
    sid = domain.build_structure(
        session, {"kind": "bulk", "name": "Cu", "crystalstructure": "fcc", "a": 3.6}
    )["structure_id"]
    domain._attach(session, sid, cid)
    # The escape hatch sees the same live object the tool created.
    energy = eval("atoms.get_potential_energy()", session.namespace)
    assert isinstance(energy, float)


def test_list_models_reports_availability():
    out = domain.list_models()
    assert "available" in out
    assert isinstance(out["models"], list)
