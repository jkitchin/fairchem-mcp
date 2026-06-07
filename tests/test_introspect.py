"""Tests for code-awareness (introspect.py)."""

from __future__ import annotations

from fairchem_mcp.introspect import introspect
from fairchem_mcp.session import Session


def test_static_introspect_stdlib_callable():
    out = introspect("math.hypot", live=False, namespace={})
    assert out["kind"] == "callable"
    assert "docstring" in out


def test_static_introspect_class_lists_members():
    out = introspect("ase.atoms.Atoms", live=False, namespace={})
    assert out["kind"] == "class"
    assert "get_potential_energy" in out["members"]


def test_static_completion_with_trailing_dot():
    out = introspect("ase.build.", live=False, namespace={})
    names = {c["name"] for c in out["completions"]}
    assert "bulk" in names
    assert "molecule" in names


def test_live_introspect_uses_session_namespace():
    from fairchem_mcp import domain

    session = Session()
    domain.attach_emt(session)
    domain.build_structure(
        session, {"kind": "bulk", "name": "Cu", "crystalstructure": "fcc", "a": 3.6}
    )
    # 'atoms' alias now exists in the live namespace.
    out = introspect("atoms", live=True, namespace=session.namespace)
    assert out["target"] == "atoms"

    members = introspect("atoms.", live=True, namespace=session.namespace)
    names = {c["name"] for c in members["completions"]}
    assert "get_potential_energy" in names


def test_static_introspect_fairchem_if_available():
    import importlib.util

    if importlib.util.find_spec("fairchem") is None:
        return  # fairchem optional; skip silently
    out = introspect(
        "fairchem.core.calculate.ase_calculator.FAIRChemCalculator",
        live=False,
        namespace={},
    )
    assert out["kind"] == "class"
    assert "from_model_checkpoint" in out["members"]
