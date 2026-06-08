"""Tests for the formation-energy convex-hull tool.

The hull math is exercised directly with synthetic energies (exact, fast), and
the full job path is checked end-to-end with EMT on a Cu-Au system.
"""

from __future__ import annotations

from ase.build import bulk

from fairchem_mcp import domain
from fairchem_mcp.jobs import JobManager, _hull_from_entries
from fairchem_mcp.session import Session

from conftest import wait_for


# --- pure hull math --------------------------------------------------------
def test_hull_math_picks_the_stable_compound():
    # A-B binary. Pure A, pure B at mu=0 each; AB sits below the tie-line, AB3
    # sits above it. Energies are total (per formula unit) = mu*n + N*Ef.
    entries = [
        {"label": "A", "counts": {"A": 1}, "energy": 0.0},      # mu_A = 0
        {"label": "B", "counts": {"B": 1}, "energy": 0.0},      # mu_B = 0
        {"label": "AB", "counts": {"A": 1, "B": 1}, "energy": -0.4},   # Ef = -0.2/atom
        {"label": "AB3", "counts": {"A": 1, "B": 3}, "energy": -0.2},  # Ef = -0.05/atom
    ]
    res = _hull_from_entries(entries)
    by = {p["label"]: p for p in res["phases"]}

    assert by["AB"]["formation_energy_per_atom"] == -0.2
    assert by["AB3"]["formation_energy_per_atom"] == -0.05
    # Pure elements and AB are on the hull; AB3 is above the A-AB tie-line.
    assert by["A"]["on_hull"] and by["B"]["on_hull"]
    assert by["AB"]["on_hull"]
    assert not by["AB3"]["on_hull"]
    assert by["AB3"]["energy_above_hull_per_atom"] > 0.0
    assert set(res["stable_phases"]) == {"A", "B", "AB"}


def test_hull_math_overrides_references():
    # No pure entries -> must supply references; here both at 0 so Ef = E/N.
    entries = [
        {"label": "AB", "counts": {"A": 1, "B": 1}, "energy": -1.0},
    ]
    res = _hull_from_entries(entries, overrides={"A": 0.0, "B": 0.0})
    assert res["phases"][0]["formation_energy_per_atom"] == -0.5


def test_hull_math_missing_reference_raises():
    import pytest

    entries = [{"label": "AB", "counts": {"A": 1, "B": 1}, "energy": -1.0}]
    with pytest.raises(ValueError, match="reference energy"):
        _hull_from_entries(entries)


# --- full job path with EMT ------------------------------------------------
def _cu_au_system(session: Session) -> list:
    """Pure Cu, pure Au, and an ordered Cu3Au-ish L1_2 cell (all EMT elements)."""
    cu = bulk("Cu", "fcc", a=3.6, cubic=True)          # 4-atom conventional cell
    au = bulk("Au", "fcc", a=4.08, cubic=True)
    cu3au = bulk("Cu", "fcc", a=3.75, cubic=True)      # L1_2: corners Au, faces Cu
    cu3au.symbols = ["Au", "Cu", "Cu", "Cu"]
    return [session.add_structure(a) for a in (cu, au, cu3au)]


def test_convex_hull_emt_cu_au():
    session = Session()
    cid = domain.attach_emt(session)["calculator_id"]
    sids = _cu_au_system(session)

    res = domain.start_convex_hull(session, sids, cid, relax=True)
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: not job.is_active(), timeout=120)

    status = job.status_dict()
    assert status["status"] == "finished", status
    result = job.result
    assert result["elements"] == ["Au", "Cu"]

    by = {p["label"]: p for p in result["phases"]}
    # Pure elements define the references -> Ef = 0 and on the hull.
    pures = [p for p in result["phases"] if len(p["composition"]) == 1]
    for p in pures:
        assert abs(p["formation_energy_per_atom"]) < 1e-6
        assert p["on_hull"]
    # Every phase has a non-negative distance above the hull.
    assert all(p["energy_above_hull_per_atom"] >= 0.0 for p in result["phases"])
    assert len(result["stable_phases"]) >= 2  # at least the two pure elements


def test_convex_hull_missing_reference_fails_cleanly():
    session = Session()
    cid = domain.attach_emt(session)["calculator_id"]
    # Only a mixed Cu-Au structure, no pure references and none supplied.
    cu3au = bulk("Cu", "fcc", a=3.75, cubic=True)
    cu3au.symbols = ["Au", "Cu", "Cu", "Cu"]
    sid = session.add_structure(cu3au)

    res = domain.start_convex_hull(session, [sid], cid)
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: not job.is_active(), timeout=60)
    assert job.status == "failed"
    assert "reference energy" in (job.error or "")
    # Energies are still preserved for the user.
    assert job.result and "energies" in job.result


def test_convex_hull_abort():
    session = Session()
    cid = domain.attach_emt(session)["calculator_id"]
    sids = _cu_au_system(session)

    res = domain.start_convex_hull(
        session, sids, cid, relax=True, step_delay=0.2
    )
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: job.status == "running")
    JobManager(session).steer(job, "abort")
    assert wait_for(lambda: job.status == "aborted")
