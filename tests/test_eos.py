"""Tests for the equation-of-state (cell-strain) scan using EMT."""

from __future__ import annotations

from fairchem_mcp import domain
from fairchem_mcp.jobs import JobManager
from fairchem_mcp.session import Session

from conftest import wait_for


def _cu_bulk(session: Session) -> str:
    return domain.build_structure(
        session, {"kind": "bulk", "name": "Cu", "crystalstructure": "fcc", "a": 3.6}
    )["structure_id"]


def test_eos_scan_fits_reasonable_bulk_modulus():
    session = Session()
    cid = domain.attach_emt(session)["calculator_id"]
    sid = _cu_bulk(session)

    res = domain.start_eos_scan(session, sid, cid, n_points=9, strain_range=0.05)
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: not job.is_active(), timeout=60)

    status = job.status_dict()
    assert status["status"] == "finished", status
    result = job.result
    assert result["n_points"] == 9
    assert len(result["volumes"]) == 9 and len(result["energies"]) == 9
    # The fit produced an equilibrium near the input cell and a positive modulus.
    assert "bulk_modulus_GPa" in result
    assert result["bulk_modulus_GPa"] > 0.0
    # EMT Cu bulk modulus is ~134 GPa; allow a generous window around the curve.
    assert 50.0 < result["bulk_modulus_GPa"] < 250.0
    # V0 sits inside the scanned volume window.
    assert min(result["volumes"]) <= result["V0"] <= max(result["volumes"])


def test_eos_scan_rejects_nonperiodic_structure():
    session = Session()
    cid = domain.attach_emt(session)["calculator_id"]
    sid = domain.build_structure(session, {"kind": "molecule", "name": "H2O"})[
        "structure_id"
    ]

    res = domain.start_eos_scan(session, sid, cid, n_points=5)
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: not job.is_active(), timeout=30)
    # A molecule has no 3-D cell -> the runner fails with a clear message.
    assert job.status == "failed"
    assert "periodic cell" in (job.error or "")


def test_eos_scan_abort():
    session = Session()
    cid = domain.attach_emt(session)["calculator_id"]
    sid = _cu_bulk(session)

    res = domain.start_eos_scan(session, sid, cid, n_points=20, step_delay=0.1)
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: job.status == "running")
    JobManager(session).steer(job, "abort")
    assert wait_for(lambda: job.status == "aborted")
