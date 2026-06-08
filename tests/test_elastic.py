"""Tests for the elastic stiffness-tensor (stress-vs-strain) scan using EMT."""

from __future__ import annotations

import numpy as np

from fairchem_mcp import domain
from fairchem_mcp.jobs import JobManager
from fairchem_mcp.session import Session

from conftest import wait_for


def _cu_bulk(session: Session) -> str:
    return domain.build_structure(
        session, {"kind": "bulk", "name": "Cu", "crystalstructure": "fcc", "a": 3.6}
    )["structure_id"]


def test_elastic_scan_cubic_metal():
    session = Session()
    cid = domain.attach_emt(session)["calculator_id"]
    sid = _cu_bulk(session)

    res = domain.start_elastic_scan(session, sid, cid, n_strains=5, max_strain=0.01)
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: not job.is_active(), timeout=120)

    status = job.status_dict()
    assert status["status"] == "finished", status
    result = job.result

    C = np.array(result["C_GPa"])
    assert C.shape == (6, 6)
    # The fitted stiffness matrix is symmetric (we symmetrize) ...
    assert np.allclose(C, C.T, atol=1e-6)
    # ... and for a cubic metal C11 > C12 > 0 and C44 > 0.
    assert C[0, 0] > 0 and C[3, 3] > 0
    assert C[0, 0] > abs(C[0, 1])

    # Born mechanical stability: positive-definite C.
    assert result["mechanically_stable"] is True
    assert all(x > 0 for x in result["C_eigenvalues_GPa"])

    # Hill bulk modulus should land near the EOS value for EMT Cu (~134 GPa).
    K = result["bulk_modulus_GPa"]
    assert 50.0 < K < 250.0
    assert result["shear_modulus_GPa"] > 0.0
    assert result["youngs_modulus_GPa"] > 0.0
    assert -1.0 < result["poisson_ratio"] < 0.5


def test_elastic_bulk_modulus_matches_eos():
    """The cell-strain EOS and stress-strain elastic scan should agree on K."""
    session = Session()
    cid = domain.attach_emt(session)["calculator_id"]
    sid = _cu_bulk(session)

    eos = domain.start_eos_scan(session, sid, cid, n_points=9, strain_range=0.04)
    ejob = session.get_job(eos["job_id"])
    assert wait_for(lambda: not ejob.is_active(), timeout=60)
    K_eos = ejob.result["bulk_modulus_GPa"]

    el = domain.start_elastic_scan(session, sid, cid, n_strains=5, max_strain=0.008)
    cjob = session.get_job(el["job_id"])
    assert wait_for(lambda: not cjob.is_active(), timeout=120)
    K_elastic = cjob.result["bulk_modulus_GPa"]

    # Both probe the same energy surface; agree to ~25% (different strain modes,
    # ions clamped, finite-difference grids).
    assert abs(K_eos - K_elastic) / K_eos < 0.25, (K_eos, K_elastic)


def test_elastic_scan_rejects_nonperiodic_structure():
    session = Session()
    cid = domain.attach_emt(session)["calculator_id"]
    sid = domain.build_structure(session, {"kind": "molecule", "name": "H2O"})[
        "structure_id"
    ]

    res = domain.start_elastic_scan(session, sid, cid, n_strains=5)
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: not job.is_active(), timeout=30)
    assert job.status == "failed"
    assert "periodic cell" in (job.error or "")


def test_elastic_scan_abort():
    session = Session()
    cid = domain.attach_emt(session)["calculator_id"]
    sid = _cu_bulk(session)

    res = domain.start_elastic_scan(
        session, sid, cid, n_strains=9, max_strain=0.01, step_delay=0.1
    )
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: job.status == "running")
    JobManager(session).steer(job, "abort")
    assert wait_for(lambda: job.status == "aborted")
