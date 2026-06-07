"""Tests for NEB and phonon jobs using the fast EMT calculator."""

from __future__ import annotations

import pytest

from fairchem_mcp import domain
from fairchem_mcp.jobs import JobManager
from fairchem_mcp.session import Session

from conftest import wait_for


def _adatom_hop_endpoints(session: Session, cid: str):
    """Two endpoints for an Au adatom hopping between hollow sites on Al(100)."""
    from ase.build import add_adsorbate, fcc100
    from ase.constraints import FixAtoms

    slab = fcc100("Al", size=(2, 2, 3), vacuum=10.0)
    zmean = slab.positions[:, 2].mean()
    slab.set_constraint(FixAtoms(mask=[p[2] < zmean for p in slab.positions]))
    add_adsorbate(slab, "Au", 1.7, "hollow")

    initial = slab.copy()
    final = slab.copy()
    final.positions[-1, 0] += final.cell[0, 0] / 2  # shift adatom one hollow over

    sid_i = session.add_structure(initial)
    sid_f = session.add_structure(final)
    # Lightly relax each endpoint.
    for sid in (sid_i, sid_f):
        res = domain.start_relaxation(session, sid, cid, fmax=0.1, steps=200)
        assert wait_for(lambda: not session.get_job(res["job_id"]).is_active())
    return sid_i, sid_f


def test_neb_converges_with_positive_barrier():
    session = Session()
    cid = domain.attach_emt(session)["calculator_id"]
    sid_i, sid_f = _adatom_hop_endpoints(session, cid)

    res = domain.start_neb(session, sid_i, sid_f, cid, nimages=3, optimizer="LBFGS",
                           fmax=0.1, steps=300)
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: not job.is_active(), timeout=60)

    status = job.status_dict()
    assert status["status"] == "converged", status
    assert job.result["barrier"] > 0.0
    assert job.result["n_images"] == 5  # 3 intermediate + 2 endpoints


def test_neb_set_climb_steer():
    session = Session()
    cid = domain.attach_emt(session)["calculator_id"]
    sid_i, sid_f = _adatom_hop_endpoints(session, cid)

    res = domain.start_neb(session, sid_i, sid_f, cid, nimages=3, optimizer="LBFGS",
                           fmax=0.05, steps=400, step_delay=0.02)
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: job.step >= 2)
    JobManager(session).steer(job, "set_climb", climb=True)
    assert job.neb.climb is True

    assert wait_for(lambda: not job.is_active(), timeout=60)
    assert job.status_dict()["status"] == "converged"


def test_phonon_completes_and_reports_stability():
    session = Session()
    cid = domain.attach_emt(session)["calculator_id"]
    sid = domain.build_structure(
        session, {"kind": "bulk", "name": "Cu", "crystalstructure": "fcc", "a": 3.6}
    )["structure_id"]

    res = domain.start_phonons(session, sid, cid, supercell=(2, 2, 2), delta=0.05)
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: not job.is_active(), timeout=60)

    status = job.status_dict()
    assert status["status"] == "finished", status
    result = job.result
    assert result["stable"] is True
    # Primitive fcc cell: 3 acoustic modes, ~0 at gamma.
    assert len(result["gamma_frequencies_THz"]) == 3
    assert result["n_imaginary_modes"] == 0


def test_phonon_abort():
    session = Session()
    cid = domain.attach_emt(session)["calculator_id"]
    sid = domain.build_structure(
        session, {"kind": "bulk", "name": "Cu", "crystalstructure": "fcc", "a": 3.6}
    )["structure_id"]

    # step_delay keeps it alive long enough to abort deterministically.
    res = domain.start_phonons(session, sid, cid, supercell=(2, 2, 2), step_delay=0.1)
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: job.status == "running")
    JobManager(session).steer(job, "abort")
    assert wait_for(lambda: job.status == "aborted")


def test_neb_and_phonon_are_bound_in_namespace():
    session = Session()
    cid = domain.attach_emt(session)["calculator_id"]
    sid = domain.build_structure(
        session, {"kind": "bulk", "name": "Cu", "crystalstructure": "fcc", "a": 3.6}
    )["structure_id"]
    res = domain.start_phonons(session, sid, cid, supercell=(2, 2, 2))
    wait_for(lambda: not session.get_job(res["job_id"]).is_active(), timeout=60)
    # The Phonons object is reachable from the execute/inspect escape hatch.
    assert "ph" in session.namespace
