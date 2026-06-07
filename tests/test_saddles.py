"""Tests for the two saddle-point searches (EMT, fast).

* dimer method (ASE MinModeTranslate) — Hessian-free, single-ended.
* POUNCE find_saddles adapter — multistart eigenvector following with explicit
  Morse-index classification (skips if `pounce` isn't installed).

Both use an Al adatom hopping between hollow sites on Al(100): the hollow site is
a minimum, the bridge site is the index-1 saddle (transition state).
"""

from __future__ import annotations

import numpy as np
import pytest

from fairchem_mcp import domain
from fairchem_mcp.jobs import JobManager
from fairchem_mcp.session import Session

from conftest import wait_for


def _relaxed_adatom(session: Session):
    """A relaxed Al adatom in a hollow site on a frozen Al(100) slab."""
    from ase.build import add_adsorbate, fcc100
    from ase.constraints import FixAtoms

    slab = fcc100("Al", size=(2, 2, 3), vacuum=10.0)
    zmean = slab.positions[:, 2].mean()
    slab.set_constraint(FixAtoms(mask=[p[2] < zmean for p in slab.positions]))
    add_adsorbate(slab, "Al", 1.7, "hollow")

    cid = domain.attach_emt(session)["calculator_id"]
    sid = session.add_structure(slab)
    res = domain.start_relaxation(session, sid, cid, fmax=0.05, steps=200)
    assert wait_for(lambda: not session.get_job(res["job_id"]).is_active(), timeout=60)
    return sid, cid


def _bridge_push(natoms: int) -> list[float]:
    """A length-3N displacement nudging the (last) adatom toward the bridge."""
    dvec = np.zeros((natoms, 3))
    dvec[-1, 0] = 0.3
    return dvec.ravel().tolist()


# --- dimer -----------------------------------------------------------------

def test_dimer_finds_index1_saddle():
    session = Session()
    sid, cid = _relaxed_adatom(session)
    natoms = len(session.get_structure(sid))

    res = domain.start_saddle_search(
        session, sid, cid, displacement_vector=_bridge_push(natoms),
        fmax=0.05, steps=300,
    )
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: not job.is_active(), timeout=60)

    status = job.status_dict()
    assert status["status"] == "converged", status
    result = job.result
    # A transition state: one negative curvature mode.
    assert result["is_index1_saddle"] is True
    assert result["curvature"] < 0.0
    assert result["max_force"] <= 0.05 + 1e-6
    # The saddle sits above the relaxed minimum in energy.
    e_min = session.get_job(
        [j for j in session.jobs.values() if j.kind == "relax"][0].id
    ).status_dict()["energy"]
    assert result["energy"] > e_min


def test_dimer_abort():
    session = Session()
    sid, cid = _relaxed_adatom(session)

    res = domain.start_saddle_search(
        session, sid, cid, displace_magnitude=0.05, fmax=0.001, steps=10000,
        step_delay=0.05, seed=3,
    )
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: job.status == "running")
    JobManager(session).steer(job, "abort")
    assert wait_for(lambda: job.status == "aborted")


def test_dimer_curvature_is_reported_during_run():
    session = Session()
    sid, cid = _relaxed_adatom(session)
    natoms = len(session.get_structure(sid))

    res = domain.start_saddle_search(
        session, sid, cid, displacement_vector=_bridge_push(natoms),
        fmax=0.05, steps=300, step_delay=0.01,
    )
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: job.step >= 1)
    # curvature is surfaced in the live status while running.
    assert wait_for(lambda: job.status_dict().get("curvature") is not None)
    assert wait_for(lambda: not job.is_active(), timeout=60)


# --- Sella -----------------------------------------------------------------

def test_sella_finds_index1_transition_state():
    pytest.importorskip("sella", reason="sella not installed")
    session = Session()
    sid, cid = _relaxed_adatom(session)
    natoms = len(session.get_structure(sid))

    res = domain.start_sella_search(
        session, sid, cid, order=1,
        displacement_vector=_bridge_push(natoms), fmax=0.05, steps=200,
    )
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: not job.is_active(), timeout=120)

    status = job.status_dict()
    assert status["status"] == "converged", status
    result = job.result
    # An order-1 saddle: exactly one negative Hessian eigenvalue.
    assert result["n_negative_eigenvalues"] == 1
    assert result["is_target_order_saddle"] is True
    assert result["lowest_eigenvalue"] < 0.0
    assert result["max_force"] <= 0.05 + 1e-6
    assert session.get_structure(result["structure_id"]) is not None


def test_sella_abort():
    pytest.importorskip("sella", reason="sella not installed")
    session = Session()
    sid, cid = _relaxed_adatom(session)

    res = domain.start_sella_search(
        session, sid, cid, displace_magnitude=0.05, fmax=0.0001, steps=10000,
        step_delay=0.05, seed=3,
    )
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: job.status == "running")
    JobManager(session).steer(job, "abort")
    assert wait_for(lambda: job.status == "aborted")


# --- POUNCE find_saddles ---------------------------------------------------

pounce = pytest.importorskip("pounce", reason="pounce-solver not installed")


def test_pounce_saddles_finds_index1_transition_state():
    session = Session()
    sid, cid = _relaxed_adatom(session)
    adatom = len(session.get_structure(sid)) - 1

    res = domain.start_pounce_saddles(
        session, sid, cid, active_indices=[adatom], index=1, n_saddles=2,
        grad_tol=1e-3, max_step=0.15, dedup=0.1, displace_magnitude=0.25, seed=0,
    )
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: not job.is_active(), timeout=120)

    result = job.result
    assert result["n_found"] >= 1, result
    sad = result["saddles"][0]
    assert sad["morse_index"] == 1
    assert sad["grad_norm"] <= 1e-3 + 1e-6
    # Exactly one negative Hessian eigenvalue for an index-1 saddle.
    assert sum(1 for e in sad["eigenvalues"] if e < 0) == 1
    # The registered saddle structure is retrievable.
    assert session.get_structure(sad["structure_id"]) is not None


def test_pounce_saddles_rejects_all_fixed():
    session = Session()
    sid, cid = _relaxed_adatom(session)
    with pytest.raises(ValueError, match="no active atoms"):
        domain.start_pounce_saddles(session, sid, cid, active_indices=[])
