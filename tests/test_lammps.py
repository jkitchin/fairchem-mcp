"""Tests for the LAMMPS-as-ASE-calculator integration.

LAMMPS is an optional engine. These tests skip cleanly if the `lammps` package
isn't importable/loadable in this environment (e.g. no LAMMPS build, or the macOS
MPI dylib fix can't be applied). When it is available, we drive a short MD and a
relaxation through the normal steering engine with a simple LJ potential (no
external potential file needed).
"""

from __future__ import annotations

import numpy as np
import pytest

from fairchem_mcp import domain
from fairchem_mcp.jobs import JobManager
from fairchem_mcp.session import Session

from conftest import wait_for


def _lammps_available() -> bool:
    try:
        domain._ensure_lammps_loadable()
        from ase.calculators.lammpslib import LAMMPSlib  # noqa: F401
        from lammps import lammps

        lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
        lmp.close()
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _lammps_available(), reason="lammps not importable/loadable in this env"
)


def _cu_cluster(session: Session) -> str:
    from ase.cluster import Icosahedron

    atoms = Icosahedron("Cu", noshells=2)
    atoms.center(vacuum=10.0)
    atoms.pbc = True
    return session.add_structure(atoms)


# Simple LJ for Cu: epsilon (eV), sigma (Å). Qualitative, but enough to drive
# dynamics without shipping a potential file.
_LJ = dict(pair_style="lj/cut 7.0", pair_coeff="1 1 0.4 2.34", atom_types={"Cu": 1})


def test_attach_lammps_computes_energy_and_forces():
    session = Session()
    sid = _cu_cluster(session)
    cid = domain.attach_lammps(session, **_LJ)["calculator_id"]

    atoms = session.get_structure(sid)
    atoms.calc = session.get_calculator(cid)
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    assert np.isfinite(e)
    assert f.shape == (len(atoms), 3)
    assert np.isfinite(f).all()


def test_lammps_drives_md_under_steering_engine():
    session = Session()
    sid = _cu_cluster(session)
    cid = domain.attach_lammps(session, **_LJ)["calculator_id"]

    res = domain.start_md(session, sid, cid, temperature_K=300.0, steps=15)
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: not job.is_active(), timeout=120)
    status = job.status_dict()
    assert status["status"] == "finished", status
    assert status["step"] >= 1
    assert status["temperature_K"] is not None


def test_lammps_drives_relaxation():
    session = Session()
    sid = _cu_cluster(session)
    cid = domain.attach_lammps(session, **_LJ)["calculator_id"]

    res = domain.start_relaxation(session, sid, cid, fmax=0.1, steps=200)
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: not job.is_active(), timeout=120)
    assert job.status_dict()["status"] in ("converged", "finished")


def test_lammps_md_abort():
    session = Session()
    sid = _cu_cluster(session)
    cid = domain.attach_lammps(session, **_LJ)["calculator_id"]

    res = domain.start_md(session, sid, cid, temperature_K=300.0, steps=10000,
                          step_delay=0.05)
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: job.status == "running")
    JobManager(session).steer(job, "abort")
    assert wait_for(lambda: job.status == "aborted")
