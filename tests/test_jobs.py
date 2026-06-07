"""Tests for the steering engine (jobs.py) using the fast EMT calculator.

No GPU or FAIRChem model required.
"""

from __future__ import annotations

from fairchem_mcp import domain
from fairchem_mcp.jobs import JobManager

from conftest import wait_for


def test_relaxation_converges_and_lowers_energy(emt_setup):
    session, sid, cid = emt_setup
    res = domain.start_relaxation(session, sid, cid, optimizer="FIRE", fmax=0.05, steps=500)
    job = session.get_job(res["job_id"])

    done = wait_for(lambda: not job.is_active())
    assert done, "job did not finish in time"

    status = job.status_dict()
    assert status["status"] == "converged", status
    assert status["max_force"] <= 0.05

    # Energy went down from first to last snapshot.
    history = job.history
    assert history[-1]["energy"] < history[0]["energy"]


def test_abort_stops_a_running_job(emt_setup):
    session, sid, cid = emt_setup
    # step_delay keeps it alive long enough to abort deterministically.
    res = domain.start_relaxation(
        session, sid, cid, fmax=1e-8, steps=500, step_delay=0.02
    )
    job = session.get_job(res["job_id"])

    assert wait_for(lambda: job.status == "running")
    JobManager(session).steer(job, "abort")

    assert wait_for(lambda: job.status == "aborted")
    assert job.status == "aborted"
    assert job.step < 500  # stopped early


def test_pause_then_resume(emt_setup):
    session, sid, cid = emt_setup
    res = domain.start_relaxation(
        session, sid, cid, fmax=1e-8, steps=500, step_delay=0.02
    )
    job = session.get_job(res["job_id"])

    assert wait_for(lambda: job.status == "running")
    JobManager(session).steer(job, "pause")
    assert wait_for(lambda: job.status == "paused")

    step_at_pause = job.step
    # While paused, the step counter should not advance.
    import time

    time.sleep(0.2)
    assert job.step == step_at_pause

    JobManager(session).steer(job, "resume")
    assert wait_for(lambda: job.step > step_at_pause)

    JobManager(session).steer(job, "abort")
    assert wait_for(lambda: not job.is_active())


def test_switch_optimizer_carries_positions(emt_setup):
    session, sid, cid = emt_setup
    res = domain.start_relaxation(
        session, sid, cid, optimizer="FIRE", fmax=0.05, steps=1000, step_delay=0.01
    )
    job = session.get_job(res["job_id"])

    assert wait_for(lambda: job.step >= 3)
    energy_before = job.atoms.get_potential_energy()
    JobManager(session).steer(job, "switch_optimizer", optimizer="LBFGS")

    done = wait_for(lambda: not job.is_active())
    assert done
    status = job.status_dict()
    assert status["status"] == "converged", status
    # Positions carried over: final energy is no worse than at the switch point.
    assert job.atoms.get_potential_energy() <= energy_before + 1e-6


def test_set_fmax_loosening_lets_it_converge(emt_setup):
    session, sid, cid = emt_setup
    # Start with an impossibly tight target so it won't converge on its own.
    res = domain.start_relaxation(
        session, sid, cid, fmax=1e-8, steps=2000, step_delay=0.01
    )
    job = session.get_job(res["job_id"])

    assert wait_for(lambda: job.step >= 5)
    JobManager(session).steer(job, "set_fmax", fmax=0.1)

    done = wait_for(lambda: not job.is_active())
    assert done
    status = job.status_dict()
    assert status["status"] == "converged", status
    assert status["max_force"] <= 0.1


def test_single_active_job_policy(emt_setup):
    session, sid, cid = emt_setup
    domain.start_relaxation(session, sid, cid, fmax=1e-8, steps=500, step_delay=0.02)
    import pytest

    with pytest.raises(RuntimeError):
        domain.start_relaxation(session, sid, cid, fmax=0.05, steps=10)


def test_md_runs_and_records_temperature(emt_setup):
    session, sid, cid = emt_setup
    res = domain.start_md(
        session, sid, cid, ensemble="NVE", temperature_K=300, steps=20, step_delay=0.005
    )
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: not job.is_active())
    assert job.status == "finished"
    assert any("temperature_K" in h for h in job.history)
