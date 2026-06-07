"""Shared helpers for the saddle-search examples.

Both examples run out-of-the-box with the fast **EMT** calculator (no GPU, no
model download). EMT energies are only qualitatively meaningful — for real
numbers, swap in a FAIRChem UMA model by setting ``FAIRCHEM_MCP_EXAMPLE_MODEL``
(see the catalysis examples' note on float32 noise and saddle searches).

These scripts call the same ``fairchem_mcp.domain`` functions the MCP tools wrap,
so the flow mirrors exactly what an agent does through the server (see the
matching ``*.md`` walkthrough for the tool-call version).
"""

from __future__ import annotations

import os
import time

from ase.build import add_adsorbate, fcc100
from ase.constraints import FixAtoms

from fairchem_mcp import domain
from fairchem_mcp.session import Session

# Set FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p1 to run on a real model.
_MODEL = os.environ.get("FAIRCHEM_MCP_EXAMPLE_MODEL")
_TASK = os.environ.get("FAIRCHEM_MCP_EXAMPLE_TASK", "oc20")


def calculator(session: Session) -> str:
    """Return a calculator_id — EMT by default, UMA if the env var is set."""
    if _MODEL:
        info = domain.load_model(session, model=_MODEL, task=_TASK)
        print(f"[calculator] UMA {_MODEL} (task={_TASK})")
        return info["calculator_id"]
    print("[calculator] EMT (fast/qualitative; set FAIRCHEM_MCP_EXAMPLE_MODEL for UMA)")
    return domain.attach_emt(session)["calculator_id"]


def wait(session: Session, job_id: str, timeout: float = 300.0):
    """Block until a background job finishes; return the finished Job."""
    job = session.get_job(job_id)
    t0 = time.time()
    while job.is_active():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"job {job_id} did not finish within {timeout}s")
        time.sleep(0.1)
    return job


def relaxed_adatom(session: Session, calculator_id: str):
    """Relax an Al adatom into a hollow site on a frozen Al(100) slab.

    The hollow site is a minimum; the bridge site between two hollows is the
    index-1 saddle (the diffusion transition state). Returns ``(structure_id,
    energy_of_minimum)``. This is the common starting point for both searches.
    """
    slab = fcc100("Al", size=(2, 2, 3), vacuum=10.0)
    zmean = slab.positions[:, 2].mean()
    slab.set_constraint(FixAtoms(mask=[p[2] < zmean for p in slab.positions]))
    add_adsorbate(slab, "Al", 1.7, "hollow")

    sid = session.add_structure(slab)
    job = wait(
        session,
        domain.start_relaxation(session, sid, calculator_id, fmax=0.05, steps=200)["job_id"],
    )
    return sid, job.status_dict()["energy"]
