"""Shared helpers for the catalysis examples.

Every example runs out-of-the-box with the fast **EMT** calculator (no GPU, no
model download) so you can see the workflow immediately. EMT energies are only
qualitatively meaningful — for real catalysis numbers, swap in a FAIRChem UMA
model with the catalysis task by replacing the ``calculator(session)`` call:

    # EMT (default, fast, qualitative):
    cid = domain.attach_emt(session)["calculator_id"]

    # UMA, oc20 task (inorganic catalysis / adsorption):
    cid = domain.load_model(session, model="uma-s-1p1", task="oc20")["calculator_id"]

These scripts call the same ``fairchem_mcp.domain`` functions the MCP tools wrap,
so the flow mirrors exactly what an agent does through the server (see the
matching ``*.md`` walkthrough for the tool-call version).
"""

from __future__ import annotations

import os
import time

from fairchem_mcp import domain
from fairchem_mcp.session import Session

# Set FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p1 to run the examples on a real model.
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


def energy(session: Session, structure_id: str) -> float:
    """Potential energy of a (relaxed) structure; its calculator is still attached."""
    return float(session.get_structure(structure_id).get_potential_energy())
