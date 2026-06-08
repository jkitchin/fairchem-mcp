"""Shared helpers for the alloy / mechanical-properties examples.

Everything runs out-of-the-box on the fast **EMT** calculator (no GPU, no model
download). EMT is only qualitative — its alloy formation energies and elastic
constants are illustrative, not publication numbers — but the *workflow* is
exactly what you'd run on a real model: set ``FAIRCHEM_MCP_EXAMPLE_MODEL`` (e.g.
``uma-s-1p2`` with ``FAIRCHEM_MCP_EXAMPLE_TASK=omat``) and the same scripts give
meaningful numbers.

These scripts call the same ``fairchem_mcp.domain`` functions the MCP tools wrap,
so the flow mirrors what an agent does through the server (see the matching
``*.md`` walkthrough for the tool-call version).
"""

from __future__ import annotations

import os
import time

from ase.build import bulk

from fairchem_mcp import domain
from fairchem_mcp.session import Session

_MODEL = os.environ.get("FAIRCHEM_MCP_EXAMPLE_MODEL")
_TASK = os.environ.get("FAIRCHEM_MCP_EXAMPLE_TASK", "omat")


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


# --- a small Cu-Au chemical system, all EMT-supported -----------------------
# Pure fcc Cu and Au plus three ordered intermetallics on the fcc lattice. The
# cubic conventional cell has 4 sites; we decorate them to get each composition.
#
# Cu-Au (not Ni-Al) on purpose: ASE's built-in EMT is a single-element-fit
# potential whose alloy cross-interaction is crude, so it gets most binary metals
# qualitatively wrong (Ni-Al, e.g., comes out with *positive* formation energies).
# Cu-Au is the system Jacobsen's EMT parameters were validated against, and even
# the toy ASE EMT reproduces the correct sign here (Cu3Au / CuAu fall on the hull).
# For other systems use asap3's alloy-parameterized EMT or a UMA model.
def cu_au_system(session: Session) -> list:
    """Build (structure_id, label) pairs spanning the Cu-Au composition line.

    Returns a list of ``(structure_id, label)``: pure Cu, pure Au, and the
    L1_2/L1_0 orderings Cu3Au, CuAu, CuAu3. Lattice constants are rough
    Vegard-style guesses; the elastic/hull tools relax the cell from here.
    """
    a_cu, a_au = 3.60, 4.08

    def fcc4(a, symbols):
        cell = bulk("Cu", "fcc", a=a, cubic=True)  # 4-atom conventional fcc cell
        cell.symbols = symbols
        return cell

    specs = [
        ("Cu", fcc4(a_cu, ["Cu", "Cu", "Cu", "Cu"])),
        ("Au", fcc4(a_au, ["Au", "Au", "Au", "Au"])),
        ("Cu3Au", fcc4(0.75 * a_cu + 0.25 * a_au, ["Au", "Cu", "Cu", "Cu"])),
        ("CuAu", fcc4(0.5 * (a_cu + a_au), ["Au", "Au", "Cu", "Cu"])),
        ("CuAu3", fcc4(0.25 * a_cu + 0.75 * a_au, ["Au", "Au", "Au", "Cu"])),
    ]
    return [(session.add_structure(atoms), label) for label, atoms in specs]
