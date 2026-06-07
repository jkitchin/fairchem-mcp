"""Transition state by the dimer method (Hessian-free, single-ended).

The dimer method (ASE ``MinModeTranslate``) climbs the lowest-curvature mode and
relaxes the rest, using only forces — no Hessian — so it is robust with noisy ML
potentials. Starting from a relaxed Al adatom in a hollow site on Al(100), we
nudge it toward the bridge site and converge onto the diffusion transition state.
A negative converged curvature confirms an index-1 saddle.

Run:  python examples/saddles/01_dimer_saddle.py
"""

from __future__ import annotations

import numpy as np

from _common import calculator, relaxed_adatom, wait
from fairchem_mcp import domain
from fairchem_mcp.session import Session


def main() -> None:
    session = Session()
    cid = calculator(session)

    sid, e_min = relaxed_adatom(session, cid)
    natoms = len(session.get_structure(sid))

    # Seed the search: nudge the (last) adatom toward the bridge site.
    dvec = np.zeros((natoms, 3))
    dvec[-1, 0] = 0.3

    res = domain.start_saddle_search(
        session, sid, cid,
        displacement_vector=dvec.ravel().tolist(),
        fmax=0.05, steps=300, step_delay=0.02,
    )
    job = session.get_job(res["job_id"])

    # The live curvature is surfaced while the dimer runs — watch it go negative.
    while job.step < 2 and job.is_active():
        pass
    curv = job.status_dict().get("curvature")
    if curv is not None:
        print(f"[watch] early curvature = {curv:+.3f} eV/Ang^2 "
              f"({'saddle-like' if curv < 0 else 'still uphill'})")

    wait(session, res["job_id"], timeout=120)

    r = job.result
    print("\n=== Al adatom diffusion TS on Al(100) — dimer ===")
    print(f"  status            : {job.status_dict()['status']}")
    print(f"  index-1 saddle?   : {r['is_index1_saddle']}")
    print(f"  curvature         : {r['curvature']:+.3f} eV/Ang^2  (negative = TS)")
    print(f"  max force         : {r['max_force']:.4f} eV/Ang")
    print(f"  barrier (E_TS-E0) : {r['energy'] - e_min:.3f} eV")
    print("\n(EMT is qualitative; set FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p1 for real numbers.)")


if __name__ == "__main__":
    main()
