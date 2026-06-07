"""Transition state with Sella (rational-function optimizer + approx. Hessian).

Sella is an ASE optimizer that climbs toward an order-k saddle (1 = transition
state) using a partitioned rational-function step on an approximate Hessian it
refines as it goes. It needs no product state (unlike NEB) and no full Hessian
(unlike POUNCE), converges in few force calls, and — because it is an ASE
Optimizer — is fully steerable in the same loop as a relaxation. The lowest
Hessian eigenvalue is reported live so you can watch it cross zero.

Same Al-adatom hop as the other two examples, so you can compare convergence.

Requires `sella` (pip install sella).

Run:  python examples/saddles/03_sella_saddle.py
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

    # Seed toward the bridge site, same nudge as the dimer example.
    dvec = np.zeros((natoms, 3))
    dvec[-1, 0] = 0.3

    res = domain.start_sella_search(
        session, sid, cid, order=1,
        displacement_vector=dvec.ravel().tolist(),
        fmax=0.05, steps=200, step_delay=0.02,
    )
    job = session.get_job(res["job_id"])

    # The lowest Hessian eigenvalue streams in the live status — watch it go negative.
    while job.step < 3 and job.is_active():
        pass
    low = job.status_dict().get("lowest_eigenvalue")
    if low is not None:
        print(f"[watch] lowest eigenvalue = {low:+.3f} eV/Ang^2 "
              f"({'saddle-like' if low < 0 else 'still in a basin'})")

    wait(session, res["job_id"], timeout=120)

    r = job.result
    print("\n=== Al adatom diffusion TS on Al(100) — Sella ===")
    print(f"  status              : {job.status_dict()['status']}")
    print(f"  order requested     : {r['order_requested']}")
    print(f"  negative eigenvalues: {r['n_negative_eigenvalues']}  "
          f"(want {r['order_requested']} for the target saddle)")
    print(f"  is order-1 saddle?  : {r['is_target_order_saddle']}")
    print(f"  lowest eigenvalue   : {r['lowest_eigenvalue']:+.3f} eV/Ang^2")
    print(f"  max force           : {r['max_force']:.4f} eV/Ang")
    print(f"  barrier (E_TS-E0)   : {r['energy'] - e_min:.3f} eV")
    print("\n(EMT is qualitative; set FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p1 for real numbers.)")


if __name__ == "__main__":
    main()
