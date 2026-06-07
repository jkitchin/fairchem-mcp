"""Enumerate saddles by Morse index with POUNCE's find_saddles.

Where the dimer finds one saddle near a seed, POUNCE's eigenvector-following
(Cerjan-Miller) runs a multistart search and classifies every critical point it
finds by its Morse index (number of negative Hessian eigenvalues). Ask for
``index=1`` to get transition states explicitly labeled as such.

The optimization variables are the Cartesian coordinates of ``active_indices``
(default: the unconstrained atoms); the Hessian is a central finite difference of
the calculator forces. Same physical system as example 1 — the Al adatom hop.

Requires `pounce-solver` (pip install pounce-solver).

Run:  python examples/saddles/02_pounce_saddles.py
"""

from __future__ import annotations

from _common import calculator, relaxed_adatom, wait
from fairchem_mcp import domain
from fairchem_mcp.session import Session


def main() -> None:
    session = Session()
    cid = calculator(session)

    sid, e_min = relaxed_adatom(session, cid)
    adatom = len(session.get_structure(sid)) - 1  # the adsorbate is added last

    # Search over just the adatom's coordinates (keeps it low-dimensional and
    # avoids the slab's rigid/soft modes polluting the Morse-index count).
    res = domain.start_pounce_saddles(
        session, sid, cid,
        active_indices=[adatom], index=1, n_saddles=2,
        grad_tol=1e-3, max_step=0.15, dedup=0.1,
        displace_magnitude=0.25, seed=0,
    )
    job = session.get_job(res["job_id"])
    wait(session, res["job_id"], timeout=180)

    r = job.result
    print("\n=== Al adatom saddles on Al(100) — POUNCE find_saddles ===")
    print(f"  solver status : {r['solver_status']}  ({r['n_solves']} solves)")
    print(f"  saddles found : {r['n_found']}")
    for k, sad in enumerate(r["saddles"]):
        neg = sum(1 for e in sad["eigenvalues"] if e < 0)
        print(f"  [{k}] Morse index {sad['morse_index']}  "
              f"E-E0={sad['energy'] - e_min:+.3f} eV  "
              f"|grad|={sad['grad_norm']:.1e}  neg_eigs={neg}")
    print("\n(EMT is qualitative; for UMA loosen grad_tol and prefer a float64 model.)")


if __name__ == "__main__":
    main()
