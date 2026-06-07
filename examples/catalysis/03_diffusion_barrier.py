"""Diffusion barrier of an adatom hopping between hollow sites (NEB).

Surface diffusion is an elementary step in many catalytic mechanisms. We relax
two endpoints — an Au adatom in adjacent hollow sites on Al(100) — then run a
nudged elastic band between them. Near convergence we steer ``set_climb`` on, so
the climbing image lands exactly on the saddle point for an accurate barrier.

The same recipe works for any reaction coordinate (dissociation, hops, flips):
relax the two endpoints, then ``start_neb``.

Run:  python examples/catalysis/03_diffusion_barrier.py
"""

from __future__ import annotations

from ase.build import add_adsorbate, fcc100
from ase.constraints import FixAtoms

from _common import calculator, wait
from fairchem_mcp import domain
from fairchem_mcp.jobs import JobManager
from fairchem_mcp.session import Session


def main() -> None:
    session = Session()
    cid = calculator(session)

    slab = fcc100("Al", size=(2, 2, 3), vacuum=10.0)
    zmean = slab.positions[:, 2].mean()
    slab.set_constraint(FixAtoms(mask=[p[2] < zmean for p in slab.positions]))
    add_adsorbate(slab, "Au", 1.7, "hollow")

    initial = slab.copy()
    final = slab.copy()
    final.positions[-1, 0] += final.cell[0, 0] / 2  # adatom one hollow over

    sid_i = session.add_structure(initial)
    sid_f = session.add_structure(final)
    for sid in (sid_i, sid_f):
        wait(session, domain.start_relaxation(session, sid, cid, fmax=0.05, steps=200)["job_id"])

    res = domain.start_neb(
        session, sid_i, sid_f, cid,
        nimages=5, optimizer="LBFGS", fmax=0.05, steps=400, step_delay=0.02,
    )
    job = session.get_job(res["job_id"])

    # Turn the climbing image on once the band has taken a few steps.
    while job.step < 3 and job.is_active():
        pass
    if job.is_active():
        JobManager(session).steer(job, "set_climb", climb=True)
        print("[steer] climbing image enabled for an accurate saddle point")

    wait(session, res["job_id"], timeout=300)

    r = job.result
    print("\n=== Au adatom diffusion on Al(100) ===")
    print(f"  status          : {job.status_dict()['status']}")
    print(f"  forward barrier : {r['barrier']:.3f} eV")
    print(f"  reaction energy : {r['delta_E']:.3f} eV")
    print(f"  image energies  : {[round(e, 3) for e in r['energies']]}")
    print("\n(EMT is qualitative; set FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p1 for real numbers.)")


if __name__ == "__main__":
    main()
