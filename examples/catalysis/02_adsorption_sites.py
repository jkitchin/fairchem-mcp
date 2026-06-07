"""Find the distinct adsorption sites of an H adatom on Pt(111).

A surface offers several binding sites (top, bridge, fcc-hollow, hcp-hollow) —
each a *distinct local minimum* of the PES. We freeze the slab so only the H atom
moves, then use ``start_minima_search`` (flooding) to discover the sites: it
relaxes repeatedly on a PES biased to repel the minima already found, polishing
each escape on the true PES. Every distinct site is registered as its own
structure and reported with its energy (lowest = most stable site).

Run:  python examples/catalysis/02_adsorption_sites.py
"""

from __future__ import annotations

from ase.build import add_adsorbate, fcc111
from ase.constraints import FixAtoms

from _common import calculator, wait
from fairchem_mcp import domain
from fairchem_mcp.session import Session


def main() -> None:
    session = Session()
    cid = calculator(session)

    # Freeze the whole slab so the search explores only the adatom's landscape.
    slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)
    slab.set_constraint(FixAtoms(indices=list(range(len(slab)))))
    add_adsorbate(slab, "H", height=1.5, position="ontop")
    sid = session.add_structure(slab)

    res = domain.start_minima_search(
        session, sid, cid,
        n_minima=4, kernel="flooding",
        sigma=0.8, amplitude=1.0,        # bump ~0.8 Å wide, 1 eV tall
        optimizer="FIRE", fmax=0.02, steps=200,
        energy_tol=0.03, rmsd_tol=0.4,   # sites differ by > 0.4 Å (H only)
    )
    job = wait(session, res["job_id"])

    minima = job.result["minima"]
    print(f"\n=== H on Pt(111): found {len(minima)} distinct site(s) ===")
    e0 = minima[0]["energy"]
    for rank, m in enumerate(minima, 1):
        atoms = session.get_structure(m["structure_id"])
        x, y, z = atoms.get_positions()[-1]  # the H atom (added last)
        rel = m["energy"] - e0
        print(
            f"  {rank}. {m['structure_id']:>9}  E = {m['energy']:8.3f} eV "
            f"(+{rel:5.3f})   H @ (x={x:5.2f}, y={y:5.2f}, z={z:4.2f}) Å"
        )
    print("\nThe most stable site is listed first. Each site is a registered")
    print("structure you can feed to start_neb (diffusion) or start_phonons.")


if __name__ == "__main__":
    main()
