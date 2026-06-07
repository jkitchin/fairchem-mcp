"""Adsorption energy of CO on Pt(111).

    E_ads = E(slab+CO) - E(slab) - E(CO_gas)

A negative E_ads means binding is favorable. We relax three systems on the same
resident calculator: the clean slab, the gas-phase CO, and the slab with CO
adsorbed on a top site. Bottom layers are fixed so the slab acts as a substrate.

Run:  python examples/catalysis/01_adsorption_energy.py
"""

from __future__ import annotations

from ase import Atoms
from ase.build import add_adsorbate, fcc111
from ase.constraints import FixAtoms

from _common import calculator, energy, wait
from fairchem_mcp import domain
from fairchem_mcp.session import Session


def _pt_slab() -> Atoms:
    slab = fcc111("Pt", size=(2, 2, 3), vacuum=10.0)
    # Fix the bottom two layers (tags 2 and 3); the top layer (tag 1) relaxes.
    slab.set_constraint(FixAtoms(mask=[atom.tag >= 2 for atom in slab]))
    return slab


def main() -> None:
    session = Session()
    cid = calculator(session)

    # 1) Clean slab.
    sid_slab = session.add_structure(_pt_slab())
    wait(session, domain.start_relaxation(session, sid_slab, cid, fmax=0.05, steps=200)["job_id"])
    e_slab = energy(session, sid_slab)

    # 2) Gas-phase CO (C-down geometry, isolated in a box).
    co = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.13]])
    co.center(vacuum=6.0)
    sid_co = session.add_structure(co)
    wait(session, domain.start_relaxation(session, sid_co, cid, fmax=0.05, steps=200)["job_id"])
    e_co = energy(session, sid_co)

    # 3) Slab + CO on a top site (C binds to Pt).
    slab_co = _pt_slab()
    co_ads = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.13]])
    add_adsorbate(slab_co, co_ads, height=2.0, position="ontop", mol_index=0)
    sid_sc = session.add_structure(slab_co)
    wait(session, domain.start_relaxation(session, sid_sc, cid, fmax=0.05, steps=300)["job_id"])
    e_slab_co = energy(session, sid_sc)

    e_ads = e_slab_co - e_slab - e_co
    print("\n=== CO on Pt(111) ===")
    print(f"  E(slab)        = {e_slab:9.3f} eV")
    print(f"  E(CO gas)      = {e_co:9.3f} eV")
    print(f"  E(slab+CO)     = {e_slab_co:9.3f} eV")
    print(f"  E_ads          = {e_ads:9.3f} eV   ({'binds' if e_ads < 0 else 'unfavorable'})")
    print("\n(EMT is qualitative; set FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p1 for real numbers.)")


if __name__ == "__main__":
    main()
