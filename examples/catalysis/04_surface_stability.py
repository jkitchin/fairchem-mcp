"""Dynamical stability of a relaxed surface (phonons).

A catalyst surface is only meaningful if it is dynamically stable: no imaginary
phonon modes. Imaginary modes signal a soft mode / reconstruction. We relax a
Pt(111) slab, then run a finite-displacement phonon calculation and report the
gamma-point frequencies and any imaginary (unstable) modes.

For a known-stable reference, the bulk metal (set SURFACE=0 below) gives the
3 acoustic modes ~0 at gamma and nothing imaginary.

Run:  python examples/catalysis/04_surface_stability.py
"""

from __future__ import annotations

import os

from ase.build import bulk, fcc111
from ase.constraints import FixAtoms

from _common import calculator, wait
from fairchem_mcp import domain
from fairchem_mcp.session import Session

SURFACE = os.environ.get("SURFACE", "1") != "0"


def main() -> None:
    session = Session()
    cid = calculator(session)

    if SURFACE:
        atoms = fcc111("Pt", size=(1, 1, 3), vacuum=8.0)
        atoms.set_constraint(FixAtoms(mask=[a.tag >= 2 for a in atoms]))
        label = "Pt(111) slab"
    else:
        atoms = bulk("Pt", "fcc", a=3.94)
        label = "bulk fcc Pt"

    sid = session.add_structure(atoms)
    wait(session, domain.start_relaxation(session, sid, cid, fmax=0.02, steps=200)["job_id"])

    res = domain.start_phonons(session, sid, cid, supercell=(2, 2, 1), delta=0.03)
    job = wait(session, res["job_id"], timeout=300)

    r = job.result
    print(f"\n=== Dynamical stability: {label} ===")
    print(f"  supercell             : {r['supercell']}")
    print(f"  gamma freqs (THz)     : {r['gamma_frequencies_THz']}")
    print(f"  min frequency (THz)   : {r['min_frequency_THz']}")
    print(f"  imaginary modes       : {r['n_imaginary_modes']}")
    print(f"  dynamically stable    : {r['stable']}")
    if not r["stable"]:
        print("  -> soft/imaginary mode: the surface may reconstruct.")
    print("\n(EMT is qualitative; set FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p1 for real numbers.)")


if __name__ == "__main__":
    main()
