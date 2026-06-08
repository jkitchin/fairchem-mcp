"""Elastic stiffness tensor and mechanical moduli of a crystal.

Relax a bulk crystal at its equilibrium cell, then strain it in each of the six
Voigt directions and read the stress to fit the 6x6 stiffness tensor C_ij. From C
we get the Voigt-Reuss-Hill bulk/shear/Young's moduli, the Poisson and Pugh
ratios (Pugh G/K < ~0.57 suggests a ductile metal), and a Born mechanical-
stability check (C must be positive-definite).

Run:  python examples/alloys/01_elastic_tensor.py
"""

from __future__ import annotations

from ase.build import bulk

from _common import calculator, wait
from fairchem_mcp import domain
from fairchem_mcp.session import Session


def main() -> None:
    session = Session()
    cid = calculator(session)

    # Start from fcc Cu and let the cell relax to its equilibrium lattice constant
    # first — elastic constants are only defined about the energy minimum.
    sid = session.add_structure(bulk("Cu", "fcc", a=3.6, cubic=True))
    wait(session, domain.start_relaxation(
        session, sid, cid, relax_cell=True, fmax=0.01, steps=300)["job_id"])

    res = domain.start_elastic_scan(
        session, sid, cid, n_strains=7, max_strain=0.01, step_delay=0.0)
    job = wait(session, res["job_id"], timeout=180)
    r = job.result

    C = r["C_GPa"]
    print("\n=== fcc Cu — elastic tensor (EMT, qualitative) ===")
    print(f"  C11, C12, C44     : {C[0][0]:.1f}, {C[0][1]:.1f}, {C[3][3]:.1f} GPa")
    print(f"  bulk modulus  K   : {r['bulk_modulus_GPa']:.1f} GPa  (Hill)")
    print(f"  shear modulus G   : {r['shear_modulus_GPa']:.1f} GPa")
    print(f"  Young's modulus E : {r['youngs_modulus_GPa']:.1f} GPa")
    print(f"  Poisson ratio     : {r['poisson_ratio']:.3f}")
    print(f"  Pugh ratio G/K    : {r['pugh_ratio_G_over_K']:.3f}  "
          f"({'ductile' if r['pugh_ratio_G_over_K'] < 0.57 else 'brittle'})")
    print(f"  Born stable?      : {r['mechanically_stable']}  "
          f"(eigs > 0: {all(x > 0 for x in r['C_eigenvalues_GPa'])})")
    print("\n(EMT is qualitative; set FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p2 "
          "FAIRCHEM_MCP_EXAMPLE_TASK=omat for real numbers.)")


if __name__ == "__main__":
    main()
