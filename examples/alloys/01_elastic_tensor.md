# Elastic tensor & mechanical moduli

**Goal:** from a relaxed crystal, measure the full elastic stiffness tensor and
the engineering moduli derived from it. `start_elastic_scan` strains the cell by a
few magnitudes in each of the six Voigt directions, reads the resulting stress,
and least-squares fits `C_ij = dσ_i/dε_j`. Voigt-Reuss-Hill averaging of `C` then
gives the polycrystalline bulk/shear/Young's moduli, and the eigenvalues of `C`
give a Born mechanical-stability verdict.

Elastic constants are only defined **about the energy minimum**, so relax the cell
first.

```text
attach_emt()                                          -> calc_1
execute('from ase.build import bulk; session.add_structure(bulk("Cu","fcc",a=3.6,cubic=True))')

# 1. Relax to the equilibrium lattice constant (cell + ions).
start_relaxation("struct_1", "calc_1", relax_cell=True, fmax=0.01)  -> job_1  # wait

# 2. Stress-vs-strain elastic scan (6 directions x n_strains deformations).
start_elastic_scan("struct_1", "calc_1", n_strains=7, max_strain=0.01) -> job_2
get_status("job_2")   -> {status:"running", done:18, total:42, fraction:0.43, ...}

get_results("job_2")
#   {result:{ C_GPa:[[...6x6...]],
#             bulk_modulus_GPa:134.1, shear_modulus_GPa:56.7,
#             youngs_modulus_GPa:149.0, poisson_ratio:0.315,
#             pugh_ratio_G_over_K:0.423, mechanically_stable:true,
#             C_eigenvalues_GPa:[...] }}
```

**What the numbers mean**
- **C11, C12, C44** — the three independent stiffnesses of a cubic crystal.
- **K, G, E** — bulk, shear, Young's moduli (Voigt-Reuss-Hill average), the
  polycrystalline engineering numbers.
- **Poisson ratio** — lateral vs. axial strain.
- **Pugh ratio G/K** — a ductility proxy: **< ~0.57 → ductile**, above → brittle.
- **mechanically_stable** — Born criterion: `C` is positive-definite (all
  eigenvalues > 0). A negative eigenvalue means the structure isn't a true
  mechanical minimum.

**Steering payoff:** the scan is a background job that reports `done`/`total`, so
the agent can watch progress and `abort`/`pause` between deformations — useful when
each stress evaluation is an expensive ML inference.

**Relax the ions too?** Pass `relax_ions=True` to relax internal coordinates at
each strained cell (the clamped-ion correction). It matters for multi-atom bases
with internal degrees of freedom; for a monatomic fcc metal it's negligible and
slower.

**Runnable version:** [`01_elastic_tensor.py`](01_elastic_tensor.py)

```
=== fcc Cu — elastic tensor (EMT, qualitative) ===
  C11, C12, C44     : 172.0, 115.1, 89.6 GPa
  bulk modulus  K   : 134.1 GPa  (Hill)
  shear modulus G   : 56.7 GPa
  Young's modulus E : 149.0 GPa
  Poisson ratio     : 0.315
  Pugh ratio G/K    : 0.423  (ductile)
  Born stable?      : True  (eigs > 0: True)
```

EMT Cu lands remarkably close to experiment here (C11≈168, C12≈121, C44≈75 GPa;
K≈140; ν≈0.34), and the Pugh ratio correctly flags copper as ductile. For other
systems EMT is only qualitative — point it at a UMA model for real numbers.

**Next:** [`02_convex_hull`](02_convex_hull.md) screens which compositions are even
stable; [`03_alloy_design_loop`](03_alloy_design_loop.md) combines stability and
stiffness into a design screen.
