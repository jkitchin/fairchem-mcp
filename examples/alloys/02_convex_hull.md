# Formation-energy convex hull (phase stability)

**Goal:** given several compositions in a chemical system, decide which ones are
thermodynamically **stable**. `start_convex_hull` evaluates each structure,
references it to the pure elements to get a **formation energy per atom**
`E_f = (E − Σ nᵢ μᵢ)/N`, and builds the **lower convex hull** of `E_f` vs.
composition. Phases that sit on the hull are stable; a phase's vertical distance
above it (`energy_above_hull_per_atom`) is how metastable it is.

```text
attach_emt()                                          -> calc_1
# Build the Cu-Au line: pure Cu, pure Au, and Cu3Au / CuAu / CuAu3 orderings.
execute('''
from ase.build import bulk
def fcc4(a, syms):
    c = bulk("Cu","fcc",a=a,cubic=True); c.symbols = syms; return c
for a, syms in [(3.60,["Cu"]*4),(4.08,["Au"]*4),(3.72,["Au","Cu","Cu","Cu"]),
                (3.84,["Au","Au","Cu","Cu"]),(3.96,["Au","Au","Au","Cu"])]:
    session.add_structure(fcc4(a, syms))
''')

start_convex_hull(["struct_1",...,"struct_5"], "calc_1",
                  relax=True, relax_cell=True,
                  labels=["Cu","Au","Cu3Au","CuAu","CuAu3"])   -> job_1
get_status("job_1")  -> {status:"running", done:3, total:5, ...}

get_results("job_1")
#   {result:{ elements:["Au","Cu"],
#             references_eV_per_atom:{Cu:..., Au:...},
#             phases:[{label:"Cu3Au", composition:{Au:0.25,Cu:0.75},
#                      formation_energy_per_atom:-0.0103, energy_above_hull_per_atom:0.0,
#                      on_hull:true}, ...],
#             stable_phases:["Cu","Au","Cu3Au","CuAu"] }}
```

**Inputs.** Pass the candidate `structure_ids` spanning the system. The pure
elements must be present (they define the references) **or** supplied as
`references={element: energy_per_atom}`. `relax=True` relaxes ions at each fixed
cell; add `relax_cell=True` to relax the lattice too (closer to each phase's true
energy). It works for any number of elements — the hull math is a small linear
program per phase.

**Runnable version:** [`02_convex_hull.py`](02_convex_hull.py)

```
=== Cu-Au convex hull (EMT, qualitative) ===
  elements   : ['Au', 'Cu']
  references : Cu=-0.007 eV/atom, Au=-0.000 eV/atom
  phase     x_Au  Ef (eV/at)  above hull  stable
  Cu        0.00     +0.0000      0.0000  *
  Cu3Au     0.25     -0.0103      0.0000  *
  CuAu      0.50     -0.0078      0.0000  *
  CuAu3     0.75     +0.0072      0.0111
  Au        1.00     +0.0000      0.0000  *

  stable phases (on hull): ['Cu', 'Au', 'Cu3Au', 'CuAu']
```

**Reading it.** Cu₃Au and CuAu have **negative** formation energies and land on the
hull (Cu-Au is a real ordering system — these are the classic L1₂/L1₀ phases),
while CuAu₃ sits ~0.011 eV/atom above it and is metastable. The `on_hull` set is
exactly the phases a design screen should carry forward.

**A sharp caveat about the potential.** Cu-Au works here *because* ASE's built-in
EMT was validated on it. EMT is a single-element-fit potential and its alloy
cross-interaction is crude — for many binaries it gets the sign wrong (Ni-Al, for
instance, comes out with *positive* formation energies even when fully relaxed to
zero stress, so its intermetallics wrongly appear unstable). For quantitative
alloy energetics use `asap3`'s alloy-parameterized EMT, or — for any system — a
UMA model (`FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p2 FAIRCHEM_MCP_EXAMPLE_TASK=omat`).
The *workflow* is identical; only the energies get trustworthy.

**Steering payoff:** a background job with `done`/`total`, abortable between
structures — handy when screening dozens of candidate compositions on a real model.

**Next:** [`03_alloy_design_loop`](03_alloy_design_loop.md) feeds these stable
phases into an elastic-property screen.
