# Transition state — Sella (RFO + refined approximate Hessian)

**Goal:** a third route to a transition state, complementing the dimer and
POUNCE. `start_sella_search` wraps [Sella](https://github.com/zadorlab/sella), an
ASE optimizer that climbs toward an order-`k` saddle (1 = TS) using a partitioned
rational-function step on an approximate Hessian it **refines as it goes**. It
needs no product state (unlike NEB) and no full Hessian (unlike POUNCE), and
because it *is* an ASE optimizer it runs in the same steerable loop as a
relaxation. Same Al adatom hop as the other two examples.

Requires `sella` (`pip install sella`).

```text
attach_emt()                                          -> calc_1

# Relax the adatom into the hollow-site minimum (same as the other examples).
start_relaxation("struct_1", "calc_1", fmax=0.05)     -> job_1   # converged

# Sella TS search (order=1), seeded toward the bridge site.
start_sella_search("struct_1", "calc_1", order=1,
    displacement_vector=[...0, 0.3 on the adatom...], fmax=0.05) -> job_2

# The lowest Hessian eigenvalue streams live — watch it cross zero.
get_status("job_2")    -> {status:"running", lowest_eigenvalue:+0.39, step:3, ...}
get_status("job_2")    -> {status:"running", lowest_eigenvalue:-0.39, step:8, ...}

get_results("job_2")
#   {result:{order_requested:1, n_negative_eigenvalues:1,
#            is_target_order_saddle:true, lowest_eigenvalue:-0.394,
#            max_force:0.048, energy:..., structure_id:"struct_2"}}
```

**Where Sella fits among the three routes:**

| | dimer | Sella | POUNCE |
|---|---|---|---|
| Hessian | none | approximate, refined online | full, finite-differenced |
| Force calls to converge | moderate | few (RFO is efficient) | many (multistart) |
| Output | one saddle | one order-`k` saddle | several, by Morse index |
| Internal coordinates | no | optional (`internal=True`) | no |
| Steerable mid-run | yes | yes | no |

**`internal=True`** switches Sella to automatic internal (bond/angle/dihedral)
coordinates — usually the better choice for **molecules**, where Cartesian steps
couple awkwardly. For a rigid slab the Cartesian default is fine.

**Steering payoff:** Sella keeps its approximate Hessian across a `set_fmax`, so
you can start loose, watch `lowest_eigenvalue` go negative (confirming you're on
the saddle's ridge), then tighten `fmax` without throwing away the curvature
information already learned.

**Runnable version:** [`03_sella_saddle.py`](03_sella_saddle.py)

```
=== Al adatom diffusion TS on Al(100) — Sella ===   (EMT, qualitative)
  order requested     : 1
  negative eigenvalues: 1  (want 1 for the target saddle)
  is order-1 saddle?  : True
  lowest eigenvalue   : -0.394 eV/Ang^2
  barrier (E_TS-E0)   : 0.234 eV
```

All three routes land on the same bridge transition state (barrier ≈ 0.23 eV on
EMT) — a good cross-check, and a template for picking the route that fits your
problem.
