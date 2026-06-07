# Enumerate saddles by Morse index — POUNCE `find_saddles`

**Goal:** find *several* saddle points and have each one **classified by Morse
index** (number of negative Hessian eigenvalues), instead of one saddle near a
seed. `start_pounce_saddles` wraps POUNCE's eigenvector-following (Cerjan-Miller)
multistart solver. Ask for `index=1` to get transition states; `index=2` for
second-order saddles, etc.

Requires `pounce-solver` (`pip install pounce-solver`).

```text
attach_emt()                                          -> calc_1

# Relax the adatom into the hollow-site minimum (same as the dimer example).
start_relaxation("struct_1", "calc_1", fmax=0.05)     -> job_1   # converged

# Eigenvector-following over just the adatom's coordinates.
start_pounce_saddles("struct_1", "calc_1",
    active_indices=[12], index=1, n_saddles=2,
    grad_tol=1e-3, displace_magnitude=0.25)            -> job_2

get_results("job_2")
#   {result:{n_found:2, solver_status:"target_reached", n_solves:4, saddles:[
#       {structure_id:"struct_2", morse_index:1, energy:...,
#        grad_norm:7.5e-5, eigenvalues:[-0.33, 1.05, 4.03]}, ...]}}
```

**How it differs from the dimer:**

| | dimer (`start_saddle_search`) | POUNCE (`start_pounce_saddles`) |
|---|---|---|
| Hessian | none (forces only) | full, by finite difference of forces |
| Output | one saddle near the seed | several, each labeled by Morse index |
| Classification | curvature sign | exact eigenvalue count (`index`) |
| Noise tolerance | high | needs loose `grad_tol` / float64 |
| Steerable mid-run | yes | no (POUNCE owns the multistart loop) |

**Active-atom subset matters.** The variables default to the unconstrained atoms,
but the full Hessian over a slab has near-zero/soft modes that pollute the
Morse-index count. Restricting `active_indices` to the reacting atom(s) keeps the
problem low-dimensional and the index counting clean.

**ML potentials.** The full Hessian is finite-differenced from forces, so float32
noise (~1e-3 eV/Å for UMA) makes the tight default tolerances unreachable: loosen
`grad_tol` (and `eig_tol`) and prefer a float64 model. EMT is smooth, so it
converges tight here.

**Runnable version:** [`02_pounce_saddles.py`](02_pounce_saddles.py)

```
=== Al adatom saddles on Al(100) — POUNCE find_saddles ===   (EMT, qualitative)
  solver status : target_reached  (4 solves)
  saddles found : 2
  [0] Morse index 1  E-E0=+0.261 eV  |grad|=7.5e-05  neg_eigs=1
  [1] Morse index 1  E-E0=+0.261 eV  |grad|=8.0e-06  neg_eigs=1
```

(Both hits are the same symmetry-equivalent bridge saddle — the multistart finds
it twice; `dedup` controls how aggressively duplicates are merged.)
