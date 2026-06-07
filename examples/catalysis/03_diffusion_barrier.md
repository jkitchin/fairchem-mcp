# Diffusion / reaction barrier — NEB

**Goal:** the energy barrier for an elementary step (here an Au adatom hopping
between adjacent hollow sites on Al(100); the same recipe covers dissociation,
flips, hops). Relax the two endpoints, run a nudged elastic band, and steer the
climbing image on near convergence for an accurate saddle.

```text
attach_emt()                                          -> calc_1

# Build + relax the two endpoints (initial / final adatom positions).
execute('''
from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
slab = fcc100("Al", size=(2,2,3), vacuum=10.0)
z = slab.positions[:,2].mean()
slab.set_constraint(FixAtoms(mask=[p[2] < z for p in slab.positions]))
add_adsorbate(slab, "Au", 1.7, "hollow")
initial = slab.copy(); final = slab.copy()
final.positions[-1,0] += final.cell[0,0]/2
sid_i = session.add_structure(initial); sid_f = session.add_structure(final)
''')
start_relaxation("struct_1", "calc_1", fmax=0.05)     -> job_1   # wait: converged
start_relaxation("struct_2", "calc_1", fmax=0.05)     -> job_2   # wait: converged

# Nudged elastic band between the relaxed endpoints.
start_neb("struct_1", "struct_2", "calc_1",
    nimages=5, optimizer="LBFGS", fmax=0.05)          -> job_3

# Once the band has taken a few steps and is near converged, climb for the saddle:
get_status("job_3")    -> {status:"running", barrier:0.41, step:5, trend:{...}}
steer("job_3", "set_climb")                            # climbing image ON
get_status("job_3")    -> {status:"converged", barrier:0.37, ...}

get_results("job_3")
#   {result:{barrier:0.374, delta_E:-0.000, energies:[...], n_images:7}}
```

**Steering payoff:** `get_status` surfaces the running `barrier` and a convergence
`trend`, so the agent decides *when* to enable the climbing image (too early
destabilizes the band, too late wastes steps) — and can `switch_optimizer` or
`set_fmax` if it stalls, all on the live band.

**Runnable version:** [`03_diffusion_barrier.py`](03_diffusion_barrier.py)

```
=== Au adatom diffusion on Al(100) ===   (EMT, qualitative)
  forward barrier : 0.374 eV
  reaction energy : -0.000 eV   (symmetric hop)
```

**Tip:** get the endpoints from example 2's site search — the two most stable
adsorption sites are the natural initial/final states for a diffusion path.
