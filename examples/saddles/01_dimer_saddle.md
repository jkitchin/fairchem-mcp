# Transition state — dimer method (Hessian-free)

**Goal:** find a transition state from a single starting structure, using only
forces. The dimer method (`start_saddle_search`, ASE `MinModeTranslate`) climbs
the lowest-curvature mode and relaxes the rest — no Hessian, so it tolerates the
float32 noise of ML potentials. Here: an Al adatom hopping from a hollow site to
the bridge site (the index-1 saddle) on Al(100).

```text
attach_emt()                                          -> calc_1

# Relax the adatom into the hollow-site minimum.
execute('''
from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
slab = fcc100("Al", size=(2,2,3), vacuum=10.0)
z = slab.positions[:,2].mean()
slab.set_constraint(FixAtoms(mask=[p[2] < z for p in slab.positions]))
add_adsorbate(slab, "Al", 1.7, "hollow")
sid = session.add_structure(slab)
''')
start_relaxation("struct_1", "calc_1", fmax=0.05)     -> job_1   # wait: converged

# Dimer search, seeded toward the bridge site (nudge the adatom in +x).
start_saddle_search("struct_1", "calc_1",
    displacement_vector=[...0, 0.3 on the adatom...], fmax=0.05) -> job_2

# The live curvature is surfaced while it runs — watch it cross zero.
get_status("job_2")    -> {status:"running", curvature:+0.12, step:3, ...}
get_status("job_2")    -> {status:"running", curvature:-0.33, step:9, ...}

get_results("job_2")
#   {result:{is_index1_saddle:true, curvature:-0.33, max_force:0.04,
#            energy:..., structure_id:"struct_2"}}
```

**Why the dimer here:** single-ended (you don't need the *product* state, unlike
NEB) and Hessian-free, so it stays stable on noisy ML forces. The converged
`curvature < 0` is the proof you landed on an index-1 saddle, not a shoulder.

**Steering payoff:** `curvature` and `max_force` stream in the live status, so the
agent can tell a genuine saddle approach (curvature going negative) from a search
wandering off, and `abort`/`set_fmax` accordingly.

**If you omit `displacement_vector`:** a random kick of `displace_magnitude` Å is
applied to the unconstrained atoms — fine when you don't have a guess for the
reaction direction, but a physically-motivated seed converges far more reliably.

**Runnable version:** [`01_dimer_saddle.py`](01_dimer_saddle.py)

```
=== Al adatom diffusion TS on Al(100) — dimer ===   (EMT, qualitative)
  index-1 saddle?   : True
  curvature         : -0.429 eV/Ang^2  (negative = TS)
  max force         : 0.0446 eV/Ang
  barrier (E_TS-E0) : 0.234 eV
```

**Tip:** the dimer finds *one* saddle near your seed. To enumerate *several*
saddles classified by Morse index, use [`02_pounce_saddles`](02_pounce_saddles.md).
To refine with internal coordinates and an approximate Hessian, use
[`03_sella_saddle`](03_sella_saddle.md).
