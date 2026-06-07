# Adsorption energy — CO on Pt(111)

**Goal:** compute `E_ads = E(slab+CO) − E(slab) − E(CO_gas)`. Negative ⇒ binding
is favorable.

The pattern is three relaxations on one resident calculator. Through the MCP
server an agent issues:

```text
# 1) Load the model once (resident for every call). For catalysis, task=oc20.
load_model(model="uma-s-1p1", task="oc20")          -> calc_1
#   (or attach_emt() -> calc_1 for a fast, qualitative demo)

# 2) Clean slab — Pt(111), bottom layers fixed. Build it via the escape hatch
#    (add_adsorbate / FixAtoms aren't in build_structure), then relax.
execute('''
from ase.build import fcc111
from ase.constraints import FixAtoms
slab = fcc111("Pt", size=(2,2,3), vacuum=10.0)
slab.set_constraint(FixAtoms(mask=[a.tag >= 2 for a in slab]))
sid_slab = session.add_structure(slab)
''')
start_relaxation("struct_1", "calc_1", fmax=0.05)   -> job_1
get_status("job_1")                                  -> {status:"converged", ...}
inspect_expr("session.get_structure('struct_1').get_potential_energy()")  # E_slab

# 3) Gas-phase CO in a box, relaxed the same way            -> E_co
# 4) Slab + CO on a top site (add_adsorbate, mol_index=0 = C down), relaxed -> E_slab_CO

# 5) The arithmetic, in the live namespace:
inspect_expr("E_slab_CO - E_slab - E_co")            -> E_ads
```

**Why the server shines here:** the model is loaded **once** and reused across all
three relaxations (on CPU, model load dominates a batch script's runtime), and
`get_status` lets the agent watch each relaxation converge and intervene if one
stalls.

**Runnable version:** [`01_adsorption_energy.py`](01_adsorption_energy.py)

```
=== CO on Pt(111) ===   (EMT, qualitative)
  E_ads ≈ -0.36 eV   (binds)
```

**Extend it:** loop the adsorbate over `top`/`bridge`/`fcc`/`hollow` sites and
compare `E_ads`, or feed the relaxed `slab+CO` into `start_phonons` to get the
vibrational modes of the adsorbate (example 4).
