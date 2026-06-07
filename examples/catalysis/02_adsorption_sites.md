# Adsorption-site search — H on Pt(111)

**Goal:** discover the *distinct* binding sites of an adsorbate (top, bridge,
fcc-hollow, hcp-hollow). Each site is a separate local minimum of the PES, so
this is a multi-minimum search — exactly what `start_minima_search` does.

It relaxes repeatedly from the starting geometry on a PES **biased to repel the
minima already found** (`kernel="flooding"`: Gaussian bumps), polishing each
escape on the true PES, and registers every distinct site as its own structure.

```text
attach_emt()                                          -> calc_1
#   (or load_model(model="uma-s-1p1", task="oc20") -> calc_1)

# Freeze the slab so only the adatom's landscape is explored.
execute('''
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms
slab = fcc111("Pt", size=(3,3,3), vacuum=10.0)
slab.set_constraint(FixAtoms(indices=list(range(len(slab)))))
add_adsorbate(slab, "H", height=1.5, position="ontop")
sid = session.add_structure(slab)
''')

start_minima_search("struct_1", "calc_1",
    n_minima=4, kernel="flooding", sigma=0.8, amplitude=1.0,
    fmax=0.02, energy_tol=0.03, rmsd_tol=0.4)         -> job_1

# Watch it accumulate sites:
get_status("job_1")    -> {status:"running", n_found:2, target:4, phase:"escape#2", ...}
get_status("job_1")    -> {status:"converged", n_found:4, ...}

# The distinct sites, each a registered structure, sorted by energy:
get_results("job_1")
#   {result:{n_found:4, minima:[{structure_id:"struct_2", energy:...}, ...]}}
```

**Knobs that matter:**
- `kernel` — `"flooding"` (smooth Gaussian bumps; default) or `"deflation"`
  (inverse-distance poles; sharper escape).
- `sigma` (Å) / `amplitude` (eV) — bump width / height. Roughly: `sigma` ≈ the
  spacing between sites, `amplitude` ≈ the lateral corrugation to clear.
- `energy_tol` (eV) + `rmsd_tol` (Å) — two geometries are "the same site" only if
  **both** their energies and their RMSD are within tolerance.

> **Free clusters / nanoparticles instead of a frozen slab?** The frozen slab here
> pins the frame, so the default `kernel="flooding"` + `comparator="rmsd"` works.
> A *free* cluster can rotate to dodge a spatial bump and rotated copies look
> distinct under raw RMSD — so use `kernel="basinhopping"` with
> `comparator="fingerprint"` (rotation/translation/permutation-invariant) there.

**Runnable version:** [`02_adsorption_sites.py`](02_adsorption_sites.py)

```
=== H on Pt(111): found 4 distinct site(s) ===   (EMT, qualitative)
  1. struct_5  E = 5.814 eV (+0.000)   hollow
  ...
  4. struct_2  E = 5.860 eV (+0.046)   ontop   (least stable)
```

**Design note:** this reuses the deflation/flooding *escape mechanism* from
POUNCE's `find_minima`, but drives each escape with ASE's gradient optimizers
(the right inner solver for a PES) — so every escape relaxation is a normal
steerable job. See the repo README's "Finding multiple relaxed geometries".

**Next step:** the lowest two sites become the endpoints for a diffusion-barrier
NEB (example 3).
