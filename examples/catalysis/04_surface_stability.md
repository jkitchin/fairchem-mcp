# Surface stability — phonons

**Goal:** check whether a relaxed surface is *dynamically* stable. Imaginary
phonon modes mean a soft mode / reconstruction — the flat surface is not a true
minimum. Relax the slab, then run a finite-displacement phonon calculation and
read the gamma-point frequencies.

```text
attach_emt()                                          -> calc_1
#   (or load_model(model="uma-s-1p1", task="oc20") -> calc_1)

execute('''
from ase.build import fcc111
from ase.constraints import FixAtoms
slab = fcc111("Pt", size=(1,1,3), vacuum=8.0)
slab.set_constraint(FixAtoms(mask=[a.tag >= 2 for a in slab]))
sid = session.add_structure(slab)
''')
start_relaxation("struct_1", "calc_1", fmax=0.02)     -> job_1   # wait: converged

# 1 + 6*natoms force evaluations; poll for progress.
start_phonons("struct_1", "calc_1", supercell=[2,2,1], delta=0.03)  -> job_2
get_status("job_2")    -> {status:"running", done:31, total:73, fraction:0.42}
get_status("job_2")    -> {status:"finished", ...}

get_results("job_2")
#   {result:{gamma_frequencies_THz:[...], n_imaginary_modes:2,
#            min_frequency_THz:-1.90, stable:false, supercell:[2,2,1]}}
```

`stable:false` with imaginary modes flags a soft mode (possible reconstruction).
For a clean **stable** reference, the bulk metal gives the 3 acoustic modes ≈ 0
at gamma and `n_imaginary_modes:0` (run the script with `SURFACE=0`).

The `Phonons` object is bound as `ph` in the namespace, so the agent can pull a
full band structure or DOS through the escape hatch:

```text
execute("bs = ph.get_band_structure(...)")   # or inspect_expr(...) on ph
```

**Runnable version:** [`04_surface_stability.py`](04_surface_stability.py)

```
=== Dynamical stability: bulk fcc Pt ===   (SURFACE=0, EMT)
  gamma freqs (THz)   : [-0.0, 0.0, 0.0]
  imaginary modes     : 0
  dynamically stable  : True
```

**Caveat:** EMT is qualitative and small slabs can show spurious soft modes; use a
UMA model and a converged supercell for production stability checks.
