# Phonon dispersion — fcc Cu

A finite-displacement phonon calculation on a bulk crystal. `start_phonons`
displaces the atoms in a supercell, collects the forces, and builds the force
constants. Those same constants give two things at once:

- the **Γ-point stability check** — gamma frequencies, the count of imaginary
  modes, and a `stable` flag (a crystal with imaginary modes is dynamically
  unstable); and
- a **full phonon dispersion** ω(**q**) sampled along a Brillouin-zone k-path,
  and a **phonon density of states** sampled on a q-grid — both computed by
  Fourier-summing the *already-computed* force constants, so they cost **no extra
  force evaluations**.

The k-path is chosen automatically from the cell's Bravais lattice (ASE's
Setyawan–Curtarolo special points), so fcc Cu gets a sensible
**Γ–X–W–K–Γ–L–U–W–L–K | U–X** traverse without you naming one. Pass `band_path`
(e.g. `"GXWLG"`) to override and `band_points` to set the sampling density.

Runs out of the box on the fast **EMT** calculator (no GPU, no model download).
EMT is qualitative; point it at a UMA model for publication numbers — UMA puts
the zone-boundary maximum at ≈ 7.5 THz (250 cm⁻¹), matching experiment for Cu.

## What `get_results` returns

```jsonc
{
  "gamma_frequencies_THz": [-0.0, -0.0, 0.0],   // 3 acoustic modes at Γ
  "n_imaginary_modes": 0,
  "stable": true,
  "supercell": [4, 4, 4],
  "dispersion": {
    "path": "GXWKGLUWLK,UX",
    "n_kpoints": 200,
    "n_bands": 3,
    "distances": [0.0, ...],                     // x-axis, length n_kpoints
    "frequencies_THz": [[...], ...],             // (n_kpoints, n_bands)
    "high_symmetry_points": {
      "distances": [0.0, 1.74, ...],             // tick positions
      "labels": ["Γ", "X", "W", "K", "Γ", "L", ...]
    },
    "max_frequency_THz": 7.51
  },
  "dos": {
    "frequencies_THz": [...],                    // DOS energy axis (THz)
    "dos": [...],                                // weights, states per THz
    "grid": [24, 24, 24],                        // q-mesh used
    "n_points": 200
  }
}
```

Everything needed to plot a dispersion *and* a DOS is in the result — no
post-processing of force constants required on the client side. The DOS uses a
`dos_grid`³ q-mesh (default 24; set `dos_grid=0` to skip it).

## Run it

```bash
# Fast EMT (instant, qualitative):
python examples/phonons/phonon_dispersion.py

# Publication-grade numbers on a UMA model:
FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p1 FAIRCHEM_MCP_EXAMPLE_TASK=omat \
  python examples/phonons/phonon_dispersion.py
```

The script polls the background job, prints the stability verdict and k-path,
and renders [`recordings/cu_phonon_dispersion.png`](recordings/).

## Calling the tool directly (agent / MCP)

```python
start_phonons(structure_id, calculator_id, supercell=[4, 4, 4], band_points=200)
# poll get_status for done/total, then:
get_results(job_id)   # -> gamma frequencies, stability, and the dispersion block
```
