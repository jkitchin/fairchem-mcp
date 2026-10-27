---
name: fairchem-mcp
description: Drives interactive FAIRChem/ASE atomistic simulations through the fairchem MCP server (tools such as load_model, build_structure, start_relaxation, get_status, steer). Use when the user wants to relax a structure, run MD, compute a reaction barrier (NEB, dimer, Sella), phonons, an equation of state, elastic constants, a convex hull, or find multiple minima with UMA/eSEN, EMT or LAMMPS, and wants to watch or steer the run as it goes.
license: MIT
compatibility: Requires the fairchem MCP server (install with `uvx knowhow add jkitchin/fairchem-mcp --mcp fairchem`). UMA models need the fairchem extra and Hugging Face access to facebook/UMA.
metadata:
  version: "0.1.0"
---

# Steering FAIRChem simulations through fairchem-mcp

The `fairchem` MCP server keeps a model loaded in memory and runs simulations as
background jobs. You start a job, poll it, and change it while it runs. Every
object is registered by id: structures are `struct_N`, calculators `calc_N`, jobs
`job_N`. Pass those ids between tools.

If the tools are missing, the server is not installed. Tell the user to run
`uvx knowhow add jkitchin/fairchem-mcp --mcp fairchem` and restart the agent.

## The basic loop

1. **Get a calculator.** `load_model(model="uma-s-1p1", task=...)` for UMA, where
   `task` is `omat` (inorganic bulk), `omol` (molecules), `oc20` (catalysis,
   adsorbates on surfaces), `odac` (MOFs) or `omc` (molecular crystals). For quick
   tests, or when FAIRChem is not installed, use `attach_emt()` (Cu, Ag, Au, Ni, Pd,
   Pt, Al, Pb only). `attach_lammps(pair_style, pair_coeff, atom_types)` gives a
   classical potential.
2. **Get a structure.** `build_structure(spec)` or `load_structure(path)` (any
   ASE-readable file).
3. **Start a job.** Any `start_*` tool returns a `job_id` immediately.
4. **Watch it.** `get_status(job_id)` gives step, energy, max force and a `trend`
   label. `get_trajectory(job_id, last_n)` gives recent history.
5. **Steer it** when the trend says so (see below).
6. **Read the answer.** `get_results(job_id)` once the status is `converged` or
   `finished`. `finished` on a relaxation means the step budget ran out before
   `fmax` was reached, so the geometry is not converged. Other end states are
   `aborted` and `failed`.

Only one job runs at a time. Start the next job after the previous one finishes or
is aborted.

## Structure specs

```json
{"kind": "bulk", "name": "Cu", "crystalstructure": "fcc", "a": 3.6, "repeat": [2,2,2]}
{"kind": "molecule", "name": "H2O", "vacuum": 6.0}
{"kind": "surface", "name": "Pt", "size": [2,2,4], "vacuum": 10.0}
{"kind": "surface", "symbol": "Pt", "indices": [1,1,0], "layers": 4, "vacuum": 10.0}
```

Add `"rattle": 0.1` to perturb positions and `"repeat": [n,m,k]` to make a
supercell. For anything more complex (adsorbates, constraints, alloys), build it
with `execute` (see the escape hatch) and register it there.

## Reading the trend and steering

`get_status` returns `trend.label`, one of `starting`, `decreasing`, `plateaued`,
`stuck` or `diverging`. Act on it with `steer(job_id, command, ...)`:

| Trend | What to do |
|---|---|
| `decreasing` | Leave it alone; poll again later. |
| `plateaued` | Usually fine near convergence. If far from `fmax`, switch optimizer. |
| `stuck` | `steer(job, "switch_optimizer", optimizer="LBFGS")` (or `FIRE`/`BFGS`). Positions carry over. |
| `diverging` | `steer(job, "pause")`, inspect the geometry, then `abort` or switch to `FIRE`. |

Other commands: `set_fmax` (with `value`) to tighten or loosen convergence,
`set_temperature` (with `value`, in K) for MD, `set_climb` to turn on the climbing
image once a NEB is nearly converged, and `pause` / `resume` / `abort`.

With fast calculators such as EMT, a job may finish before you can poll it. Pass
`step_delay` (seconds per step) to `start_*` when the user wants to watch or steer.

## Which job for which question

| Question | Tool | Notes |
|---|---|---|
| Relaxed geometry / energy | `start_relaxation` | `relax_cell=True` to relax the cell too. |
| Dynamics, temperature effects | `start_md` | `ensemble="NVT"` (Langevin) or `"NVE"`. |
| Barrier between two known minima | `start_neb` | Relax both endpoints first. Turn on climb near the end. |
| Transition state from one guess | `start_saddle_search` (dimer), `start_sella_search` | Sella needs the `saddles` extra. |
| Enumerate saddles | `start_pounce_saddles` | Needs the `saddles` extra. |
| Vibrations, dynamic stability | `start_phonons` | Results include Gamma frequencies, dispersion and DOS. |
| Lattice constant, bulk modulus | `start_eos_scan` | Returns V0, E0, bulk modulus in GPa. |
| Elastic tensor, Born stability | `start_elastic_scan` | Returns C_ij and VRH moduli. |
| Phase stability of compositions | `start_convex_hull` | Pass a list of structure ids. |
| Several distinct minima (sites, isomers) | `start_minima_search` | See below. |

For `start_minima_search`, use `kernel="flooding"` (default) or `"deflation"` for
fixed-frame problems such as an adsorbate on a frozen slab. Use
`kernel="basinhopping"` with `comparator="fingerprint"` for free clusters and
molecules, where rotations would otherwise be counted as new minima. Each minimum
found is registered as its own structure.

Call `introspect("fairchem_mcp.server.<tool>")` if you need a tool's full
parameter list.

## Escape hatch

`execute(code)` runs Python in the same namespace the tools use. Structures,
calculators and jobs are bound by id (`struct_1`, `calc_1`, `job_1`), and the most
recent ones are also bound as `atoms`, `calc` and `job`. `inspect_expr(expr)`
evaluates an expression and returns its repr, for example
`atoms.get_potential_energy()`. `introspect(target, live=True)` shows signatures
and members of live objects; a trailing `.` lists members (`"atoms."`).

Use these to build structures the spec cannot express, add constraints, or compute
derived quantities (adsorption energies, distances) from finished jobs. They run
arbitrary code locally, so only do what the user asked for.

## Pitfalls

- Use the UMA `task` that matches the system. A molecule with `omat` or a bulk
  oxide with `omol` gives poor energies. Compare energies only within one
  calculator and task.
- EMT is for testing workflows, not for reporting numbers to the user. Say so when
  you use it.
- Relax NEB endpoints, and the reference structures for adsorption or formation
  energies, with the same calculator and `fmax` as the main calculation.
- A converged relaxation can still be a saddle point. If it matters, check with
  `start_phonons` for imaginary modes.
- `load_model` fails without the `fairchem` extra or without Hugging Face access to
  `facebook/UMA`. Report the error to the user rather than silently falling back
  to EMT.
