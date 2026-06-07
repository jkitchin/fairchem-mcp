# fairchem-mcp

An **agent-steerable MCP server** for [FAIRChem](https://github.com/facebookresearch/fairchem)
and [ASE](https://wiki.fysik.dtu.dk/ase/) simulations.

Most LLM-driven simulation today is *batch*: the agent writes a script, runs it,
waits, and reads the output. `fairchem-mcp` makes it *interactive*. The model is
loaded once and kept resident; relaxations and MD run in the background; and the
agent can **watch a simulation as it runs and steer it mid-flight** — switch the
optimizer when it stalls, tighten `fmax`, change temperature, pause, or abort.

## Why

- **Resident model.** Load a UMA/eSEN model once; reuse it across every call. No
  reloading the model (seconds–minutes) on each run.
- **Live monitoring.** `get_status` returns step, energy, max force, and a
  `trend` verdict (`decreasing` / `plateaued` / `stuck` / `diverging`) so the
  agent can decide whether to intervene.
- **Mid-flight steering.** `steer` can `pause` / `resume` / `abort`, `set_fmax`,
  `switch_optimizer`, or `set_temperature` on a running job. Switching optimizer
  carries the atomic positions over — it just rebuilds the driver.
- **Hybrid namespace.** High-level tools (`start_relaxation`, …) and the
  `execute` / `inspect_expr` escape hatch share one Python namespace, so the
  agent can drop to raw Python on the very same live `Atoms`.
- **Code awareness.** `introspect` reads the *installed* API (real signatures and
  docstrings) and live objects via `jedi` — not possibly-stale docs.

## Install

```bash
pip install -e .              # core: mcp + ase + jedi + numpy (works with EMT)
pip install -e ".[fairchem]"  # add FAIRChem (torch + models) for UMA/eSEN
pip install -e ".[lammps]"    # add LAMMPS as a classical force engine
pip install -e ".[saddles]"   # add Sella + POUNCE for transition-state searches
```

ASE, numpy, jedi and mcp are **core** dependencies (installed automatically) — the
server is fully usable out of the box with the built-in EMT calculator.

### FAIRChem (optional)

`pip install -e ".[fairchem]"` pulls in `fairchem-core` (PyTorch + models). To
actually load a UMA model you also need:

- a **Hugging Face account** with approved access to the `facebook/UMA` model
  repository, and `huggingface-cli login` (or `HF_TOKEN`) set;
- PyTorch for your platform (CUDA build for GPU; CPU works but is slower).

Everything except `load_model` works with a plain ASE calculator (`attach_emt`),
so you can develop and test without a GPU or model.

### LAMMPS (optional)

`pip install -e ".[lammps]"` installs the `lammps` Python package. LAMMPS is used
as a **force engine** (classical potentials) while ASE drives the dynamics, so
`attach_lammps` works with every steerable job (MD, relaxation, NEB, phonons,
EOS, minima search).

**macOS note (Homebrew MPICH):** the PyPI `lammps` wheel links
`@rpath/libmpi.12.dylib` and `libpmpi.12.dylib`, which the dynamic loader can't
find by default. If you have Homebrew's `mpich` (`brew install mpich`),
`attach_lammps` **auto-symlinks** those libs into the `lammps` package directory
on first use. If that can't be applied (read-only install, non-Homebrew MPI),
either symlink them yourself or export before launching the server:

```bash
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_FALLBACK_LIBRARY_PATH
```

Verify with: `python -c "from lammps import lammps; lammps(cmdargs=['-log','none','-screen','none']).close(); print('ok')"`.

### Transition-state searches (optional)

`pip install -e ".[saddles]"` adds two extra saddle-point backends:
[`sella`](https://github.com/zadorlab/sella) (a rational-function ASE optimizer)
and [`pounce-solver`](https://pypi.org/project/pounce-solver/) (multistart
eigenvector following). The **dimer** method (`start_saddle_search`) is pure ASE
and needs neither. See [`examples/saddles/`](examples/saddles/) for all three.

## Register with Claude Code

Add to your MCP config (see `examples/claude_mcp_config.json`):

```json
{
  "mcpServers": {
    "fairchem": { "command": "fairchem-mcp" }
  }
}
```

## Tools

| Tool | Purpose |
|---|---|
| `list_models` | List FAIRChem pretrained models |
| `load_model` | Load a UMA/eSEN model as a resident calculator |
| `attach_emt` | Attach a fast EMT calculator (no GPU/model) |
| `attach_lammps` | Attach LAMMPS (classical potentials) as the force engine |
| `build_structure` / `load_structure` | Make/register an ASE structure |
| `start_relaxation` / `start_md` | Launch a background relaxation / MD (returns a `job_id`) |
| `start_neb` | Launch a steerable nudged-elastic-band (reaction barrier) |
| `start_saddle_search` | Transition state via the dimer method (Hessian-free) |
| `start_sella_search` | Transition state via Sella (RFO + approximate Hessian) |
| `start_pounce_saddles` | Enumerate saddles by Morse index (POUNCE eigenvector following) |
| `start_phonons` | Launch a finite-displacement phonon calculation |
| `start_eos_scan` | Scan cell strain → fit equation of state (V0, E0, bulk modulus) |
| `start_minima_search` | Find multiple distinct relaxed geometries via deflation/flooding |
| `get_status` / `get_trajectory` | Observe a running job |
| `get_results` | Final results: NEB barrier/energies, phonon frequencies & stability, distinct minima, EOS fit |
| `steer` | `pause`/`resume`/`abort`/`set_fmax`/`switch_optimizer`/`set_temperature`/`set_climb` |
| `introspect` | Signatures/docstrings/members of installed code or live objects |
| `execute` / `inspect_expr` | Run/eval Python in the shared session namespace |

Resources: `sim://models`, `sim://job/{id}/status`, `sim://job/{id}/trajectory`.

## Example flow

```
attach_emt()                      -> calc_1
build_structure({"kind":"bulk","name":"Cu","crystalstructure":"fcc",
                 "a":3.6,"repeat":[2,2,2],"rattle":0.2})  -> struct_1
start_relaxation("struct_1","calc_1",optimizer="FIRE",fmax=0.01)  -> job_1
get_status("job_1")               -> {status:"running", trend:{label:"stuck", ...}}
steer("job_1","switch_optimizer",optimizer="LBFGS")
get_status("job_1")               -> {status:"converged", ...}
introspect("atoms", live=True)    -> live object signature/docstring
```

## Finding multiple relaxed geometries

`start_minima_search` finds several *distinct* local minima of the PES — useful
for surface adsorption sites, cluster isomers, or conformers. It relaxes
repeatedly from the starting structure on a PES **biased to repel the minima
already found**, then polishes each escape on the true PES:

- `kernel="flooding"` (default) adds Gaussian bumps (`sigma` Å, `amplitude` eV);
  `kernel="deflation"` adds inverse-distance poles (`eta`, `power`). Both are best
  for **fixed-frame** problems (an adsorbate on a frozen slab, an anchored
  conformer).
- `kernel="basinhopping"` (random kick + relax + Metropolis accept) is the right
  tool for **free clusters / nanoparticles**, whose rigid-body rotation defeats a
  spatial bias. Pair it with `comparator="fingerprint"` (see below).
- New minima are deduplicated by energy (`energy_tol`) plus a structure
  `comparator`: `"rmsd"` (raw coords, frame-dependent — fine for a fixed frame) or
  `"fingerprint"` (sorted pairwise distances; rotation/translation/permutation
  invariant — use for free clusters and molecules, or rotated copies get
  miscounted as distinct). Each accepted minimum is registered as its own
  structure.

This reuses the *escape mechanism* from POUNCE's `find_minima` (deflation /
flooding) but drives it with ASE's gradient optimizers — the right inner solver
for a PES — so each escape relaxation is a normal steerable job (watch the trend,
`switch_optimizer`, `set_fmax`, pause/abort). POUNCE's interior-point solver is
deliberately *not* used as the inner relaxer.

## Examples

[`examples/catalysis/`](examples/catalysis/) has four end-to-end catalysis
workflows — adsorption energy, adsorption-site search, diffusion-barrier NEB, and
surface-stability phonons — each as a runnable script **and** an MCP tool-call
walkthrough. They run on EMT out of the box; set `FAIRCHEM_MCP_EXAMPLE_MODEL` for
UMA.

[`examples/saddles/`](examples/saddles/) covers the three single-ended
transition-state routes — dimer, Sella, and POUNCE eigenvector following — on one
shared system so you can compare them.

## Safety

`execute` / `inspect_expr` run arbitrary Python in-process. This is a **trusted
local developer tool** — do not expose it to untrusted input or over a network.

## Notes

- Only one job runs at a time (serializes model/GPU access).
- All optimizers/integrators use `logfile=None`; the stdio transport reserves
  stdout for the MCP protocol.
