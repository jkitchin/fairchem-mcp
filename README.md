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
pip install -e .            # core: mcp + ase + jedi (works with EMT)
pip install -e ".[fairchem]"  # add FAIRChem (torch + models) for UMA/eSEN
```

FAIRChem is **optional**. Every tool except `load_model` works with a plain ASE
calculator (`attach_emt`), so you can develop and test without a GPU or model.

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
| `build_structure` / `load_structure` | Make/register an ASE structure |
| `start_relaxation` / `start_md` | Launch a background relaxation / MD (returns a `job_id`) |
| `start_neb` | Launch a steerable nudged-elastic-band (reaction barrier) |
| `start_phonons` | Launch a finite-displacement phonon calculation |
| `start_minima_search` | Find multiple distinct relaxed geometries via deflation/flooding |
| `get_status` / `get_trajectory` | Observe a running job |
| `get_results` | Final results: NEB barrier/energies, phonon frequencies & stability, distinct minima |
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
  `kernel="deflation"` adds inverse-distance poles (`eta`, `power`).
- New minima are deduplicated by energy (`energy_tol`) + RMSD (`rmsd_tol`) and
  each is registered as its own structure.

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

## Safety

`execute` / `inspect_expr` run arbitrary Python in-process. This is a **trusted
local developer tool** — do not expose it to untrusted input or over a network.

## Notes

- Only one job runs at a time (serializes model/GPU access).
- All optimizers/integrators use `logfile=None`; the stdio transport reserves
  stdout for the MCP protocol.
