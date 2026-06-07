# Catalysis examples

Four end-to-end catalysis workflows for `fairchem-mcp`. Each comes in **two
forms**:

- a **runnable Python script** (`NN_*.py`) that calls the same `fairchem_mcp`
  domain functions the MCP tools wrap — run it directly to see the flow;
- a **MCP tool-call walkthrough** (`NN_*.md`) showing the exact sequence an agent
  issues through the server in Claude Code.

| # | Scenario | Key tool | Script | Walkthrough |
|---|----------|----------|--------|-------------|
| 1 | Adsorption energy (CO/Pt(111)) | `start_relaxation` | [`01_adsorption_energy.py`](01_adsorption_energy.py) | [md](01_adsorption_energy.md) |
| 2 | Adsorption-site search (H/Pt(111)) | `start_minima_search` | [`02_adsorption_sites.py`](02_adsorption_sites.py) | [md](02_adsorption_sites.md) |
| 3 | Diffusion barrier (NEB) | `start_neb` + `steer set_climb` | [`03_diffusion_barrier.py`](03_diffusion_barrier.py) | [md](03_diffusion_barrier.md) |
| 4 | Surface stability (phonons) | `start_phonons` | [`04_surface_stability.py`](04_surface_stability.py) | [md](04_surface_stability.md) |

## Running the scripts

```bash
pip install -e .                       # from the repo root
python examples/catalysis/01_adsorption_energy.py
```

They run with the fast **EMT** calculator by default (no GPU/model) so they
finish in seconds — EMT energies are only *qualitative*. For real catalysis
numbers, point them at a FAIRChem UMA model with the catalysis (`oc20`) task:

```bash
export FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p1   # add: FAIRCHEM_MCP_EXAMPLE_TASK=oc20
python examples/catalysis/01_adsorption_energy.py
```

## Task domains

For catalysis with UMA, use **`task="oc20"`** (catalysis / adsorption on
inorganic surfaces). Other domains: `omat` (bulk inorganic), `omol` (molecules),
`odac` (MOFs), `omc` (molecular crystals).
