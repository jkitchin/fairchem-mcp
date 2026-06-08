# Alloys & mechanical-properties examples

Two new property tools plus the design loop they enable. Each example comes in
**two forms**:

- a **runnable Python script** (`NN_*.py`) calling the same `fairchem_mcp` domain
  functions the MCP tools wrap;
- a **MCP tool-call walkthrough** (`NN_*.md`) showing the agent's exact sequence.

| # | What | Tool | Output | Script | Walkthrough |
|---|------|------|--------|--------|-------------|
| 1 | Elastic stiffness tensor | `start_elastic_scan` | C_ij, K/G/E, Poisson, Pugh, Born stability | [`01_elastic_tensor.py`](01_elastic_tensor.py) | [md](01_elastic_tensor.md) |
| 2 | Formation-energy convex hull | `start_convex_hull` | E_f, energy-above-hull, stable phases | [`02_convex_hull.py`](02_convex_hull.py) | [md](02_convex_hull.md) |
| 3 | Alloy design loop | both, in a loop | stiffest stable phase | [`03_alloy_design_loop.py`](03_alloy_design_loop.py) | [md](03_alloy_design_loop.md) |

## The idea

- **Elastic tensor** — the anisotropic analog of an equation of state (see
  [`../catalysis`](../catalysis) and `start_eos_scan`): strain the cell in each
  Voigt direction, read the stress, fit `C_ij`, average to engineering moduli, and
  check Born mechanical stability.
- **Convex hull** — *which compositions actually form?* Formation energies
  referenced to the pure elements, lower-hull-pruned into a set of stable phases.
- **Design loop** — gate candidates on stability (the hull), then rank the
  survivors by a mechanical property (here Young's modulus, Born-stable only). Swap
  the objective and you have a different design study.

## Running the scripts

```bash
pip install -e .                      # nothing extra needed (scipy is a core dep)
python examples/alloys/01_elastic_tensor.py
python examples/alloys/02_convex_hull.py
python examples/alloys/03_alloy_design_loop.py
```

All run on the fast **EMT** calculator by default (no GPU/model) and finish in
seconds. For real numbers, point them at a FAIRChem UMA model:

```bash
export FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p2
export FAIRCHEM_MCP_EXAMPLE_TASK=omat
python examples/alloys/03_alloy_design_loop.py
```

**EMT caveat (important).** ASE's built-in EMT nails simple-metal elastic
constants (its fcc Cu moduli land close to experiment) but is a single-element-fit
potential with a crude alloy *cross*-interaction. It happens to be **right for
Cu-Au** — the system Jacobsen's EMT parameters were validated on — which is why
these examples use it: Cu₃Au and CuAu correctly fall on the hull. For **most other
binaries it gets the sign wrong**: Ni-Al, for instance, comes out with positive
formation energies (verified at machine-zero stress and forces — it's the
potential, not the relaxation), so its real intermetallics wrongly look unstable.
For quantitative alloy energetics use `asap3`'s alloy-parameterized EMT, or — for
any system — point these scripts at a UMA model. The *workflow* is identical; only
the energies get trustworthy.
