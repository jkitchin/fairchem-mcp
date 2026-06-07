# Saddle-point / transition-state examples

Three routes to a **transition state** in `fairchem-mcp`, all demonstrated on the
same physical system — an Al adatom hopping from a hollow site to the bridge site
on Al(100), the index-1 saddle of surface diffusion — so you can compare them
directly. Each comes in **two forms**:

- a **runnable Python script** (`NN_*.py`) calling the same `fairchem_mcp` domain
  functions the MCP tools wrap;
- a **MCP tool-call walkthrough** (`NN_*.md`) showing the agent's exact sequence.

| # | Method | Tool | Hessian | Output | Script | Walkthrough |
|---|--------|------|---------|--------|--------|-------------|
| 1 | Dimer (min-mode following) | `start_saddle_search` | none (forces only) | one saddle near a seed | [`01_dimer_saddle.py`](01_dimer_saddle.py) | [md](01_dimer_saddle.md) |
| 2 | Eigenvector following (multistart) | `start_pounce_saddles` | full, finite-difference | several, by Morse index | [`02_pounce_saddles.py`](02_pounce_saddles.py) | [md](02_pounce_saddles.md) |
| 3 | Sella (rational-function opt.) | `start_sella_search` | approximate, refined online | one order-`k` saddle | [`03_sella_saddle.py`](03_sella_saddle.py) | [md](03_sella_saddle.md) |

For a *known reaction with both endpoints*, the climbing-image NEB in
[`../catalysis/03_diffusion_barrier`](../catalysis/03_diffusion_barrier.md) is
often the most robust choice. These three are **single-ended** — they need only a
starting structure, not the product — which is what you want when you don't know
the product geometry, or want to *enumerate* saddles.

## Which one?

- **Dimer** — most robust on noisy ML potentials (no Hessian at all). Use when you
  have a guess for the reaction direction (the `displacement_vector` seed) and
  want a single transition state.
- **POUNCE `find_saddles`** — when you want to *enumerate* several distinct saddles
  and have each labeled by Morse index (1 = TS, 2 = second-order, …). Builds a
  full finite-difference Hessian, so restrict `active_indices` to the reacting
  atoms and loosen `grad_tol` for float32 ML potentials.
- **Sella** — efficient (few force calls), steerable, and supports internal
  coordinates (`internal=True`, good for molecules). A strong general default for
  a single order-`k` saddle.

## Running the scripts

```bash
pip install -e ".[saddles]"            # adds sella + pounce-solver
python examples/saddles/01_dimer_saddle.py
```

The dimer needs no extra dependency (it's pure ASE). Sella needs `sella`; POUNCE
needs `pounce-solver` — both are in the `[saddles]` extra. All run with the fast
**EMT** calculator by default (no GPU/model) and finish in seconds; EMT energies
are only *qualitative*. For real numbers, point them at a FAIRChem UMA model:

```bash
export FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p1   # add: FAIRCHEM_MCP_EXAMPLE_TASK=oc20
python examples/saddles/03_sella_saddle.py
```

**ML-potential note.** Saddle searches are curvature-sensitive, and UMA's float32
forces carry ~1e-3 eV/Å noise. The dimer and Sella tolerate this well; POUNCE's
full-Hessian index counting is the most sensitive — loosen `grad_tol` and prefer
a float64 model. All three land on the same bridge saddle here (barrier ≈ 0.23 eV
on EMT), a useful cross-check.
