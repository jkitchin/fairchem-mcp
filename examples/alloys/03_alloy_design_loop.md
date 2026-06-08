# Alloy design loop — stability gate + property screen

**Goal:** put the two property tools together into the closed loop an agent
actually drives: *screen which phases can form, then optimize a mechanical
property over the survivors.* This is "design an alloy for a property" in
miniature.

```text
  candidates ──▶ [1] convex hull ──▶ keep phases on/near the hull
                                          │
                                          ▼
                                 [2] elastic tensor each
                                          │
                                          ▼
                       [3] rank by Young's modulus (Born-stable only)
```

The order matters: **stability first.** There's no point computing the stiffness
of a phase that won't form, so the hull screen prunes the candidate set before the
(more expensive) elastic scans run.

```text
attach_emt()  -> calc_1     # candidates: Cu, Au, Cu3Au, CuAu, CuAu3 (see _common.py)

# 1. Stability screen.
start_convex_hull([...5 sids...], "calc_1", relax=True, relax_cell=True,
                  labels=["Cu","Au","Cu3Au","CuAu","CuAu3"])  -> job_1
get_results("job_1")        # keep phases with energy_above_hull <= 0.01 eV/atom

# 2. For each surviving phase: relax the cell, then elastic scan.
start_relaxation(sid, "calc_1", relax_cell=True, fmax=0.02)   -> job_k   # wait
start_elastic_scan(sid, "calc_1", n_strains=5, max_strain=0.01) -> job_k+1
get_results(...)            # youngs_modulus_GPa, bulk_modulus_GPa, pugh_ratio, stable

# 3. Rank the Born-stable survivors by Young's modulus.
```

**Runnable version:** [`03_alloy_design_loop.py`](03_alloy_design_loop.py)

```
[1/3] screening phase stability (convex hull) ...
      4/5 phases within 0.01 eV/atom of the hull: ['Cu', 'Au', 'Cu3Au', 'CuAu']
[2/3] computing elastic moduli of accessible phases ...
      Cu       E= 149.0 GPa  K= 134.1 GPa  Pugh=   0.4  Born-stable=True
      Au       E=  89.2 GPa  K= 174.9 GPa  Pugh=   0.2  Born-stable=True
      Cu3Au    E= 138.4 GPa  K= 145.2 GPa  Pugh=   0.4  Born-stable=True
      CuAu     E= 112.5 GPa  K= 152.9 GPa  Pugh=   0.3  Born-stable=True
[3/3] ranking by Young's modulus (Born-stable only) ...

=== Cu-Au design result (EMT, qualitative) ===
  winner: Cu  E=149.0 GPa, K=134.1 GPa, Pugh=0.42 (ductile), 0.000 eV/atom above hull
  runners-up: Cu3Au (E=138), CuAu (E=112), Au (E=89)
```

**Reading it.** The hull gate prunes **CuAu₃** (it sits above the hull) and carries
the four stable phases into the elastic step. Ranking by stiffness, pure Cu wins,
with the ordered **Cu₃Au** a close second — a genuine intermetallic-vs-pure-metal
trade-off (and if your objective were *bulk* modulus instead, Au and CuAu would
jump ahead). Numbers are EMT-qualitative; the picture is the point.

**Make it your own — change one line:**
- **Objective:** rank by `bulk_modulus_GPa` (incompressible), or by *lowest*
  `pugh_ratio_G_over_K` (most ductile), or a weighted score.
- **Constraints:** tighten/loosen `HULL_TOLERANCE`, or require `pugh < 0.57`.
- **Candidates:** swap in a different system or a denser composition grid in
  `_common.cu_au_system`. (For systems EMT can't do — most binaries other than
  Cu-Au — point it at a UMA model; see the note in
  [`02_convex_hull`](02_convex_hull.md).)

**Why this shape suits the server.** Each step is a steerable background job with
live status, sharing one namespace, so an agent can watch the hull screen, decide
how many phases to carry forward, fan out the elastic scans, and report the winner
— adjusting the objective mid-loop if you ask it to. That's the batch-script
workflow turned into a conversation.
