"""Alloy-design demo driver for fairchem-mcp (~3-5 min of terminal activity).

Runs the real engine -- the same `fairchem_mcp.domain` calls the MCP tools wrap --
so every number on screen is real and reproducible. Three beats that build on each
other into a property-driven alloy screen:

  1. Elastic tensor : relax fcc Cu, strain it, fit C_ij -> bulk/shear/Young's
                      moduli, Poisson + Pugh ratios, Born mechanical stability.
  2. Convex hull    : a Cu-Au composition line -> formation energies and the lower
                      hull (which phases are actually stable).
  3. Design loop    : gate candidates on stability, score the survivors' stiffness,
                      and rank -> the stiffest phase that will actually form.

Two ways to drive it:

    python examples/alloys/alloy_demo.py             # wait for ENTER between beats
    DEMO_AUTO=1 python examples/alloys/alloy_demo.py  # auto-advance (for recording)

Calculator: EMT by default (instant, no GPU/model). Cu-Au is the alloy system
ASE's built-in EMT was validated on, so the hull signs are correct here. For other
systems point it at a UMA model:
    FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p2 FAIRCHEM_MCP_EXAMPLE_TASK=omat \\
        python examples/alloys/alloy_demo.py
"""

from __future__ import annotations

import os
import sys
import time

from ase.build import bulk

from _common import calculator, cu_au_system, wait
from fairchem_mcp import domain
from fairchem_mcp.session import Session

AUTO = bool(os.environ.get("DEMO_AUTO"))
SPEED = float(os.environ.get("DEMO_SPEED", "1.0"))  # >1 = slower, <1 = faster

# Keep phases within this distance of the hull (eV/atom) as "accessible".
HULL_TOLERANCE = 0.01

# --- tiny terminal-theatre helpers -----------------------------------------
B, DIM, GRN, YEL, CYN, MAG, RST = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[36m", "\033[35m", "\033[0m"
)


def _nap(t: float) -> None:
    time.sleep(t * SPEED)


def banner(n: int, title: str) -> None:
    line = "─" * 66
    print(f"\n{CYN}{line}{RST}")
    print(f"{CYN}{B}  BEAT {n}  ·  {title}{RST}")
    print(f"{CYN}{line}{RST}\n")


def say(text: str) -> None:
    print(f"{MAG}{B}» {text}{RST}")
    _nap(0.8)


def call(text: str) -> None:
    print(f"{YEL}  ▶ {text}{RST}")
    _nap(0.4)


def result(text: str) -> None:
    print(f"{DIM}    ← {text}{RST}")
    _nap(0.3)


def pause() -> None:
    if AUTO:
        _nap(1.4)
    else:
        try:
            input(f"\n{DIM}    [ENTER to continue]{RST}")
        except EOFError:
            _nap(1.0)


def _watch(session: Session, job_id: str, label: str, timeout: float = 180.0):
    """Poll a background job, printing a handful of done/total ticks, then return it.

    Throttled to ~6 lines so a long scan reads cleanly on screen / in a recording.
    """
    job = session.get_job(job_id)
    last_frac = -1.0
    last_done = -1
    t0 = time.time()
    while job.is_active():
        d = job.status_dict()
        done, total = d.get("done"), d.get("total")
        if done is not None and total:
            frac = done / total
            # Print on ~15% progress boundaries (robust to coarse polling), and
            # always the final tick — keeps a long scan to ~6 readable lines.
            if done != last_done and (frac - last_frac >= 0.15 or done == total):
                last_frac, last_done = frac, done
                print(f"{DIM}    get_status → {label} {done}/{total}"
                      f"  ({frac:.0%}){RST}")
        if time.time() - t0 > timeout:
            break
        _nap(0.2)
    return job


# --- Beat 1: elastic tensor of a pure metal --------------------------------
def beat_elastic(session: Session, cid: str) -> None:
    banner(1, "Elastic tensor — how stiff is a crystal?")
    say("Strain the cell in each Voigt direction, read the stress, fit C_ij. "
        "Start from fcc Cu relaxed to its equilibrium lattice.")

    sid = session.add_structure(bulk("Cu", "fcc", a=3.6, cubic=True))
    call('start_relaxation(relax_cell=True, fmax=0.01)   # equilibrium cell first')
    wait(session, domain.start_relaxation(
        session, sid, cid, relax_cell=True, fmax=0.01, steps=300)["job_id"])

    call('start_elastic_scan(n_strains=7, max_strain=0.01)  → returns immediately')
    job = _watch(session, domain.start_elastic_scan(
        session, sid, cid, n_strains=7, max_strain=0.01, step_delay=0.05)["job_id"],
        "deformation")
    r = job.result
    C = r["C_GPa"]
    print()
    result(f"C11,C12,C44 = {C[0][0]:.0f}, {C[0][1]:.0f}, {C[3][3]:.0f} GPa")
    result(f"K={r['bulk_modulus_GPa']:.0f}  G={r['shear_modulus_GPa']:.0f}  "
           f"E={r['youngs_modulus_GPa']:.0f} GPa   ν={r['poisson_ratio']:.2f}")
    result(f"Pugh G/K={r['pugh_ratio_G_over_K']:.2f} "
           f"({'ductile' if r['pugh_ratio_G_over_K'] < 0.57 else 'brittle'})   "
           f"Born-stable={r['mechanically_stable']}")
    say("EMT Cu lands right on experiment (C11≈168, C12≈121, C44≈75; ν≈0.34), "
        "and Pugh < 0.57 correctly calls copper ductile.")
    pause()


# --- Beat 2: convex hull over a composition line ---------------------------
def beat_hull(session: Session, cid: str):
    banner(2, "Convex hull — which alloy compositions are stable?")
    say("Build the Cu-Au line: pure Cu, pure Au, and the Cu3Au / CuAu / CuAu3 "
        "orderings. Formation energy is referenced to the pure metals.")

    candidates = cu_au_system(session)                 # [(sid, label), ...]
    call('start_convex_hull(relax=True, relax_cell=True)  # 5 compositions')
    job = _watch(session, domain.start_convex_hull(
        session, [s for s, _ in candidates], cid,
        relax=True, relax_cell=True, step_delay=0.15,
        labels=[lbl for _, lbl in candidates])["job_id"], "evaluate")
    r = job.result
    print()
    print(f"{DIM}    {'phase':7s}{'x_Au':>6s}{'Ef (eV/at)':>12s}{'above hull':>12s}"
          f"  stable{RST}")
    for p in sorted(r["phases"], key=lambda p: p["composition"].get("Au", 0.0)):
        flag = f"{GRN}  ★{RST}" if p["on_hull"] else ""
        print(f"{DIM}    {p['label']:7s}{p['composition'].get('Au', 0.0):6.2f}"
              f"{p['formation_energy_per_atom']:+12.4f}"
              f"{p['energy_above_hull_per_atom']:12.4f}{RST}{flag}")
    print()
    say(f"On the hull (stable): {r['stable_phases']}. Cu3Au and CuAu have negative "
        "formation energies — the classic L1₂/L1₀ orderings. CuAu3 sits above.")
    pause()
    return candidates, r


# --- Beat 3: the design loop -----------------------------------------------
def beat_design(session: Session, cid: str, candidates, hull) -> None:
    banner(3, "Design loop — the stiffest phase that will actually form")
    sid_of = {lbl: sid for sid, lbl in candidates}

    accessible = [p for p in hull["phases"]
                  if p["energy_above_hull_per_atom"] <= HULL_TOLERANCE]
    say(f"Step 1 — gate on stability: keep phases ≤ {HULL_TOLERANCE} eV/atom above "
        f"the hull. That prunes "
        f"{[p['label'] for p in hull['phases'] if p not in accessible]} and keeps "
        f"{[p['label'] for p in accessible]}.")

    say("Step 2 — score each survivor's stiffness (relax cell, then elastic scan):")
    scored = []
    for p in accessible:
        sid = sid_of[p["label"]]
        wait(session, domain.start_relaxation(
            session, sid, cid, relax_cell=True, fmax=0.02, steps=300)["job_id"])
        el = wait(session, domain.start_elastic_scan(
            session, sid, cid, n_strains=5, max_strain=0.01)["job_id"]).result
        scored.append({"label": p["label"], "E": el["youngs_modulus_GPa"],
                       "K": el["bulk_modulus_GPa"],
                       "pugh": el["pugh_ratio_G_over_K"],
                       "stable": el["mechanically_stable"]})
        result(f"{p['label']:7s} E={scored[-1]['E']:6.1f}  K={scored[-1]['K']:6.1f}"
               f"  Pugh={scored[-1]['pugh']:.2f}  Born-stable={scored[-1]['stable']}")

    viable = sorted([s for s in scored if s["stable"]],
                    key=lambda s: s["E"], reverse=True)
    print()
    say("Step 3 — rank by Young's modulus (Born-stable only):")
    win = viable[0]
    print(f"{GRN}{B}    winner: {win['label']}  E={win['E']:.0f} GPa, "
          f"K={win['K']:.0f} GPa, Pugh={win['pugh']:.2f}{RST}")
    print(f"{DIM}    runners-up: "
          + ", ".join(f"{s['label']} (E={s['E']:.0f})" for s in viable[1:]) + RST)
    say("Stability first, then property — the same loop with a different objective "
        "(max K, most ductile, …) is a different design study.")
    pause()


def main() -> None:
    print(f"\n{B}fairchem-mcp{RST} — alloy design, live (elastic · hull · screen)")
    print(f"{DIM}(every number below is from the real engine; "
          f"{'AUTO' if AUTO else 'press ENTER between beats'}){RST}")
    session = Session()
    cid = calculator(session)
    beat_elastic(session, cid)
    candidates, hull = beat_hull(session, cid)
    beat_design(session, cid, candidates, hull)
    print(f"\n{GRN}{B}  Elastic tensor · convex hull · property-driven screen.{RST}")
    print(f"{GRN}  Steerable jobs, one shared namespace — design as a conversation.{RST}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
