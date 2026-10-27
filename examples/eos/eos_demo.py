"""Equation-of-state demo for fairchem-mcp: fcc Cu on EMT (~30 s of activity).

A one-variable lab. The cell volume is the single knob: we build a spread of
isotropically strained copies of fcc Cu, evaluate the energy at each, and fit a
Birch-Murnaghan equation of state for the equilibrium volume V0, the minimum
energy E0, and the bulk modulus B. This is the same `fairchem_mcp.domain` call
the MCP `start_eos_scan` tool wraps, so every number on screen is real.

Calculator is EMT by default (instant, no GPU/model) -- qualitative but the
workflow is identical on a real model. Point it at UMA for meaningful numbers:
    FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p2 FAIRCHEM_MCP_EXAMPLE_TASK=omat \\
        python examples/eos/eos_demo.py

Two ways to drive it:
    python examples/eos/eos_demo.py                 # press ENTER between beats
    DEMO_AUTO=1 python examples/eos/eos_demo.py      # auto-advance (for recording)
"""

from __future__ import annotations

import os
import time

from ase.build import bulk

from fairchem_mcp import domain
from fairchem_mcp.session import Session

AUTO = bool(os.environ.get("DEMO_AUTO"))
SPEED = float(os.environ.get("DEMO_SPEED", "1.0"))  # >1 = slower pauses

# --- tiny terminal-theatre helpers -----------------------------------------
B, DIM, GRN, YEL, CYN, MAG, RST = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[36m", "\033[35m", "\033[0m"
)


def _nap(t: float) -> None:
    time.sleep(t * SPEED)


def banner(title: str) -> None:
    line = "─" * 66
    print(f"\n{CYN}{line}{RST}")
    print(f"{CYN}{B}  {title}{RST}")
    print(f"{CYN}{line}{RST}\n")


def step(msg: str) -> None:
    print(f"{DIM}$ {msg}{RST}")
    _nap(0.6)


def gate() -> None:
    if AUTO:
        _nap(1.2)
    else:
        input(f"\n{DIM}  [ENTER]{RST}")


def calculator(session: Session) -> str:
    model = os.environ.get("FAIRCHEM_MCP_EXAMPLE_MODEL")
    if model:
        task = os.environ.get("FAIRCHEM_MCP_EXAMPLE_TASK", "omat")
        info = domain.load_model(session, model=model, task=task)
        print(f"  calculator: {GRN}UMA {model}{RST} (task={task})")
        return info["calculator_id"]
    print(f"  calculator: {GRN}EMT{RST}  (fast/qualitative; set "
          f"FAIRCHEM_MCP_EXAMPLE_MODEL for UMA)")
    return domain.attach_emt(session)["calculator_id"]


def ev_plot(volumes: list, energies: list, v0: float, e0: float) -> None:
    """A compact ASCII E-V curve so the parabola is visible in the recording."""
    width, height = 50, 12
    vmin, vmax = min(volumes), max(volumes)
    emin, emax = min(energies), max(energies)
    espan = (emax - emin) or 1.0
    grid = [[" "] * width for _ in range(height)]
    for v, e in zip(volumes, energies):
        col = int(round((v - vmin) / ((vmax - vmin) or 1.0) * (width - 1)))
        row = int(round((e - emin) / espan * (height - 1)))
        grid[height - 1 - row][col] = f"{CYN}●{RST}"
    # mark the fitted minimum
    col0 = int(round((v0 - vmin) / ((vmax - vmin) or 1.0) * (width - 1)))
    grid[height - 1][min(col0, width - 1)] = f"{YEL}{B}V{RST}"
    print(f"   {DIM}E (eV){RST}")
    for r, line in enumerate(grid):
        axis = "┤" if r else "┐"
        print(f"        {DIM}{axis}{RST} " + "".join(line))
    print(f"        {DIM}└{'─' * width}{RST}")
    print(f"         {DIM}{vmin:.1f} Å³{' ' * (width - 12)}volume{' ' * 4}"
          f"{vmax:.1f} Å³{RST}")


def main() -> None:
    session = Session()

    banner("fairchem-mcp · equation of state · fcc Cu")
    print("  Sweep the cell volume, fit an equation of state, read off the")
    print("  equilibrium volume, cohesive minimum, and bulk modulus.\n")
    calc = calculator(session)
    gate()

    step("build_structure(fcc Cu, a=3.6 Å, cubic)")
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    sid = session.add_structure(atoms)
    print(f"  {len(atoms)} atoms, V = {atoms.get_volume():.2f} Å³  "
          f"({atoms.get_volume() / len(atoms):.2f} Å³/atom)")
    _nap(0.6)

    step("start_eos_scan(n_points=11, strain_range=0.05)")
    print(f"  {DIM}11 volumes spanning (1 ± 0.05)·V0; energy at each{RST}\n")
    res = domain.start_eos_scan(
        session, sid, calc, n_points=11, strain_range=0.05, step_delay=0.25)
    job = session.get_job(res["job_id"])

    # Live progress: print each volume point as the background job finishes it.
    seen = 0
    while job.is_active():
        s = job.status_dict()
        done = s.get("done", 0)
        if done > seen and s.get("volume") is not None:
            bar = "█" * done + "·" * (s.get("total", 11) - done)
            print(f"  {GRN}{bar}{RST}  point {done:>2}/{s.get('total', 11)}   "
                  f"V = {s['volume']:7.2f} Å³   E = {s['energy']:9.4f} eV")
            seen = done
        time.sleep(0.05)
    # catch the final point
    s = job.status_dict()
    if s.get("done", 0) > seen and s.get("volume") is not None:
        done = s["done"]
        bar = "█" * done + "·" * (s.get("total", 11) - done)
        print(f"  {GRN}{bar}{RST}  point {done:>2}/{s.get('total', 11)}   "
              f"V = {s['volume']:7.2f} Å³   E = {s['energy']:9.4f} eV")

    r = job.result or {}
    gate()

    banner("fit: Birch-Murnaghan equation of state")
    ev_plot(r["volumes"], r["energies"], r["V0"], r["E0"])
    print()
    natoms = len(atoms)
    print(f"  equilibrium volume  V0 : {YEL}{B}{r['V0']:.3f} Å³{RST}"
          f"   ({r['V0'] / natoms:.3f} Å³/atom)")
    print(f"  minimum energy      E0 : {YEL}{B}{r['E0']:.4f} eV{RST}"
          f"   ({r['E0'] / natoms:.4f} eV/atom)")
    print(f"  bulk modulus        B  : {YEL}{B}{r['bulk_modulus_GPa']:.1f} GPa{RST}"
          f"   {DIM}(expt. fcc Cu ≈ 140 GPa){RST}")
    print(f"\n  {DIM}EMT is qualitative; set FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p2 "
          f"for real numbers.{RST}\n")


if __name__ == "__main__":
    main()
