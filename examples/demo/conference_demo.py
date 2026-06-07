"""Conference demo driver for fairchem-mcp (~5-7 min of terminal activity).

Runs the real engine — the same `fairchem_mcp.domain` / `introspect` calls the MCP
tools wrap — so every number on screen is real and reproducible. Three beats:

  1. Watch + steer   : a relaxation that grinds in the FIRE tail, switched to
                       LBFGS mid-flight (positions carry over -> it converges).
  2. Transition state: a dimer saddle search; watch the curvature flip negative.
  3. Escape hatch    : introspect / evaluate Python on the *same live atoms*.

Two ways to drive it:

    python examples/demo/conference_demo.py          # wait for ENTER between beats
                                                     # (use this when presenting live)
    DEMO_AUTO=1 python examples/demo/conference_demo.py   # auto-advance with pauses
                                                     # (use this when recording)

Calculator: EMT by default (instant, no GPU/model). For a real model:
    FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p2 FAIRCHEM_MCP_EXAMPLE_TASK=omat \\
        python examples/demo/conference_demo.py
"""

from __future__ import annotations

import os
import sys
import time

from fairchem_mcp import domain, introspect as introspect_mod
from fairchem_mcp.jobs import JobManager
from fairchem_mcp.session import Session

AUTO = bool(os.environ.get("DEMO_AUTO"))
SPEED = float(os.environ.get("DEMO_SPEED", "1.0"))  # >1 = slower, <1 = faster
_MODEL = os.environ.get("FAIRCHEM_MCP_EXAMPLE_MODEL")
_TASK = os.environ.get("FAIRCHEM_MCP_EXAMPLE_TASK", "omat")

# --- tiny terminal-theatre helpers -----------------------------------------
B, DIM, GRN, YEL, CYN, MAG, RST = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[36m", "\033[35m", "\033[0m"
)


def _nap(t: float) -> None:
    time.sleep(t * SPEED)


def banner(n: int, title: str) -> None:
    line = "─" * 64
    print(f"\n{CYN}{line}{RST}")
    print(f"{CYN}{B}  BEAT {n}  ·  {title}{RST}")
    print(f"{CYN}{line}{RST}\n")


def say(text: str) -> None:
    """Narration — the line you'd say out loud."""
    print(f"{MAG}{B}» {text}{RST}")
    _nap(0.8)


def call(text: str) -> None:
    """A tool call the agent issues."""
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


def calculator(session: Session) -> str:
    if _MODEL:
        cid = domain.load_model(session, model=_MODEL, task=_TASK)["calculator_id"]
        say(f"Model is already loaded and RESIDENT: UMA {_MODEL} (task={_TASK}).")
        return cid
    say("Calculator attached (EMT — instant; swap one call for a UMA model).")
    return domain.attach_emt(session)["calculator_id"]


# --- Beat 1: watch a live relaxation and steer it --------------------------
def beat_watch_and_steer(session: Session, cid: str) -> str:
    banner(1, "Watch a live relaxation — and steer it mid-flight")
    say("Batch tools run a script and wait. Here the job runs in the background "
        "and we watch it.")

    call('build_structure(surface Cu(111) 4x4x4, rattled 0.25 Å)')
    sid = domain.build_structure(
        session,
        {"kind": "surface", "name": "Cu", "size": [4, 4, 4],
         "vacuum": 10.0, "rattle": 0.25, "seed": 3},
    )["structure_id"]
    result(f"{len(session.get_structure(sid))} atoms, perturbed off their lattice sites")

    call('start_relaxation(optimizer="FIRE", fmax=0.01)  → returns immediately')
    job = session.get_job(
        domain.start_relaxation(
            session, sid, cid, optimizer="FIRE", fmax=0.01, steps=400, step_delay=0.08
        )["job_id"]
    )
    print()
    say("Polling the live status — energy, max force, and a TREND verdict:")

    # Watch FIRE crush the big forces, then crawl in the convergence tail.
    t0 = time.time()
    while job.is_active():
        d = job.status_dict()
        f = d.get("max_force")
        if f is not None:
            print(f"{DIM}    get_status → step {d['step']:>3}  "
                  f"max_force {f:6.3f}  trend={d['trend']['label']}{RST}")
        if f is not None and f < 0.20 and d["step"] >= 25:
            break
        if time.time() - t0 > 60:
            break
        _nap(0.7)

    at_switch = job.status_dict()
    print()
    say("Forces fell from ~3 to ~0.1, but FIRE has no curvature model — it's "
        "dawdling in the tail. Switch optimizer WITHOUT restarting:")
    call('steer("switch_optimizer", optimizer="LBFGS")')
    JobManager(session).steer(job, "switch_optimizer", optimizer="LBFGS")

    t0 = time.time()
    while job.is_active():
        if time.time() - t0 > 60:
            break
        _nap(0.2)
    d = job.status_dict()
    result(f"status={d['status']}  step={d['step']}  max_force={d['max_force']:.4f}")
    print()
    say(f"It picked up from the SAME positions (step {at_switch['step']} → "
        f"{d['step']}) — no restart — and converged. That's mid-flight steering.")
    pause()
    return sid


# --- Beat 2: a transition state, with a live curvature readout -------------
def beat_transition_state(session: Session, cid: str) -> None:
    banner(2, "Find a transition state — watch the curvature flip negative")
    from ase.build import add_adsorbate, fcc100
    from ase.constraints import FixAtoms

    say("An Al adatom hops between hollow sites on Al(100). The bridge site "
        "between them is the transition state.")

    slab = fcc100("Al", size=(2, 2, 3), vacuum=10.0)
    zmean = slab.positions[:, 2].mean()
    slab.set_constraint(FixAtoms(mask=[p[2] < zmean for p in slab.positions]))
    add_adsorbate(slab, "Al", 1.7, "hollow")
    sid = session.add_structure(slab)
    call("start_relaxation(...)  # relax the adatom into the hollow minimum")
    rjob = session.get_job(
        domain.start_relaxation(session, sid, cid, fmax=0.05, steps=200)["job_id"]
    )
    while rjob.is_active():
        _nap(0.05)
    e_min = rjob.status_dict()["energy"]
    result(f"hollow-site minimum: E = {e_min:.3f} eV")

    natoms = len(session.get_structure(sid))
    dvec = [0.0] * (3 * natoms)
    dvec[-3] = 0.3  # nudge the adatom toward the bridge in +x
    call('start_saddle_search(dimer, nudged toward the bridge)')
    sjob = session.get_job(
        domain.start_saddle_search(
            session, sid, cid, displacement_vector=dvec,
            fmax=0.05, steps=300, step_delay=0.12,
        )["job_id"]
    )
    print()
    say("Watching the dimer's curvature — it starts uphill, then goes negative "
        "as we climb onto the saddle:")
    t0 = time.time()
    last = -1
    while sjob.is_active():
        d = sjob.status_dict()
        c = d.get("curvature")
        if c is not None and d["step"] != last:
            last = d["step"]
            tag = f"{GRN}saddle-like{RST}" if c < 0 else f"{YEL}still uphill{RST}"
            print(f"{DIM}    get_status → step {d['step']:>3}  "
                  f"curvature {c:+6.3f}  {tag}")
        if time.time() - t0 > 60:
            break
        _nap(0.25)

    r = sjob.result
    print()
    result(f"is_index1_saddle={r['is_index1_saddle']}  "
           f"curvature={r['curvature']:+.3f}  barrier={r['energy'] - e_min:.3f} eV")
    say("One downhill direction = exactly an index-1 transition state. "
        "Three routes to it ship in the server: dimer, Sella, and POUNCE.")
    pause()


# --- Beat 3: the escape hatch ----------------------------------------------
def beat_escape_hatch(session: Session, sid: str, cid: str) -> None:
    banner(3, "Escape hatch — raw Python on the same live objects")
    say("When the tools run out, the agent drops to Python on the SAME live "
        "atoms — one shared namespace.")
    # Point the 'atoms' alias back at the relaxed slab from beat 1 (it still
    # carries its calculator), exactly as the most-recent-structure alias would.
    live = session.get_structure(sid)
    live.calc = session.get_calculator(cid)
    session.namespace["atoms"] = live

    call('inspect_expr("np.linalg.norm(atoms.get_forces(), axis=1).max()")')
    val = eval(  # noqa: S307 - mirrors the trusted escape hatch
        "np.linalg.norm(atoms.get_forces(), axis=1).max()", session.namespace
    )
    result(f"{float(val):.4f}  eV/Å  (max force on the live structure)")

    call('execute("print(len(atoms), \'atoms;\', set(atoms.get_chemical_symbols()))")')
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(  # noqa: S102 - mirrors the trusted escape hatch
            "print(len(atoms), 'atoms;', set(atoms.get_chemical_symbols()))",
            session.namespace,
        )
    result(buf.getvalue().strip())

    call('introspect("atoms", live=True)   # real installed API, not stale docs')
    info = introspect_mod.introspect("atoms", session.namespace, live=True)
    sig = info.get("signature") or info.get("type") or "ase.atoms.Atoms"
    doc = (info.get("docstring") or "").strip().splitlines()
    result(f"{sig}")
    if doc:
        result(doc[0][:70])
    pause()


def main() -> None:
    print(f"\n{B}fairchem-mcp{RST} — agent-steerable simulation, live demo")
    print(f"{DIM}(every number below is from the real engine; "
          f"{'AUTO' if AUTO else 'press ENTER between beats'}){RST}")
    session = Session()
    cid = calculator(session)
    slab_sid = beat_watch_and_steer(session, cid)
    beat_transition_state(session, cid)
    beat_escape_hatch(session, slab_sid, cid)
    print(f"\n{GRN}{B}  Resident model · watch & steer · Python escape hatch.{RST}")
    print(f"{GRN}  Batch scripting → collaborating on a simulation.{RST}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
