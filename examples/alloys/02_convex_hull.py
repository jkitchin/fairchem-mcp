"""Formation-energy convex hull for the Cu-Au system (phase stability).

Evaluate several Cu-Au compositions (pure Cu, pure Au, and the ordered
intermetallics Cu3Au, CuAu, CuAu3), reference each to the pure elements, and
build the lower convex hull. Phases on the hull are thermodynamically stable; a
phase's distance above the hull (``energy_above_hull_per_atom``) is how
metastable it is. This is the classic alloy phase-stability picture.

Run:  python examples/alloys/02_convex_hull.py
"""

from __future__ import annotations

from _common import calculator, cu_au_system, wait
from fairchem_mcp import domain
from fairchem_mcp.session import Session


def main() -> None:
    session = Session()
    cid = calculator(session)

    entries = cu_au_system(session)
    sids = [sid for sid, _ in entries]
    labels = [label for _, label in entries]

    # relax=True relaxes the ions at each fixed cell; relax_cell also relaxes the
    # lattice (closer to each phase's true energy, a bit slower).
    res = domain.start_convex_hull(
        session, sids, cid, relax=True, relax_cell=True, labels=labels)
    job = wait(session, res["job_id"], timeout=300)
    r = job.result

    print("\n=== Cu-Au convex hull (EMT, qualitative) ===")
    print(f"  elements   : {r['elements']}")
    print(f"  references : "
          + ", ".join(f"{k}={v:.3f} eV/atom"
                      for k, v in r["references_eV_per_atom"].items()))
    print(f"  {'phase':8s} {'x_Au':>5s} {'Ef (eV/at)':>11s} "
          f"{'above hull':>11s}  stable")
    for p in sorted(r["phases"], key=lambda p: p["composition"].get("Au", 0.0)):
        x_au = p["composition"].get("Au", 0.0)
        flag = "  *" if p["on_hull"] else ""
        print(f"  {p['label']:8s} {x_au:5.2f} "
              f"{p['formation_energy_per_atom']:+11.4f} "
              f"{p['energy_above_hull_per_atom']:11.4f}{flag}")
    print(f"\n  stable phases (on hull): {r['stable_phases']}")
    print("\n(EMT is qualitative; set FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p2 "
          "FAIRCHEM_MCP_EXAMPLE_TASK=omat for real numbers.)")


if __name__ == "__main__":
    main()
