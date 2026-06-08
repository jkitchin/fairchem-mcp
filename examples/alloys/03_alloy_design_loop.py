"""Design loop: screen Cu-Au alloys for a stiff, stable phase.

This ties the two property tools together into the kind of closed loop an agent
drives through the server:

  1. SCREEN STABILITY  - build the formation-energy convex hull over candidate
     compositions and keep only the phases on (or near) the hull. No point
     optimizing a property for a phase that won't form.
  2. SCORE PROPERTY    - for each surviving phase, compute the elastic tensor and
     read off the design target (here: Young's modulus, with a Born mechanical-
     stability gate and the Pugh ductility ratio reported alongside).
  3. RANK              - pick the stiffest mechanically-stable phase.

Swap the objective (max bulk modulus, most ductile via Pugh, etc.) or the
candidate set and the same loop becomes a different design study. On EMT the
numbers are qualitative; point it at a UMA model for real ones.

Run:  python examples/alloys/03_alloy_design_loop.py
"""

from __future__ import annotations

from _common import calculator, cu_au_system, wait
from fairchem_mcp import domain
from fairchem_mcp.session import Session

# Keep phases within this distance of the hull (eV/atom) as "accessible".
HULL_TOLERANCE = 0.01


def main() -> None:
    session = Session()
    cid = calculator(session)

    candidates = cu_au_system(session)          # [(structure_id, label), ...]
    sid_of = {label: sid for sid, label in candidates}

    # --- 1. Screen stability via the convex hull ---------------------------
    print("[1/3] screening phase stability (convex hull) ...")
    hull = wait(session, domain.start_convex_hull(
        session, [s for s, _ in candidates], cid,
        relax=True, relax_cell=True,
        labels=[l for _, l in candidates])["job_id"], timeout=300).result

    accessible = [p for p in hull["phases"]
                  if p["energy_above_hull_per_atom"] <= HULL_TOLERANCE]
    print(f"      {len(accessible)}/{len(hull['phases'])} phases within "
          f"{HULL_TOLERANCE} eV/atom of the hull: "
          f"{[p['label'] for p in accessible]}")

    # --- 2. Score the mechanical property of each accessible phase ----------
    print("[2/3] computing elastic moduli of accessible phases ...")
    scored = []
    for p in accessible:
        sid = sid_of[p["label"]]
        # Relax the cell first so the elastic tensor is taken about the minimum.
        wait(session, domain.start_relaxation(
            session, sid, cid, relax_cell=True, fmax=0.02, steps=300)["job_id"])
        el = wait(session, domain.start_elastic_scan(
            session, sid, cid, n_strains=5, max_strain=0.01)["job_id"],
            timeout=180).result
        scored.append({
            "label": p["label"],
            "above_hull": p["energy_above_hull_per_atom"],
            "stable": el.get("mechanically_stable", False),
            "E": el.get("youngs_modulus_GPa"),
            "K": el.get("bulk_modulus_GPa"),
            "pugh": el.get("pugh_ratio_G_over_K"),
        })
        print(f"      {p['label']:8s} "
              f"E={_fmt(scored[-1]['E'])} GPa  K={_fmt(scored[-1]['K'])} GPa  "
              f"Pugh={_fmt(scored[-1]['pugh'])}  Born-stable={scored[-1]['stable']}")

    # --- 3. Rank: stiffest mechanically-stable phase -----------------------
    print("[3/3] ranking by Young's modulus (Born-stable only) ...")
    viable = [s for s in scored if s["stable"] and s["E"] is not None]
    viable.sort(key=lambda s: s["E"], reverse=True)

    print("\n=== Cu-Au design result (EMT, qualitative) ===")
    if not viable:
        print("  no mechanically-stable accessible phase found.")
    else:
        win = viable[0]
        print(f"  winner: {win['label']}  "
              f"E={win['E']:.1f} GPa, K={win['K']:.1f} GPa, "
              f"Pugh={win['pugh']:.2f} "
              f"({'ductile' if win['pugh'] < 0.57 else 'brittle'}), "
              f"{win['above_hull']:.3f} eV/atom above hull")
        if len(viable) > 1:
            print("  runners-up: "
                  + ", ".join(f"{s['label']} (E={s['E']:.0f})" for s in viable[1:]))
    print("\n(EMT is qualitative; set FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p2 "
          "FAIRCHEM_MCP_EXAMPLE_TASK=omat for real numbers.)")


def _fmt(x) -> str:
    return f"{x:6.1f}" if isinstance(x, (int, float)) else f"{str(x):>6s}"


if __name__ == "__main__":
    main()
