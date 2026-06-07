"""Convergence-trend analysis over a job's snapshot history.

Turns the raw energy/force series into a small, human-readable verdict the agent
can act on: is it still making progress, has it plateaued, is it stuck or even
diverging? This is what lets the agent decide to switch optimizer or change
``fmax`` mid-run instead of waiting blindly.
"""

from __future__ import annotations

import numpy as np

# How many recent steps define "recent" behavior.
_WINDOW = 8
# Relative energy change below this (per step, averaged) counts as "flat".
_FLAT_REL = 1e-4
# Force must drop by at least this fraction over the window to count as progress.
_FORCE_PROGRESS = 0.02


def analyze_trend(history: list[dict]) -> dict:
    """Classify the recent behavior of a relaxation/MD run.

    Returns a dict with a ``label`` plus supporting numbers. Labels:
    ``starting``, ``decreasing``, ``plateaued``, ``stuck``, ``diverging``,
    ``running`` (MD, no force target).
    """
    n = len(history)
    if n < 2:
        return {"label": "starting", "n_snapshots": n}

    energies = np.array([h.get("energy") for h in history], dtype=float)
    forces = np.array(
        [h.get("max_force", np.nan) for h in history], dtype=float
    )

    window = min(_WINDOW, n)
    e_recent = energies[-window:]
    f_recent = forces[-window:]

    # Energy behavior over the recent window.
    e_span = float(e_recent[0] - e_recent[-1])  # positive == energy went down
    e_scale = max(abs(e_recent).max(), 1e-9)
    e_rel_per_step = abs(e_span) / e_scale / window

    out: dict = {
        "n_snapshots": n,
        "energy_change_recent": e_span,
        "max_force": float(forces[-1]) if not np.isnan(forces[-1]) else None,
    }

    # Force-based progress (relaxation only).
    has_forces = not np.isnan(f_recent).any()
    if has_forces:
        f0, f1 = float(f_recent[0]), float(f_recent[-1])
        force_drop = (f0 - f1) / max(f0, 1e-9)
        out["force_drop_recent"] = force_drop
        out["steps_since_best_force"] = _steps_since_min(forces)

        if e_span < -e_scale * _FLAT_REL * window:
            out["label"] = "diverging"  # energy climbing meaningfully
        elif force_drop > _FORCE_PROGRESS:
            out["label"] = "decreasing"  # forces still dropping nicely
        elif e_rel_per_step < _FLAT_REL and out["steps_since_best_force"] >= window:
            out["label"] = "stuck"  # no force improvement for a full window
        else:
            out["label"] = "plateaued"  # crawling; small but nonzero progress
        return out

    # MD or force-free: just report energy drift.
    out["label"] = "running"
    return out


def _steps_since_min(forces: np.ndarray) -> int:
    """How many steps since the lowest max-force seen so far."""
    if len(forces) == 0 or np.isnan(forces).all():
        return 0
    best_idx = int(np.nanargmin(forces))
    return int(len(forces) - 1 - best_idx)
