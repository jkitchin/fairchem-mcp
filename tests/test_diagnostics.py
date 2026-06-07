"""Tests for diagnostics.analyze_trend."""

from __future__ import annotations

from fairchem_mcp.diagnostics import analyze_trend


def _hist(energies, forces):
    return [{"step": i, "energy": e, "max_force": f} for i, (e, f) in enumerate(zip(energies, forces))]


def test_starting():
    assert analyze_trend([])["label"] == "starting"
    assert analyze_trend([{"step": 0, "energy": 1.0, "max_force": 1.0}])["label"] == "starting"


def test_decreasing():
    energies = [10 - i for i in range(12)]
    forces = [5.0 * (0.7**i) for i in range(12)]  # forces dropping fast
    out = analyze_trend(_hist(energies, forces))
    assert out["label"] == "decreasing"


def test_stuck():
    # Energy flat and force not improving for a full window.
    energies = [1.0] * 12
    forces = [0.5] * 12
    out = analyze_trend(_hist(energies, forces))
    assert out["label"] == "stuck"


def test_diverging():
    energies = [i * 1.0 for i in range(12)]  # energy climbing
    forces = [1.0 + 0.1 * i for i in range(12)]
    out = analyze_trend(_hist(energies, forces))
    assert out["label"] == "diverging"


def test_md_running_without_forces():
    history = [{"step": i, "energy": 1.0 + 0.01 * i} for i in range(12)]
    out = analyze_trend(history)
    assert out["label"] == "running"
