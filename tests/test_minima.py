"""Tests for the multi-minimum search (deflation / flooding).

A custom double-well calculator gives a deterministic PES with exactly two
minima (a single atom at x=+1 and x=-1, degenerate in energy), so we can verify
the escape mechanism end-to-end without relying on a metal cluster's isomers.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from fairchem_mcp import domain
from fairchem_mcp.deflation import (
    DeflatedCalculator,
    fingerprint,
    fingerprints_match,
    rmsd,
    same_minimum,
)
from fairchem_mcp.jobs import JobManager
from fairchem_mcp.session import Session

from conftest import wait_for


class DoubleWell(Calculator):
    """E = (x²-1)² + y² + z² for a single atom: minima at x=±1, max at x=0."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        x, y, z = atoms.get_positions()[0]
        energy = (x * x - 1.0) ** 2 + y * y + z * z
        fx = -(2.0 * (x * x - 1.0) * 2.0 * x)
        forces = np.array([[fx, -2.0 * y, -2.0 * z]])
        self.results = {"energy": float(energy), "forces": forces}


def _double_well_atom():
    # Start off-center toward the +1 well so the first relaxation is unambiguous.
    return Atoms("H", positions=[[0.3, 0.0, 0.0]])


# --- unit: dedup metric ----------------------------------------------------

def test_same_minimum_separates_degenerate_basins():
    plus = np.array([[1.0, 0.0, 0.0]])
    minus = np.array([[-1.0, 0.0, 0.0]])
    # Same energy, but far apart -> distinct minima.
    assert not same_minimum(0.0, plus, 0.0, minus, energy_tol=0.05, rmsd_tol=0.1)
    # Identical -> same minimum.
    assert same_minimum(0.0, plus, 0.0, plus.copy(), energy_tol=0.05, rmsd_tol=0.1)
    assert rmsd(plus, minus) == 2.0


# --- unit: the bias calculator pushes away from a center -------------------

def test_deflated_calculator_repels_from_center():
    base = DoubleWell()
    center = np.array([1.0, 0.0, 0.0])  # bump sits on the +1 minimum
    atoms = _double_well_atom()
    atoms.calc = DeflatedCalculator(
        base, [center], kernel="flooding", sigma=0.6, amplitude=4.0
    )
    # Biased energy exceeds the bare PES near the bumped minimum.
    bare = DoubleWell()
    atoms_bare = _double_well_atom()
    atoms_bare.calc = bare
    assert atoms.get_potential_energy() > atoms_bare.get_potential_energy()
    # The net force at x=0.3 now points toward -x (away from the +1 bump).
    assert atoms.get_forces()[0, 0] < 0.0


# --- integration: flooding finds both wells --------------------------------

def test_flooding_finds_both_minima():
    session = Session()
    sid = session.add_structure(_double_well_atom())
    cid = session.add_calculator(DoubleWell())

    res = domain.start_minima_search(
        session, sid, cid, n_minima=2, kernel="flooding",
        optimizer="FIRE", fmax=0.01, steps=300,
        sigma=0.6, amplitude=4.0, energy_tol=0.05, rmsd_tol=0.2,
    )
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: not job.is_active(), timeout=60)

    assert job.status_dict()["status"] == "converged", job.status_dict()
    assert job.result["n_found"] == 2
    # Two distinct registered structures sitting at x = +1 and x = -1.
    xs = sorted(
        round(session.get_structure(m["structure_id"]).get_positions()[0, 0], 2)
        for m in job.result["minima"]
    )
    assert xs == [-1.0, 1.0]


def test_deflation_kernel_also_finds_both():
    session = Session()
    sid = session.add_structure(_double_well_atom())
    cid = session.add_calculator(DoubleWell())

    res = domain.start_minima_search(
        session, sid, cid, n_minima=2, kernel="deflation",
        optimizer="FIRE", fmax=0.01, steps=300,
        eta=2.0, power=2.0, energy_tol=0.05, rmsd_tol=0.2,
    )
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: not job.is_active(), timeout=60)
    assert job.result["n_found"] == 2


def test_fingerprint_is_rotation_translation_permutation_invariant():
    from ase.cluster import Icosahedron

    a = Icosahedron("Cu", noshells=2)
    f0 = fingerprint(a)

    rotated = a.copy()
    rotated.rotate(37, "z", center="COP")
    rotated.translate([1.5, -2.0, 0.7])
    assert fingerprints_match(f0, fingerprint(rotated), tol=1e-6)

    permuted = a.copy()
    order = list(range(len(a)))[::-1]
    permuted = permuted[order]
    assert fingerprints_match(f0, fingerprint(permuted), tol=1e-6)


def test_fingerprint_comparator_collapses_rotated_duplicates():
    """A free Cu13 cluster: flooding produces rotated copies of one isomer; the
    fingerprint comparator must dedup them where raw RMSD does not."""
    from ase.calculators.emt import EMT
    from ase.cluster import Icosahedron

    session = Session()
    sid = session.add_structure(Icosahedron("Cu", noshells=2))
    cid = session.add_calculator(EMT())

    res = domain.start_minima_search(
        session, sid, cid, n_minima=6, kernel="flooding",
        comparator="fingerprint", sigma=0.5, amplitude=0.8,
        fmax=0.02, steps=400, energy_tol=0.02, rmsd_tol=0.1, escape_rattle=0.3,
    )
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: not job.is_active(), timeout=120)
    # Rotated copies of a single isomer must not be counted as distinct minima.
    energies = [m["energy"] for m in job.result["minima"]]
    spread = max(energies) - min(energies) if energies else 0.0
    assert job.result["n_found"] <= 2, energies
    assert spread <= 0.02, energies


def test_basinhopping_finds_both_wells():
    session = Session()
    sid = session.add_structure(_double_well_atom())
    cid = session.add_calculator(DoubleWell())

    res = domain.start_minima_search(
        session, sid, cid, n_minima=2, kernel="basinhopping",
        bh_step=1.2, bh_temperature=1.0, optimizer="FIRE", fmax=0.01, steps=200,
        energy_tol=0.05, rmsd_tol=0.2, patience=20, seed=1,
    )
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: not job.is_active(), timeout=60)
    assert job.result["n_found"] == 2
    xs = sorted(
        round(session.get_structure(m["structure_id"]).get_positions()[0, 0], 2)
        for m in job.result["minima"]
    )
    assert xs == [-1.0, 1.0]


def test_minima_search_abort():
    session = Session()
    sid = session.add_structure(_double_well_atom())
    cid = session.add_calculator(DoubleWell())

    res = domain.start_minima_search(
        session, sid, cid, n_minima=2, fmax=0.01, steps=300, step_delay=0.05,
    )
    job = session.get_job(res["job_id"])
    assert wait_for(lambda: job.status == "running")
    JobManager(session).steer(job, "abort")
    assert wait_for(lambda: job.status == "aborted")
