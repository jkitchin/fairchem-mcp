"""Bias potentials for multi-minimum search (deflation / flooding).

A relaxed geometry is a local minimum of the potential-energy surface (PES). To
find a *different* relaxed geometry we relax again on a **biased** PES that repels
the optimizer away from minima we have already found, then *polish* on the true
PES to settle exactly onto the new minimum. This is the deflation idea (Farrell,
Birkisson & Funke, 2015) and the static-bias / filled-function idea (additive
Gaussian bumps, as in metadynamics) — the same repulsion kernels POUNCE's
``find_minima`` uses, but applied here at the *gradient* level so any ASE
gradient optimizer (FIRE/LBFGS/BFGS) can drive the escape.

We deliberately do **not** reuse POUNCE's interior-point ``minimize`` as the
inner solver: gradient descent with force-based convergence is the right tool for
a PES (no bounds needed, and it tolerates the exact translational/rotational zero
modes a Hessian-based saddle check would trip over). What we reuse is the *escape
mechanism*, ported as a thin ``DeflatedCalculator`` that wraps the real
calculator and adds a repulsion energy + force from a list of known minima.

Two kernels, selected by ``kernel=``:

* ``"flooding"`` (default) — additive Gaussian bumps
  ``R(x) = Σ_k A·exp(-½‖(x-c_k)/σ‖²)``. Smooth and bounded; gentle and robust.
* ``"deflation"`` — additive inverse-distance poles
  ``R(x) = Σ_k η/‖x-c_k‖^p``. A barrier the optimizer cannot sink back into;
  sharper escape but the force diverges near a known point (floored for safety).

Centers ``c_k`` and coordinates ``x`` are flattened atomic positions (length 3N,
in Å), so ``sigma`` is an Å length scale and ``amplitude``/``eta`` are eV.
"""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


def flooding_terms(x, centers, sigma, amplitude):
    """Energy and gradient of the Gaussian flooding bias at ``x``.

    Returns ``(energy, grad)`` where ``grad = dR/dx`` (length 3N). The force
    contribution the optimizer should feel is ``-grad``.
    """
    energy = 0.0
    grad = np.zeros_like(x)
    inv_s2 = 1.0 / (sigma * sigma)
    for c in centers:
        d = x - c
        term = amplitude * np.exp(-0.5 * float(d @ d) * inv_s2)
        energy += term
        grad += -term * d * inv_s2  # d/dx [A exp(-½‖d/σ‖²)]
    return energy, grad


def deflation_terms(x, centers, eta, power, floor=1e-3):
    """Energy and gradient of the inverse-distance (pole) deflation bias.

    ``floor`` clamps the distance so the pole stays finite at a known minimum.
    """
    energy = 0.0
    grad = np.zeros_like(x)
    for c in centers:
        d = x - c
        r = max(float(np.linalg.norm(d)), floor)
        energy += eta / r**power
        grad += -eta * power * r ** (-power - 2) * d  # d/dx [η r^-p]
    return energy, grad


class DeflatedCalculator(Calculator):
    """Wrap a base ASE calculator and add a repulsion bias from known minima.

    The wrapped calculator's energy/forces are computed unchanged; the bias from
    ``centers`` (flattened positions of already-found minima) is added on top.
    Relaxing on this calculator escapes the known basins; relax again on the
    *base* calculator afterwards to polish onto the true minimum.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        base,
        centers,
        *,
        kernel="flooding",
        sigma=0.4,
        amplitude=1.0,
        eta=1.0,
        power=2,
        floor=1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if kernel not in ("flooding", "deflation"):
            raise ValueError(
                f"unknown kernel {kernel!r}; choose 'flooding' or 'deflation'"
            )
        self.base = base
        self.centers = [np.asarray(c, dtype=float).ravel() for c in centers]
        self.kernel = kernel
        self.sigma = float(sigma)
        self.amplitude = float(amplitude)
        self.eta = float(eta)
        self.power = float(power)
        self.floor = float(floor)

    def _bias(self, x):
        if self.kernel == "flooding":
            return flooding_terms(x, self.centers, self.sigma, self.amplitude)
        return deflation_terms(x, self.centers, self.eta, self.power, self.floor)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        e0 = float(self.base.get_potential_energy(atoms))
        f0 = np.asarray(self.base.get_forces(atoms), dtype=float)
        x = atoms.get_positions().ravel()
        e_bias, grad = self._bias(x)
        self.results = {
            "energy": e0 + e_bias,
            "forces": f0 - grad.reshape(-1, 3),  # force = -dE/dx
        }


def same_minimum(e1, pos1, e2, pos2, energy_tol, rmsd_tol):
    """Energy + simple-RMSD dedup: two geometries are the same minimum when their
    energies agree within ``energy_tol`` (eV) *and* their (unaligned) RMSD is
    within ``rmsd_tol`` (Å)."""
    if abs(e1 - e2) > energy_tol:
        return False
    return rmsd(pos1, pos2) <= rmsd_tol


def fingerprint(atoms):
    """A rotation/translation/permutation-invariant structure descriptor.

    The sorted vector of all pairwise interatomic distances (minimum-image for
    periodic cells). Two geometries that differ only by a rigid move or an atom
    relabeling have (nearly) identical fingerprints — unlike a raw-coordinate
    RMSD, which a free cluster defeats simply by rotating. Use this comparator
    for free clusters and molecules; the cheaper raw RMSD is fine when a fixed
    frame is enforced (e.g. an adsorbate on a frozen slab).
    """
    mic = bool(getattr(atoms, "pbc", None) is not None and any(atoms.pbc))
    d = atoms.get_all_distances(mic=mic)
    iu = np.triu_indices(len(atoms), k=1)
    return np.sort(d[iu])


def fingerprints_match(f1, f2, tol):
    """True if two fingerprints agree to within ``tol`` (Å) on every distance."""
    a = np.asarray(f1, dtype=float)
    b = np.asarray(f2, dtype=float)
    if a.shape != b.shape:
        return False
    return float(np.abs(a - b).max()) <= tol


def rmsd(pos1, pos2):
    """Root-mean-square deviation between two flattenable position arrays (Å).

    Unaligned by design (the chosen 'simple RMSD' metric): rigid moves are not
    factored out, but combined with the energy gate this reliably separates
    distinct basins in practice while staying cheap and dependency-free.
    """
    a = np.asarray(pos1, dtype=float).reshape(-1, 3)
    b = np.asarray(pos2, dtype=float).reshape(-1, 3)
    d = a - b
    return float(np.sqrt((d * d).sum() / len(a)))
