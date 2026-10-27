"""Phonon dispersion for fcc Cu via fairchem-mcp's start_phonons.

A finite-displacement phonon calculation on a bulk crystal. The same force
constants that give the gamma-point stability check are Fourier-summed along a
Brillouin-zone k-path to produce a full dispersion -- no extra force evaluations.
The k-path is chosen automatically from the cell's Bravais lattice (ASE's
Setyawan-Curtarolo special points), so fcc Cu gets a sensible Gamma-X-W-K-Gamma-L
traverse without naming one.

Runs out of the box on the fast EMT calculator (no GPU/model download). Point it
at a UMA model for publication-grade numbers:

    FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p1 FAIRCHEM_MCP_EXAMPLE_TASK=omat \\
        python examples/phonons/phonon_dispersion.py

The dispersion block returned by get_results carries everything needed to plot:
distances, per-branch frequencies_THz, and high_symmetry_points (tick distances
+ labels). This script renders it to recordings/cu_phonon_dispersion.png.
"""

from __future__ import annotations

import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fairchem_mcp import domain
from fairchem_mcp.session import Session

THZ_TO_CM1 = 33.35641  # 1 THz -> cm^-1

MODEL = os.environ.get("FAIRCHEM_MCP_EXAMPLE_MODEL")  # e.g. uma-s-1p1; None -> EMT
TASK = os.environ.get("FAIRCHEM_MCP_EXAMPLE_TASK", "omat")
# Use the UMA-relaxed lattice constant when on a model, else EMT's.
A0 = float(os.environ.get("FAIRCHEM_MCP_EXAMPLE_A0", "3.616" if MODEL else "3.61"))


def main() -> None:
    session = Session()
    sid = domain.build_structure(
        session,
        {"kind": "bulk", "name": "Cu", "crystalstructure": "fcc", "a": A0},
    )["structure_id"]

    if MODEL:
        print(f"loading {MODEL} (task={TASK}) ...")
        cid = domain.load_model(session, model=MODEL, task=TASK)["calculator_id"]
        label = f"UMA ({MODEL})"
    else:
        cid = domain.attach_emt(session)["calculator_id"]
        label = "EMT"

    print(f"starting phonons on fcc Cu (a={A0} Å), {label} ...")
    res = domain.start_phonons(session, sid, cid, supercell=(4, 4, 4), band_points=200)
    job = session.get_job(res["job_id"])
    total = res["n_displacements"]
    while job.is_active():
        print(f"  displacement {job.step}/{total}", end="\r", flush=True)
        time.sleep(0.2)
    print()

    result = job.result
    disp = result["dispersion"]
    print(f"stable: {result['stable']}  (imaginary modes: {result['n_imaginary_modes']})")
    print(f"k-path: {disp['path']}")
    print(f"max frequency: {disp['max_frequency_THz']:.3f} THz "
          f"({disp['max_frequency_THz'] * THZ_TO_CM1:.1f} cm^-1)")

    plot(disp, result.get("dos"), label)


def plot(disp: dict, dos: dict | None, label: str) -> None:
    x = np.asarray(disp["distances"])
    freqs = np.asarray(disp["frequencies_THz"])  # (nk, nbands)
    ticks = disp["high_symmetry_points"]["distances"]
    labels = disp["high_symmetry_points"]["labels"]

    if dos is not None:
        fig, (ax, axdos) = plt.subplots(
            1, 2, figsize=(9.5, 5), sharey=True,
            gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05},
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        axdos = None

    for b in range(freqs.shape[1]):
        ax.plot(x, freqs[:, b], color="#1f4e79", lw=1.6)
    for xt in ticks:
        ax.axvline(xt, color="0.7", lw=0.8)
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel("Frequency (THz)", fontsize=12)
    ax.set_title(f"fcc Cu phonon dispersion + DOS — {label}", fontsize=12)

    if axdos is not None:
        de = np.asarray(dos["frequencies_THz"])
        dw = np.asarray(dos["dos"])
        axdos.plot(dw, de, color="#1f4e79", lw=1.4)
        axdos.fill_betweenx(de, 0, dw, color="#1f4e79", alpha=0.25)
        axdos.set_xticks([])
        axdos.set_xlim(left=0)
        axdos.set_xlabel("DOS", fontsize=11)
        axdos.axhline(0, color="0.5", lw=0.8, ls="--")
        ax3 = axdos.twinx()
        ax3.set_ylim(np.array(ax.get_ylim()) * THZ_TO_CM1)
        ax3.set_ylabel(r"Frequency (cm$^{-1}$)", fontsize=11)
    else:
        ax2 = ax.twinx()
        ax2.set_ylim(np.array(ax.get_ylim()) * THZ_TO_CM1)
        ax2.set_ylabel(r"Frequency (cm$^{-1}$)", fontsize=11)

    fig.tight_layout()
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "recordings", "cu_phonon_dispersion.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
