"""Refined CO/W(110) dissociation NEB.

Fixes from the first pass: (1) use the deepest band image as the seed for the
true dissociated product minimum; (2) denser band (9 intermediate images) with
the improved-tangent method for a smoother MEP; (3) an explicit fine scan of
the first segment to put an upper bound on the (very small) barrier out of the
molecular CO* state.
"""
import json
import warnings
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import FIRE

try:
    from ase.mep import NEB, NEBTools
except ImportError:
    from ase.neb import NEB, NEBTools

from fairchem.core import FAIRChemCalculator, pretrained_mlip

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent


def log(m):
    print(m, flush=True)


log("loading uma-s-1p1 ...")
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu")


def newcalc():
    return FAIRChemCalculator(predictor, task_name="oc20")


# --- endpoints -------------------------------------------------------------
IS = read(HERE / "initial.traj")
IS.calc = newcalc()
e_is = IS.get_potential_energy()

band0 = read(HERE / "neb_band.traj", index=":")
deepest = min(band0, key=lambda a: (a.calc.results.get("energy", 0)
                                    if a.calc else 0))
# the stored band may not carry calcs; recompute to pick the deepest
es = []
for im in band0:
    im.calc = newcalc()
    es.append(im.get_potential_energy())
FS = band0[int(np.argmin(es))].copy()
FS.set_constraint(IS.constraints[0])
FS.calc = newcalc()
log("relaxing true product from deepest band image ...")
FIRE(FS, logfile=None).run(fmax=0.05, steps=300)
e_fs = FS.get_potential_energy()
co_fs = FS.get_distance(len(FS) - 2, len(FS) - 1, mic=True)
log(f"E(IS)={e_is:.3f}  E(FS')={e_fs:.3f}  dE_rxn={e_fs-e_is:+.3f} eV  C-O(FS')={co_fs:.2f} A")
write(HERE / "final_refined.traj", FS)

# --- dense NEB -------------------------------------------------------------
n_mid = 9
images = [IS] + [IS.copy() for _ in range(n_mid)] + [FS]
neb = NEB(images, climb=False, method="improvedtangent",
          allow_shared_calculator=False)
neb.interpolate(method="idpp")
for im in images[1:-1]:
    im.calc = newcalc()
log(f"dense NEB ({n_mid} images) stage 1 (no climb) ...")
FIRE(neb, logfile=None).run(fmax=0.10, steps=250)
log("stage 2 (climbing image) ...")
neb.climb = True
FIRE(neb, logfile=None).run(fmax=0.05, steps=400)

energies = np.array([im.get_potential_energy() for im in images])
rel = energies - energies[0]
ts = int(np.argmax(energies))
barrier = max(0.0, float(energies[ts] - energies[0]))
log(f"\ndense band rel energies: {[round(x,3) for x in rel]}")
log(f"TS image {ts}; forward barrier from band = {barrier:.3f} eV")

# --- fine first-segment scan: bound the barrier out of CO* -----------------
# linearly interpolate IS -> images[1] and evaluate single points
seg_e = []
p0, p1 = IS.positions, images[1].positions
for f in np.linspace(0, 1, 7):
    a = IS.copy()
    a.positions = p0 + f * (p1 - p0)
    a.calc = newcalc()
    seg_e.append(a.get_potential_energy() - e_is)
seg_e = np.array(seg_e)
seg_barrier = max(0.0, float(seg_e.max()))
log(f"first-segment scan (rel to IS): {[round(x,3) for x in seg_e]}")
log(f"upper-bound barrier out of CO* (incl. fine scan) = "
    f"{max(barrier, seg_barrier):.3f} eV")

write(HERE / "neb_band_refined.traj", images)
write(HERE / "ts_refined.traj", images[ts])
json.dump({"E_images": energies.tolist(), "rel": rel.tolist(),
           "barrier_fwd": barrier, "seg_scan": seg_e.tolist(),
           "barrier_upper_bound": max(barrier, seg_barrier),
           "dE_rxn": float(e_fs - e_is), "ts_image": ts,
           "CO_dist_FS": float(co_fs)},
          open(HERE / "neb_results_refined.json", "w"), indent=2)

# --- figure ---------------------------------------------------------------
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

nt = NEBTools(images)
fig = plt.figure(figsize=(12, 4.8))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9], hspace=0.35, wspace=0.1)
ax = fig.add_subplot(gs[0, :])
nt.plot_band(ax)
ax.set_title(f"CO dissociation on W(110):  barrier $\\leq$ "
             f"{max(barrier, seg_barrier):.2f} eV,  "
             f"$\\Delta E$ = {e_fs-e_is:+.2f} eV   (UMA uma-s-1p1 / oc20)")
for col, (atoms, ttl) in enumerate(
        [(IS, "initial: CO*"), (images[ts], f"highest image ({ts})"),
         (FS, "final: C* + O*")]):
    a = fig.add_subplot(gs[1, col])
    plot_atoms(atoms, a, rotation="-70x,0y,0z", radii=0.9)
    a.set_title(ttl, fontsize=10)
    a.set_axis_off()
handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=c, mec="k", ms=9, label=l)
           for l, c in (("W", "#2090c0"), ("C", "#404040"), ("O", "#ff2020"))]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, -0.04))
fig.savefig(HERE / "co_dissociation.png", dpi=150, bbox_inches="tight")
log(f"wrote {HERE / 'co_dissociation.png'}")
