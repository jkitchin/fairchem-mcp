"""CO dissociation barrier on W(110) via climbing-image NEB (UMA / oc20).

Initial state : CO adsorbed molecularly (C down, O up) on the surface.
Final state   : dissociated C* + O* in separate surface sites.
Path          : IDPP-interpolated NEB, then climbing-image refinement.
Barrier       : E(transition state) - E(initial state).
"""
import json
import warnings
from pathlib import Path

import numpy as np
from ase import Atom
from ase.build import bcc110
from ase.constraints import FixAtoms
from ase.io import write
from ase.optimize import FIRE

try:
    from ase.mep import NEB, NEBTools
except ImportError:                       # older ASE
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


def relax(atoms, fmax=0.05, steps=300):
    atoms.calc = newcalc()
    FIRE(atoms, logfile=None).run(fmax=fmax, steps=steps)
    return atoms.get_potential_energy()


# --- W(110) slab: 3x3, 4 layers, bottom 2 fixed ---------------------------
slab = bcc110("W", size=(3, 3, 4), a=3.165, vacuum=10.0, orthogonal=False)
slab.pbc = True
z = slab.positions[:, 2]
uz = np.unique(np.round(z, 2))
fixed = np.where(z < uz[2] - 0.1)[0]                 # bottom 2 of 4 layers
slab.set_constraint(FixAtoms(indices=fixed.tolist()))
zmax = z.max()
log(f"slab: {len(slab)} W atoms, {len(fixed)} fixed; layer z = {uz}")

# pick a central top-layer W and a neighbour ~2.5-3.4 A away in-plane
topidx = np.where(z > zmax - 0.5)[0]
toppos = slab.positions[topidx]
center = slab.cell[:2, :2].sum(0) / 2
ci = topidx[np.argmin(np.linalg.norm(toppos[:, :2] - center, axis=1))]
cpos = slab.positions[ci]
d = np.linalg.norm(toppos[:, :2] - cpos[:2], axis=1)
oi = topidx[np.where((d > 2.4) & (d < 3.5))[0][0]]
opos = slab.positions[oi]
mid = 0.5 * (cpos + opos)
log(f"C site {cpos[:2].round(2)}, O site {opos[:2].round(2)}, "
    f"separation {np.linalg.norm(cpos[:2]-opos[:2]):.2f} A")

# --- initial state: molecular CO upright over the bridge midpoint ---------
IS = slab.copy()
IS.append(Atom("C", [mid[0], mid[1], zmax + 1.45]))
IS.append(Atom("O", [mid[0], mid[1], zmax + 1.45 + 1.16]))
log("relaxing initial state (molecular CO*) ...")
e_is = relax(IS)
co_d = IS.get_distance(len(IS) - 2, len(IS) - 1, mic=True)
log(f"E(IS) = {e_is:.3f} eV,  C-O = {co_d:.2f} A")

# --- final state: dissociated C* and O* in separate sites -----------------
FS = slab.copy()
FS.append(Atom("C", [cpos[0], cpos[1], zmax + 1.05]))
FS.append(Atom("O", [opos[0], opos[1], zmax + 1.05]))
log("relaxing final state (C* + O*) ...")
e_fs = relax(FS)
co_d_fs = FS.get_distance(len(FS) - 2, len(FS) - 1, mic=True)
log(f"E(FS) = {e_fs:.3f} eV,  C-O = {co_d_fs:.2f} A,  dE_rxn = {e_fs-e_is:+.3f} eV")

write(HERE / "initial.traj", IS)
write(HERE / "final.traj", FS)

# --- NEB ------------------------------------------------------------------
n_mid = 5
images = [IS] + [IS.copy() for _ in range(n_mid)] + [FS]
neb = NEB(images, climb=False, method="aseneb", allow_shared_calculator=False)
neb.interpolate(method="idpp")
for im in images[1:-1]:
    im.calc = newcalc()

log("NEB stage 1 (no climb, fmax 0.10) ...")
FIRE(neb, logfile=None).run(fmax=0.10, steps=200)
log("NEB stage 2 (climbing image, fmax 0.05) ...")
neb.climb = True
FIRE(neb, logfile=None).run(fmax=0.05, steps=300)

# --- analyse --------------------------------------------------------------
energies = np.array([im.get_potential_energy() for im in images])
rel = energies - energies[0]
ts = int(np.argmax(energies))
barrier_fwd = energies[ts] - energies[0]
barrier_rev = energies[ts] - energies[-1]
log(f"\nForward barrier  (CO* -> C*+O*) = {barrier_fwd:.3f} eV")
log(f"Reverse barrier  (C*+O* -> CO*) = {barrier_rev:.3f} eV")
log(f"Reaction energy                 = {energies[-1]-energies[0]:+.3f} eV")
log(f"TS is image {ts} of {len(images)-1}")

write(HERE / "neb_band.traj", images)
write(HERE / "ts.traj", images[ts])
json.dump({"E_images": energies.tolist(), "rel": rel.tolist(),
           "barrier_fwd": float(barrier_fwd), "barrier_rev": float(barrier_rev),
           "dE_rxn": float(energies[-1] - energies[0]), "ts_image": ts,
           "CO_dist_IS": float(co_d), "CO_dist_FS": float(co_d_fs)},
          open(HERE / "neb_results.json", "w"), indent=2)

# --- figure: MEP profile + IS / TS / FS structures ------------------------
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

nt = NEBTools(images)
fig = plt.figure(figsize=(12, 4.8))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9], hspace=0.35, wspace=0.1)

ax = fig.add_subplot(gs[0, :])
nt.plot_band(ax)
ax.set_title(f"CO dissociation on W(110): forward barrier "
             f"{barrier_fwd:.2f} eV, $\\Delta E$ = {energies[-1]-energies[0]:+.2f} eV "
             f"(UMA uma-s-1p1 / oc20)")

for col, (atoms, ttl) in enumerate(
        [(IS, "initial: CO*"), (images[ts], f"TS (image {ts})"), (FS, "final: C*+O*")]):
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
