"""CH3* -> CH2* + H* dehydrogenation barrier on Ni(111) (UMA / oc20).

Initial : CH3 adsorbed at an fcc hollow.
Final   : CH2 + a dissociated H in a neighbouring hollow (one C-H broken).
Path    : IDPP-interpolated NEB + climbing image; barrier from the CI maximum.
A fine first-segment scan guards against a sub-resolution early barrier.
"""
import json
import warnings
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms
from ase.io import write
from ase.optimize import FIRE

try:
    from ase.mep import NEB
except ImportError:
    from ase.neb import NEB

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


# --- Ni(111) slab: 3x3, 4 layers, bottom 2 fixed --------------------------
slab = fcc111("Ni", size=(3, 3, 4), a=3.52, vacuum=10.0)
slab.pbc = True
z = slab.positions[:, 2]
uz = np.unique(np.round(z, 2))
fixed = np.where(z < uz[2] - 0.1)[0]                 # bottom 2 of 4 layers
slab.set_constraint(FixAtoms(indices=fixed.tolist()))
zsurf = z.max()
log(f"slab: {len(slab)} Ni, {len(fixed)} fixed; layers z={uz}")

# --- initial state: CH3 at fcc hollow -------------------------------------
th = np.deg2rad([0, 120, 240])
Hpos = np.array([[1.05 * np.cos(t), 1.05 * np.sin(t), 0.3] for t in th])
ch3 = Atoms("CH3", positions=np.vstack([[0, 0, 0], Hpos]))
IS = slab.copy()
add_adsorbate(IS, ch3, height=1.6, position="fcc", mol_index=0)
iC, iH = len(IS) - 4, len(IS) - 1            # C index, dissociating-H index
log("relaxing initial state (CH3*) ...")
e_is = relax(IS)
log(f"E(IS) = {e_is:.3f} eV,  C-H(diss) = {IS.get_distance(iC, iH, mic=True):.2f} A")
write(HERE / "initial.traj", IS)

# --- final state: move one H to the neighbouring hollow, relax ------------
FS = IS.copy()
shift = FS.cell[0] / 3                         # nearest-neighbour surface vector
newH = FS.positions[iC] + shift
newH[2] = zsurf + 1.0
FS.positions[iH] = newH
FS.set_constraint(FixAtoms(indices=fixed.tolist()))
log("relaxing final state (CH2* + H*) ...")
e_fs = relax(FS)
log(f"E(FS) = {e_fs:.3f} eV,  C-H(diss) = {FS.get_distance(iC, iH, mic=True):.2f} A,  "
    f"dE_rxn = {e_fs-e_is:+.3f} eV")
write(HERE / "final.traj", FS)

# --- NEB with climbing image ----------------------------------------------
n_mid = 7
images = [IS] + [IS.copy() for _ in range(n_mid)] + [FS]
neb = NEB(images, climb=False, method="improvedtangent",
          allow_shared_calculator=False)
neb.interpolate(method="idpp")
for im in images[1:-1]:
    im.calc = newcalc()
log(f"NEB ({n_mid} images) stage 1 (no climb, fmax 0.10) ...")
FIRE(neb, logfile=None).run(fmax=0.10, steps=250)
log("stage 2 (climbing image, fmax 0.05) ...")
neb.climb = True
FIRE(neb, logfile=None).run(fmax=0.05, steps=400)

energies = np.array([im.get_potential_energy() for im in images])
rel = energies - energies[0]
ts = int(np.argmax(energies))
barrier_fwd = float(energies[ts] - energies[0])
barrier_rev = float(energies[ts] - energies[-1])
ch_ts = images[ts].get_distance(iC, iH, mic=True)
log(f"\nband rel energies: {[round(x,3) for x in rel]}")
log(f"TS = image {ts}; C-H(TS) = {ch_ts:.2f} A")
log(f"Forward barrier (CH3* -> CH2*+H*) = {barrier_fwd:.3f} eV")
log(f"Reverse barrier (CH2*+H* -> CH3*) = {barrier_rev:.3f} eV")
log(f"Reaction energy                   = {energies[-1]-energies[0]:+.3f} eV")

# fine first-segment scan (bound any sub-resolution early barrier)
seg = []
p0, p1 = IS.positions, images[1].positions
for f in np.linspace(0, 1, 7):
    a = IS.copy()
    a.positions = p0 + f * (p1 - p0)
    a.calc = newcalc()
    seg.append(a.get_potential_energy() - e_is)
log(f"first-segment scan (rel IS): {[round(x,3) for x in seg]}")

write(HERE / "neb_band.traj", images)
write(HERE / "ts.traj", images[ts])
json.dump({"E_images": energies.tolist(), "rel": rel.tolist(),
           "barrier_fwd": barrier_fwd, "barrier_rev": barrier_rev,
           "dE_rxn": float(energies[-1] - energies[0]), "ts_image": ts,
           "CH_ts": float(ch_ts), "seg_scan": [float(x) for x in seg],
           "iC": iC, "iH": iH},
          open(HERE / "neb_results.json", "w"), indent=2)

# --- figure ---------------------------------------------------------------
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

s = [0.0]
for i in range(1, len(images)):
    s.append(s[-1] + np.linalg.norm(images[i].positions - images[i - 1].positions))
s = np.array(s)

fig = plt.figure(figsize=(12, 5.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.5, wspace=0.05)
ax = fig.add_subplot(gs[0, :])
ss = np.linspace(s.min(), s.max(), 300)
ax.plot(ss, np.interp(ss, s, rel), "-", color="#1f77b4", lw=1.5)
ax.plot(s, rel, "o", color="#1f77b4", mec="k", ms=7)
ax.plot(s[ts], rel[ts], "*", color="crimson", ms=18, zorder=3)
ax.axhline(0, color="0.6", lw=0.7, ls=":")
ax.set_ylabel("energy [eV]")
ax.set_xlabel("reaction coordinate [Å]")
ax.set_title(f"CH$_3$ → CH$_2$ + H on Ni(111):  $E_a$ = {barrier_fwd:.2f} eV,  "
             f"$\\Delta E_{{rxn}}$ = {energies[-1]-energies[0]:+.2f} eV   "
             f"(UMA uma-s-1p1 / oc20)")

rad = lambda a: np.array([{"Ni": 0.9, "C": 0.7, "H": 0.45}[s] for s in
                          a.get_chemical_symbols()])
for col, idx, ttl in [(0, 0, f"initial: CH$_3$* (C–H {IS.get_distance(iC,iH,mic=True):.2f} Å)"),
                      (1, ts, f"TS (C–H {ch_ts:.2f} Å)"),
                      (2, len(images) - 1, f"final: CH$_2$* + H* (C–H {FS.get_distance(iC,iH,mic=True):.2f} Å)")]:
    a = fig.add_subplot(gs[1, col])
    plot_atoms(images[idx], a, rotation="-65x,0y,0z", radii=rad(images[idx]))
    a.set_title(ttl, fontsize=9.5)
    a.set_axis_off()
handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=c, mec="k", ms=9, label=l)
           for l, c in (("Ni", "#a0a0c0"), ("C", "#404040"), ("H", "#ffffff"))]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))
fig.savefig(HERE / "ch3_dehydrogenation.png", dpi=150, bbox_inches="tight")
log(f"wrote {HERE / 'ch3_dehydrogenation.png'}")
