"""Refined CH3 -> CH2 + H NEB on Ni(111): resample the first-pass band to a
denser set of images (better saddle resolution) and converge the climbing
image tightly. Reports whether the optimisation actually converged.
"""
import json
import warnings
from pathlib import Path

import numpy as np
from ase.io import read, write
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


# --- resample the first-pass band to N_total images along its path ---------
band0 = read(HERE / "neb_band.traj", index=":")
cons = band0[0].constraints
P = [im.positions for im in band0]
s = np.zeros(len(P))
for i in range(1, len(P)):
    s[i] = s[i - 1] + np.linalg.norm(P[i] - P[i - 1])

N_total = 13
targets = np.linspace(0, s[-1], N_total)
images = []
for t in targets:
    j = int(np.clip(np.searchsorted(s, t) - 1, 0, len(P) - 2))
    f = (t - s[j]) / (s[j + 1] - s[j])
    im = band0[0].copy()
    im.positions = P[j] + f * (P[j + 1] - P[j])
    im.set_constraint(cons)
    images.append(im)
log(f"resampled {len(band0)} -> {N_total} images along the path")

for im in images:
    im.calc = newcalc()

neb = NEB(images, climb=False, method="improvedtangent",
          allow_shared_calculator=False)
log("stage 1 (smooth, no climb, fmax 0.10) ...")
c1 = FIRE(neb, logfile=None).run(fmax=0.10, steps=300)
log("stage 2 (climbing image, fmax 0.03) ...")
neb.climb = True
c2 = FIRE(neb, logfile=None).run(fmax=0.03, steps=600)
log(f"converged: stage1={c1}, stage2={c2}")

iC, iH = json.loads((HERE / "neb_results.json").read_text())["iC"], \
    json.loads((HERE / "neb_results.json").read_text())["iH"]
energies = np.array([im.get_potential_energy() for im in images])
rel = energies - energies[0]
ts = int(np.argmax(energies))
barrier_fwd = float(energies[ts] - energies[0])
barrier_rev = float(energies[ts] - energies[-1])
ch_ts = images[ts].get_distance(iC, iH, mic=True)
log(f"\nband rel energies: {[round(x,3) for x in rel]}")
log(f"TS = image {ts}/{N_total-1}; C-H(TS) = {ch_ts:.2f} A")
log(f"Forward barrier = {barrier_fwd:.3f} eV;  reverse = {barrier_rev:.3f} eV;  "
    f"dE_rxn = {energies[-1]-energies[0]:+.3f} eV")

write(HERE / "neb_band_refined.traj", images)
write(HERE / "ts_refined.traj", images[ts])
json.dump({"E_images": energies.tolist(), "rel": rel.tolist(),
           "barrier_fwd": barrier_fwd, "barrier_rev": barrier_rev,
           "dE_rxn": float(energies[-1] - energies[0]), "ts_image": ts,
           "CH_ts": float(ch_ts), "converged_stage2": bool(c2),
           "iC": iC, "iH": iH},
          open(HERE / "neb_results_refined.json", "w"), indent=2)

# --- figure ---------------------------------------------------------------
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

IS, FS = images[0], images[-1]
s2 = [0.0]
for i in range(1, len(images)):
    s2.append(s2[-1] + np.linalg.norm(images[i].positions - images[i - 1].positions))
s2 = np.array(s2)
fig = plt.figure(figsize=(12, 5.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.5, wspace=0.05)
ax = fig.add_subplot(gs[0, :])
ss = np.linspace(s2.min(), s2.max(), 300)
ax.plot(ss, np.interp(ss, s2, rel), "-", color="#1f77b4", lw=1.5)
ax.plot(s2, rel, "o", color="#1f77b4", mec="k", ms=7)
ax.plot(s2[ts], rel[ts], "*", color="crimson", ms=18, zorder=3)
ax.axhline(0, color="0.6", lw=0.7, ls=":")
ax.set_ylabel("energy [eV]")
ax.set_xlabel("reaction coordinate [Å]")
ax.set_title(f"CH$_3$ → CH$_2$ + H on Ni(111):  $E_a$ = {barrier_fwd:.2f} eV,  "
             f"$\\Delta E_{{rxn}}$ = {energies[-1]-energies[0]:+.2f} eV   "
             f"(UMA uma-s-1p1 / oc20)")
rad = lambda a: np.array([{"Ni": 0.9, "C": 0.7, "H": 0.45}[t] for t in
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
