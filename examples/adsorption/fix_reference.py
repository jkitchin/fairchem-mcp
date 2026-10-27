"""Correct the O adsorption energies to a consistent oc20 1/2-O2 reference.

The screen (o_on_ptni.py) referenced O to 1/2 E(O2) computed with the *omol*
task, which lives on a different DFT energy scale than the *oc20* slab+O
energies -> a large constant offset. The correct reference uses O2 computed
with the SAME oc20 task. Because the reference is a constant, this is a pure
shift of every E_ads (site ranking and geometry are unchanged), so we just
recompute O2 with oc20 and re-plot.
"""
import json
import warnings
from pathlib import Path

import numpy as np
from ase.build import molecule
from ase.optimize import LBFGS
from ase.io import read
from ase.visualize.plot import plot_atoms

from fairchem.core import FAIRChemCalculator, pretrained_mlip

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu")
calc = FAIRChemCalculator(predictor, task_name="oc20")

# O2 reference on the oc20 energy scale (periodic box, same task as the slab)
o2 = molecule("O2")
o2.set_distance(0, 1, 1.21)
o2.center(vacuum=8.0)
o2.pbc = True
o2.calc = calc
LBFGS(o2, logfile=None).run(fmax=0.02, steps=200)
e_o2_oc20 = o2.get_potential_energy()
print(f"E(O2, oc20) = {e_o2_oc20:.3f} eV", flush=True)

data = json.loads((HERE / "o_on_ptni_results.json").read_text())
e_o2_omol = data["E_O2"]
shift = 0.5 * e_o2_omol - 0.5 * e_o2_oc20      # add to old (omol-ref) E_ads
print(f"omol 1/2E(O2) = {0.5*e_o2_omol:.3f}; oc20 1/2E(O2) = {0.5*e_o2_oc20:.3f}")

for r in data["results"]:
    r["E_ads"] = r["E_ads"] + shift
data["E_O2_oc20"] = e_o2_oc20
data["reference"] = "1/2 E(O2), oc20 task"
best = min(data["results"], key=lambda r: r["E_ads"])
data["best"]["E_ads"] = best["E_ads"]
data["best"]["type"] = best["type"]
data["best"]["i"] = best["i"]
json.dump(data, open(HERE / "o_on_ptni_results.json", "w"), indent=2)
print(f"Most stable: site {best['i']} ({best['type']}), "
      f"E_ads = {best['E_ads']:+.3f} eV", flush=True)

# --- redraw figure with corrected energies ---------------------------------
import matplotlib.pyplot as plt

eads_all = np.array([r["E_ads"] for r in data["results"]])
types = np.array([r["type"] for r in data["results"]])
colors = {"ontop": "#4c72b0", "bridge": "#dd8452", "hollow": "#55a868"}

fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.25)

ax_h = fig.add_subplot(gs[0])
bins = np.linspace(eads_all.min() - 0.1, eads_all.max() + 0.1, 16)
ax_h.hist([eads_all[types == t] for t in colors],
          bins=bins, stacked=True,
          color=list(colors.values()), label=list(colors.keys()),
          edgecolor="white", lw=0.5)
ax_h.axvline(best["E_ads"], color="k", ls="--", lw=1.2)
ax_h.annotate(f"most stable\n{best['type']}\n{best['E_ads']:+.2f} eV",
              xy=(best["E_ads"], 0), xytext=(0.62, 0.78),
              textcoords="axes fraction", fontsize=9,
              arrowprops=dict(arrowstyle="->", lw=1))
ax_h.set_xlabel(r"$E_{ads}$ (eV)   [ref. $\frac{1}{2}\,$O$_2$]")
ax_h.set_ylabel("count")
ax_h.set_title(f"O on 50:50 Pt-Ni(111):  {len(data['results'])} sites")
ax_h.legend(frameon=False, fontsize=9)

best_atoms = read(HERE / "best_site.traj")
view = best_atoms.repeat((2, 2, 1))
for ax, rot, ttl in ((fig.add_subplot(gs[1]), "0x,0y,0z", "top view"),
                     (fig.add_subplot(gs[2]), "-90x,0y,0z", "side view")):
    plot_atoms(view, ax, rotation=rot, radii=0.9)
    ax.set_title(ttl)
    ax.set_axis_off()
fig.suptitle(f"Most stable O site: {best['type']}, "
             f"$E_{{ads}}$ = {best['E_ads']:+.2f} eV", y=1.02, fontsize=12)
fig.savefig(HERE / "o_on_ptni.png", dpi=150, bbox_inches="tight")
print(f"wrote {HERE / 'o_on_ptni.png'}", flush=True)
