"""Final CO/W(110) MEP figure (no model needed): smooth band + snapshots
showing the C-O bond breaking. Energies read from neb_results_refined.json."""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.visualize.plot import plot_atoms

HERE = Path(__file__).parent
d = json.loads((HERE / "neb_results_refined.json").read_text())
E = np.array(d["E_images"])
rel = E - E[0]
images = read(HERE / "neb_band_refined.traj", index=":")

# reaction coordinate = cumulative displacement along the band
s = [0.0]
for i in range(1, len(images)):
    s.append(s[-1] + np.linalg.norm(images[i].positions - images[i - 1].positions))
s = np.array(s)

# C-O distance along the band (last two atoms are C, O)
nC, nO = len(images[0]) - 2, len(images[0]) - 1
co = [im.get_distance(nC, nO, mic=True) for im in images]
# bond-breaking snapshot: C-O closest to 2.0 A
mid = int(np.argmin([abs(x - 2.0) for x in co]))

fig = plt.figure(figsize=(12, 5.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.5, wspace=0.05)

ax = fig.add_subplot(gs[0, :])
ss = np.linspace(s.min(), s.max(), 300)
ax.plot(ss, np.interp(ss, s, rel), "-", color="#1f77b4", lw=1.5, zorder=1)
ax.plot(s, rel, "o", color="#1f77b4", mec="k", ms=7, zorder=2)
ax.axhline(0, color="0.6", lw=0.7, ls=":")
ax.scatter([s[mid]], [rel[mid]], s=120, facecolors="none",
           edgecolors="crimson", lw=1.5, zorder=3)
ax.set_ylabel("energy [eV]")
ax.set_xlabel("reaction coordinate [Å]")
ax.set_title(f"CO dissociation on W(110): barrierless "
             f"($\\leq$0.05 eV),  $\\Delta E_{{rxn}}$ = {d['dE_rxn']:+.2f} eV   "
             f"(UMA uma-s-1p1 / oc20)")

for col, idx, ttl in [(0, 0, f"initial: CO* (C–O {co[0]:.2f} Å)"),
                      (1, mid, f"breaking: C–O {co[mid]:.2f} Å"),
                      (2, len(images) - 1, f"final: C*+O* (C–O {co[-1]:.2f} Å)")]:
    a = fig.add_subplot(gs[1, col])
    rad = np.array([{"W": 0.85, "C": 0.65, "O": 0.6}[s] for s in
                    images[idx].get_chemical_symbols()])
    plot_atoms(images[idx], a, rotation="-65x,0y,0z", radii=rad)
    a.set_title(ttl, fontsize=9.5)
    a.set_axis_off()

handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=c, mec="k", ms=9, label=l)
           for l, c in (("W", "#2090c0"), ("C", "#404040"), ("O", "#ff2020"))]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))
fig.savefig(HERE / "co_dissociation.png", dpi=150, bbox_inches="tight")
print("wrote", HERE / "co_dissociation.png", "| bond-breaking image", mid)
