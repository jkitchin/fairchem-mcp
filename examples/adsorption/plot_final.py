"""Render the final figure from the corrected results (no model needed)."""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.visualize.plot import plot_atoms

HERE = Path(__file__).parent
data = json.loads((HERE / "o_on_ptni_results.json").read_text())
results = data["results"]
best = min(results, key=lambda r: r["E_ads"])

eads = np.array([r["E_ads"] for r in results])
types = np.array([r["type"] for r in results])
colors = {"ontop": "#4c72b0", "bridge": "#dd8452", "hollow": "#55a868"}

fig = plt.figure(figsize=(12, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1, 1], wspace=0.2)

# --- histogram ---
ax = fig.add_subplot(gs[0])
bins = np.linspace(eads.min() - 0.1, eads.max() + 0.1, 16)
ax.hist([eads[types == t] for t in colors], bins=bins, stacked=True,
        color=list(colors.values()), label=list(colors.keys()),
        edgecolor="white", lw=0.5)
ax.axvline(best["E_ads"], color="k", ls="--", lw=1.2)
ax.annotate(f"most stable ({best['type']})\n{best['E_ads']:+.2f} eV",
            xy=(best["E_ads"], 0.5), xytext=(best["E_ads"] + 0.35, 6.5),
            fontsize=9, ha="left",
            arrowprops=dict(arrowstyle="->", lw=1))
ax.set_xlabel(r"$E_{\mathrm{ads}}$ (eV)   [ref. $\frac{1}{2}\,$O$_2$]")
ax.set_ylabel("number of sites")
ax.set_title(f"O on 50:50 Pt–Ni(111):  {len(results)} sites screened")
ax.legend(frameon=False, fontsize=9, loc="upper right")

# --- most stable structure (single cell -> exactly one O) ---
best_atoms = read(HERE / "best_site.traj")
for ax2, rot, ttl in ((fig.add_subplot(gs[1]), "0x,0y,0z", "top view"),
                      (fig.add_subplot(gs[2]), "-75x,0y,0z", "side view")):
    plot_atoms(best_atoms, ax2, rotation=rot, radii=0.9)
    ax2.set_title(ttl)
    ax2.set_axis_off()

# simple legend for atom colors
handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=c, mec="k", ms=10, label=l)
           for l, c in (("Pt", "#d0d0d0"), ("Ni", "#50d050"), ("O", "#ff2020"))]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           fontsize=9, bbox_to_anchor=(0.72, -0.02))

fig.suptitle(f"Most stable O adsorption site:  {best['type']},  "
             f"$E_{{\\mathrm{{ads}}}}$ = {best['E_ads']:+.2f} eV  (UMA uma-s-1p1 / oc20)",
             y=1.0, fontsize=12)
fig.savefig(HERE / "o_on_ptni.png", dpi=150, bbox_inches="tight")
print("wrote", HERE / "o_on_ptni.png")
