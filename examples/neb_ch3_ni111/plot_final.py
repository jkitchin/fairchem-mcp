"""Final CH3->CH2+H / Ni(111) figure: overlay the batch CI-NEB band and the
interactively-driven MCP band (they agree at ~0.62 eV), plus IS/TS/FS."""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.visualize.plot import plot_atoms

HERE = Path(__file__).parent

# batch climbing-image band (has geometries)
batch = json.loads((HERE / "neb_results.json").read_text())
rel_b = np.array(batch["rel"])
iC, iH, ts = batch["iC"], batch["iH"], batch["ts_image"]
band = read(HERE / "neb_band.traj", index=":")

# interactively-driven MCP band (energies only)
E_i = np.array([-185.42726743662118, -185.15226829492806, -184.90031612360238,
                -185.45695150339364, -185.34264123880624, -185.26106870615243,
                -185.4122241779542, -184.81237638437509, -185.2804030224061,
                -185.44902217828988])
rel_i = E_i - E_i[0]

xb = np.linspace(0, 1, len(rel_b))
xi = np.linspace(0, 1, len(rel_i))

fig = plt.figure(figsize=(12, 5.3))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.5, wspace=0.05)
ax = fig.add_subplot(gs[0, :])
ax.plot(xi, rel_i, "s--", color="#dd8452", ms=6, lw=1.2,
        label="interactive MCP NEB (LBFGS↔FIRE, steered)")
ax.plot(xb, rel_b, "o-", color="#1f77b4", ms=6, lw=1.4,
        label="batch climbing-image NEB")
ax.plot(xb[ts], rel_b[ts], "*", color="crimson", ms=18, zorder=3)
ax.axhline(0, color="0.6", lw=0.7, ls=":")
ax.set_ylabel("energy [eV]")
ax.set_xlabel("reaction coordinate (normalised)")
ax.set_title("CH$_3$ → CH$_2$ + H on Ni(111):  $E_a$ ≈ 0.62 eV,  "
             "$\\Delta E_{rxn}$ ≈ −0.02 eV   (UMA uma-s-1p1 / oc20)")
ax.legend(frameon=False, fontsize=9, loc="upper left")

rad = lambda a: np.array([{"Ni": 0.9, "C": 0.7, "H": 0.45}[t] for t in
                          a.get_chemical_symbols()])
IS, FS = band[0], band[-1]
for col, idx, ttl in [(0, 0, f"initial: CH$_3$* (C–H {IS.get_distance(iC,iH,mic=True):.2f} Å)"),
                      (1, ts, f"TS (C–H {band[ts].get_distance(iC,iH,mic=True):.2f} Å)"),
                      (2, len(band)-1, f"final: CH$_2$* + H* (C–H {FS.get_distance(iC,iH,mic=True):.2f} Å)")]:
    a = fig.add_subplot(gs[1, col])
    plot_atoms(band[idx], a, rotation="-65x,0y,0z", radii=rad(band[idx]))
    a.set_title(ttl, fontsize=9.5)
    a.set_axis_off()
handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=c, mec="k", ms=9, label=l)
           for l, c in (("Ni", "#a0a0c0"), ("C", "#404040"), ("H", "#ffffff"))]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, -0.03))
fig.savefig(HERE / "ch3_dehydrogenation.png", dpi=150, bbox_inches="tight")
print("wrote", HERE / "ch3_dehydrogenation.png")
