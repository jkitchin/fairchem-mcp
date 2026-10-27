"""O adsorption-site screen on a 50:50 Pt-Ni fcc(111) slab (UMA / oc20).

Workflow
--------
1. Build a 3x3, 3-layer fcc(111) slab, randomly decorated 14 Pt / 13 Ni
   (27 atoms -> closest possible to 50:50). Bottom layer fixed.
2. Relax the clean slab.
3. Enumerate every high-symmetry adsorption site (ontop / bridge / hollow)
   with pymatgen's AdsorbateSiteFinder.
4. Place an O atom at each site, relax, and compute the adsorption energy
       E_ads = E(slab+O) - E(slab) - 1/2 E(O2).
5. Histogram all the adsorption energies and render the most stable geometry.
"""
import json
import warnings
from pathlib import Path

import numpy as np
from ase import Atom
from ase.build import fcc111, molecule
from ase.constraints import FixAtoms
from ase.io import write
from ase.optimize import LBFGS
from ase.visualize.plot import plot_atoms

from fairchem.core import FAIRChemCalculator, pretrained_mlip
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
MODEL = "uma-s-1p1"
RNG = np.random.default_rng(42)


def log(msg):
    print(msg, flush=True)


def relax(atoms, calc, fmax=0.05, steps=200):
    atoms.calc = calc
    LBFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    return atoms.get_potential_energy()


# --- model / calculators (one resident predictor, two task heads) ----------
log(f"loading {MODEL} ...")
predictor = pretrained_mlip.get_predict_unit(MODEL, device="cpu")
calc_surf = FAIRChemCalculator(predictor, task_name="oc20")   # slab + adsorbate
calc_mol = FAIRChemCalculator(predictor, task_name="omol")    # O2 reference

# --- 1. build the 50:50 Pt-Ni fcc(111) slab -------------------------------
a_vegard = 0.5 * 3.924 + 0.5 * 3.524  # Pt, Ni fcc lattice consts -> 3.724 A
slab = fcc111("Pt", size=(3, 3, 3), a=a_vegard, vacuum=10.0)
slab.pbc = True

n = len(slab)                                   # 27 atoms
syms = np.array(["Pt"] * n)
ni_idx = RNG.permutation(n)[:13]                # 13 Ni / 14 Pt
syms[ni_idx] = "Ni"
slab.set_chemical_symbols(syms.tolist())
log(f"slab composition: {sum(syms=='Pt')} Pt / {sum(syms=='Ni')} Ni "
    f"({100*sum(syms=='Ni')/n:.0f}% Ni)")

# fix the bottom layer
z = slab.positions[:, 2]
bottom = np.where(z < z.min() + 0.5)[0]
slab.set_constraint(FixAtoms(indices=bottom.tolist()))

# --- 2. relax clean slab ---------------------------------------------------
log("relaxing clean slab ...")
e_slab = relax(slab, calc_surf)
log(f"E(clean slab) = {e_slab:.3f} eV")

# --- O2 gas-phase reference (triplet) --------------------------------------
o2 = molecule("O2")
o2.center(vacuum=8.0)
o2.pbc = False
o2.info.update({"spin": 3, "charge": 0})        # triplet ground state
e_o2 = relax(o2, calc_mol, fmax=0.02)
e_o_ref = 0.5 * e_o2
log(f"E(O2) = {e_o2:.3f} eV  ->  1/2 E(O2) = {e_o_ref:.3f} eV")

# --- 3. enumerate adsorption sites -----------------------------------------
struct = AseAtomsAdaptor.get_structure(slab)
asf = AdsorbateSiteFinder(struct)
sites = asf.find_adsorption_sites(
    distance=1.6, symm_reduce=0.01, near_reduce=0.01,
    positions=["ontop", "bridge", "hollow"],
)
site_list = []
for stype in ("ontop", "bridge", "hollow"):
    for pos in sites[stype]:
        site_list.append((stype, np.asarray(pos)))
log(f"enumerated {len(site_list)} candidate sites "
    f"({len(sites['ontop'])} top, {len(sites['bridge'])} bridge, "
    f"{len(sites['hollow'])} hollow)")

# --- 4. relax O at each site -----------------------------------------------
results = []
best = None
for i, (stype, pos) in enumerate(site_list):
    a2 = slab.copy()
    a2.append(Atom("O", position=pos))
    a2.set_constraint(FixAtoms(indices=bottom.tolist()))
    try:
        e = relax(a2, calc_surf)
    except Exception as exc:                    # noqa: BLE001
        log(f"  site {i:2d} ({stype}) FAILED: {exc}")
        continue
    eads = e - e_slab - e_o_ref
    results.append({"i": i, "type": stype, "E_ads": float(eads)})
    log(f"  site {i:2d} ({stype:6s})  E_ads = {eads:+.3f} eV")
    if best is None or eads < best["E_ads"]:
        best = {"i": i, "type": stype, "E_ads": float(eads), "atoms": a2.copy()}

log(f"\nMost stable: site {best['i']} ({best['type']}), "
    f"E_ads = {best['E_ads']:+.3f} eV")

# persist raw data + best structure
json.dump(
    {"E_slab": e_slab, "E_O2": e_o2,
     "n_Pt": int(sum(syms == "Pt")), "n_Ni": int(sum(syms == "Ni")),
     "results": results, "best": {k: best[k] for k in ("i", "type", "E_ads")}},
    open(HERE / "o_on_ptni_results.json", "w"), indent=2,
)
write(HERE / "best_site.traj", best["atoms"])
write(HERE / "best_site.xyz", best["atoms"])

# --- 5. figure: histogram + structure --------------------------------------
import matplotlib.pyplot as plt

eads_all = np.array([r["E_ads"] for r in results])
types = np.array([r["type"] for r in results])
colors = {"ontop": "#4c72b0", "bridge": "#dd8452", "hollow": "#55a868"}

fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.25)

# histogram (stacked by site type)
ax_h = fig.add_subplot(gs[0])
bins = np.linspace(eads_all.min() - 0.1, eads_all.max() + 0.1, 16)
ax_h.hist([eads_all[types == t] for t in colors],
          bins=bins, stacked=True,
          color=list(colors.values()), label=list(colors.keys()),
          edgecolor="white", lw=0.5)
ax_h.axvline(best["E_ads"], color="k", ls="--", lw=1.2)
ax_h.annotate(f"most stable\n{best['type']}\n{best['E_ads']:+.2f} eV",
              xy=(best["E_ads"], 0), xytext=(0.05, 0.78),
              textcoords="axes fraction", fontsize=9,
              arrowprops=dict(arrowstyle="->", lw=1))
ax_h.set_xlabel(r"$E_{ads}$ (eV)   [ref. $\frac{1}{2}\,$O$_2$]")
ax_h.set_ylabel("count")
ax_h.set_title(f"O on 50:50 Pt-Ni(111):  {len(results)} sites")
ax_h.legend(frameon=False, fontsize=9)

# most stable structure: top + side view
best_atoms = best["atoms"]
view = best_atoms.repeat((2, 2, 1))            # tile for periodic context
for ax, rot, ttl in ((fig.add_subplot(gs[1]), "0x,0y,0z", "top view"),
                     (fig.add_subplot(gs[2]), "-90x,0y,0z", "side view")):
    plot_atoms(view, ax, rotation=rot, radii=0.9)
    ax.set_title(ttl)
    ax.set_axis_off()
fig.suptitle(f"Most stable O site: {best['type']}, "
             f"$E_{{ads}}$ = {best['E_ads']:+.2f} eV", y=1.02, fontsize=12)

fig.savefig(HERE / "o_on_ptni.png", dpi=150, bbox_inches="tight")
log(f"wrote {HERE / 'o_on_ptni.png'}")
