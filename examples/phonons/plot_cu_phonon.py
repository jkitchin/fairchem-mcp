"""Plot the phonon dispersion + DOS for fcc Cu (UMA uma-s-1p1, omat task).

Data computed via fairchem-mcp start_phonons (finite displacement, 3x3x3
supercell) and saved to cu_phonon_data.json.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

here = Path(__file__).parent
data = json.loads((here / "cu_phonon_data.json").read_text())

disp = data["dispersion"]
dos = data["dos"]

x = disp["distances"]
bands = disp["frequencies_THz"]  # [n_kpoints][n_branches]
n_branches = len(bands[0])
ticks = disp["high_symmetry_points"]["distances"]
labels = disp["high_symmetry_points"]["labels"]

fig, (ax_bands, ax_dos) = plt.subplots(
    1, 2, figsize=(9, 5), sharey=True,
    gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05},
)

# --- dispersion ---
for b in range(n_branches):
    ax_bands.plot(x, [row[b] for row in bands], color="#1f77b4", lw=1.5)

for xt in ticks:
    ax_bands.axvline(xt, color="0.7", lw=0.6)
ax_bands.axhline(0.0, color="0.4", lw=0.8, ls=":")

ax_bands.set_xticks(ticks)
ax_bands.set_xticklabels(labels)
ax_bands.set_xlim(x[0], x[-1])
ax_bands.set_ylabel("Frequency (THz)")
ax_bands.set_title("fcc Cu phonon dispersion  (UMA uma-s-1p1 / omat)")

# --- DOS ---
ax_dos.plot(dos["dos"], dos["frequencies_THz"], color="#d62728", lw=1.2)
ax_dos.fill_betweenx(dos["frequencies_THz"], 0, dos["dos"],
                     color="#d62728", alpha=0.2)
ax_dos.set_xlabel("DOS")
ax_dos.set_xticks([])
ax_dos.set_title("DOS")

fig.tight_layout()
out = here / "cu_phonon_dispersion.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
